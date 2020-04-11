
import logging
from collections import namedtuple
from typing import Optional, Union, Iterable, List, Tuple

import numpy as np

from sklearn import ensemble
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler

from .utils import tqdm
from .wpercentile import wpercentile

__all__ = [
    "Model",
    "Posterior",
    "resample_posterior",
    "posterior_percentile"
]

LOGGER = logging.getLogger(__name__)

# Posteriors are represented as a collection of weighted samples
Posterior = namedtuple("Posterior", ["samples", "weights"])


def resample_posterior(posterior: Posterior, num_draws: int) -> Posterior:

    p = posterior.weights / posterior.weights.sum()
    indices = np.random.choice(len(posterior.samples), size=num_draws, p=p)

    new_weights = np.bincount(indices, minlength=len(posterior.samples))
    mask = new_weights != 0
    new_samples = posterior.samples[mask]
    new_weights = posterior.weights[mask]

    return Posterior(new_samples, new_weights)


def posterior_percentile(
        posterior: Posterior,
        percentiles: Union[float, Iterable[float]]) -> np.ndarray:

    samples, weights = posterior
    return wpercentile(samples, weights, percentiles, axis=0)


class Model:

    def __init__(
            self,
            num_trees: int,
            num_jobs: int,
            names: Iterable[str],
            ranges: Iterable[Tuple[float, float]],
            colors: Iterable[str],
            enable_posterior: bool = True,
            verbose: int = 1):

        scaler = MinMaxScaler(feature_range=(0, 100))
        self.random_forest = ensemble.RandomForestRegressor(
            n_estimators=num_trees,
            oob_score=True,
            verbose=verbose,
            n_jobs=num_jobs,
            max_features="sqrt",
            min_impurity_decrease=0.01
        )

        self.scaler = scaler

        self.num_trees = num_trees
        self.num_jobs = num_jobs
        self.verbose = verbose

        self.ranges = ranges
        self.names = names
        self.colors = colors

        # To compute the posteriors
        self.enable_posterior = enable_posterior
        self.data_leaves = None
        self.data_weights = None
        self.data_y = None

    def _scaler_fit(self, y):
        if y.ndim == 1:
            y = y[:, None]

        self.scaler.fit(y)

    def scaler_transform(self, y: np.ndarray) -> np.ndarray:
        if y.ndim == 1:
            y = y[:, None]
            return self.scaler.transform(y)[:, 0]

        return self.scaler.transform(y)

    def scaler_inverse_transform(self, y: np.ndarray) -> np.ndarray:

        if y.ndim == 1:
            y = y[:, None]
            # return self.scaler.inverse_transform(y)[:, 0]

        return self.scaler.inverse_transform(y)

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._scaler_fit(y)
        self.random_forest.fit(x, self.scaler_transform(y))

        # Build the structures to quickly compute the posteriors
        if self.enable_posterior:
            data_leaves = self.random_forest.apply(x).T
            self.data_leaves = _as_smallest_udtype(data_leaves)
            self.data_weights = np.array(
                [_tree_weights(tree, len(y)) for tree in self.random_forest]
            )
            self.data_y = y

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = self.random_forest.predict(x)
        return self.scaler_inverse_transform(pred)

    def predict_median(
            self,
            x: np.ndarray,
            prior_samples: Optional[np.ndarray] = None
            ) -> np.ndarray:

        return self.predict_percentile(x, 50, prior_samples)

    def predict_percentile(
            self,
            x: np.ndarray,
            percentile: Union[float, Iterable[float]],
            prior_samples: Optional[np.ndarray] = None
            ) -> np.ndarray:
        """
        Equivalent to  compute the posterior for each element of `x` and then
        calling `posterior_percentile`. However, this is usually more efficient
        than the naive approach when `x` has many elements.
        """

        if not self.enable_posterior:
            raise ValueError("Cannot compute posteriors with this model. "
                             "Set `enable_posterior` to True to enable "
                             "posterior computation.")

        # Find the leaves for the query points
        leaves_x = self.random_forest.apply(x)

        if prior_samples is None:
            prior_samples = self.data_y

        if len(x) > self.num_trees:
            # If there are many queries,
            # it is faster to find points using a cache
            return _posterior_percentile_cache(
                self.data_leaves, self.data_weights,
                prior_samples, leaves_x, percentile
            )

        # For few queries, it is faster if we just compute the posterior
        # for each element
        return _posterior_percentile_nocache(
            self.data_leaves, self.data_weights,
            prior_samples, leaves_x, percentile
        )

    def posterior(
            self,
            x: np.ndarray,
            prior_samples: Optional[np.ndarray] = None
            ) -> Posterior:

        if not self.enable_posterior:
            raise ValueError("Cannot compute posteriors with this model. "
                             "Set `enable_posterior` to True to enable "
                             "posterior computation.")

        if x.ndim > 1:
            raise ValueError("x.ndim must be 1")

        leaves_x = self.random_forest.apply(x[None, :])[0]

        if prior_samples is None:
            prior_samples = self.data_y

        return _posterior(
            self.data_leaves, self.data_weights,
            prior_samples, leaves_x
        )

    def get_params(self, deep=True):
        return {
            "num_trees": self.num_trees,
            "num_jobs": self.num_jobs,
            "names": self.names,
            "ranges": self.ranges,
            "colors": self.colors,
            "enable_posterior": self.enable_posterior,
            "verbose": self.verbose
        }


def _posterior(
        data_leaves: np.ndarray,
        data_weights: np.ndarray,
        data_y: np.ndarray,
        query_leaves: np.ndarray
        ) -> Posterior:

    weights_x = (query_leaves[:, None] == data_leaves) * data_weights
    weights_x = _as_smallest_udtype(weights_x.sum(0))

    # Remove samples with weight zero
    mask = weights_x != 0
    samples = data_y[mask]
    weights = weights_x[mask].astype(np.int_)

    return Posterior(samples, weights)


def _posterior_percentile_nocache(
        data_leaves: np.ndarray,
        data_weights: np.ndarray,
        prior_samples: np.ndarray,
        query_leaves: np.ndarray,
        percentile: Union[float, Iterable[float]]
        ) -> np.ndarray:

    values = []

    LOGGER.info("Computing percentiles...")
    for leaves_x_i in tqdm(query_leaves):
        posterior = _posterior(
            data_leaves, data_weights,
            prior_samples, leaves_x_i
        )
        # samples = np.repeat(posterior.samples, posterior.weights, axis=0)
        # value = np.percentile(samples, percentile, axis=0)
        value = posterior_percentile(posterior, percentile)
        values.append(value)

    return np.array(values)


def _posterior_percentile_cache(
        data_leaves: np.ndarray,
        data_weights: np.ndarray,
        prior_samples: np.ndarray,
        query_leaves: np.ndarray,
        percentile: Union[float, Iterable[float]]
        ) -> np.ndarray:

    # Build a dictionary for fast access of the contents of the leaves.
    LOGGER.info("Building cache...")
    cache = [
        _build_leaves_cache(leaves_i, weights_i)
        for leaves_i, weights_i in zip(data_leaves, data_weights)
    ]

    values = []
    # Check the contents of the leaves in leaves_x
    LOGGER.info("Computing percentiles...")
    for leaves_x_i in tqdm(query_leaves):
        data_elements: List[int] = []
        for tree, leaves_x_i_j in enumerate(leaves_x_i):
            aux = cache[tree][leaves_x_i_j]
            data_elements.extend(aux)
        value = np.percentile(prior_samples[data_elements], percentile, axis=0)
        values.append(value)

    return np.array(values)


def _build_leaves_cache(leaves, weights):

    result = {}
    for index, (leaf, weight) in enumerate(zip(leaves, weights)):
        if weight == 0:
            continue

        if leaf not in result:
            result[leaf] = [index] * weight
        else:
            result[leaf].extend([index] * weight)

    return result


def _generate_sample_indices(random_state, n_samples):
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices


def _tree_weights(tree, n_samples):
    indices = _generate_sample_indices(tree.random_state, n_samples)
    res = np.bincount(indices, minlength=n_samples)
    return _as_smallest_udtype(res)


def _as_smallest_udtype(arr):
    return arr.astype(_smallest_udtype(arr.max()))


def _smallest_udtype(value):

    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]

    for dtype in dtypes:
        if value <= np.iinfo(dtype).max:
            return dtype

    raise ValueError("value is too large for any dtype")
