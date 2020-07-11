
import logging
from typing import Optional, Union, Dict, Iterable, List

import numpy as np

from sklearn import ensemble
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler

from .utils import tqdm
from .posterior import Posterior, posterior_percentile

__all__ = [
    "Model",
]

LOGGER = logging.getLogger(__name__)


class Model(BaseEstimator):

    def __init__(
            self,
            n_estimators: int = 1000,
            criterion: str = 'mse',
            max_features: Union[str, int, float] = 'sqrt',
            min_impurity_decrease: float = 0.0,
            bootstrap: bool = True,
            n_jobs: Optional[int] = None,
            random_state: Union[int, np.random.RandomState, None] = None,
            verbose: int = 0,
            max_samples: Union[float, int, None] = None,
            enable_posterior: bool = True
            ):

        scaler = MinMaxScaler(feature_range=(0, 100))
        self.random_forest = ensemble.RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            max_samples=max_samples
        )

        self.scaler = scaler

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

        if len(x) > self.random_forest.n_estimators:
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


def _posterior(
        data_leaves: np.ndarray,
        data_weights: np.ndarray,
        data_y: np.ndarray,
        query_leaves: np.ndarray
        ) -> Posterior:

    weights_x = (query_leaves[:, None] == data_leaves) * data_weights
    weights_x = weights_x.sum(0)

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
    # This can be parallelized with multiprocessing.
    for leaves_x_i in tqdm(query_leaves):
        posterior = _posterior(
            data_leaves, data_weights,
            prior_samples, leaves_x_i
        )
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
    # This can be parallelized with multiprocessing.
    for leaves_x_i in tqdm(query_leaves):
        indices: List[int] = []
        weights: List[int] = []
        for tree, leaves_x_i_j in enumerate(leaves_x_i):
            cur_indices = cache[tree][0][leaves_x_i_j]
            cur_weights = cache[tree][1][leaves_x_i_j]
            indices.extend(cur_indices)
            weights.extend(cur_weights)

        posterior = Posterior(prior_samples[indices], weights)
        value = posterior_percentile(posterior, percentile)
        values.append(value)

    return np.array(values)


def _build_leaves_cache(data_leaves, data_weights):

    indices: Dict[int, List[int]] = {}
    weights: Dict[int, List[int]] = {}

    for index, (leaf, weight) in enumerate(zip(data_leaves, data_weights)):
        if weight == 0:
            continue

        if leaf not in indices:
            indices[leaf] = [index]
            weights[leaf] = [weight]
        else:
            indices[leaf].append(index)
            weights[leaf].append(weight)

    return indices, weights


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
