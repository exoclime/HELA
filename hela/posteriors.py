import numpy as np

from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state

from tqdm import tqdm

from .wpercentile import wpercentile

__all__ = ['PosteriorRandomForest', 'Posterior', 'resample_posterior']


class PosteriorRandomForest(object):
    """
    Produces posterior samples from a random forest.
    """
    def __init__(self, num_trees, num_jobs, names, ranges, colors,
                 verbose=1, enable_posterior=True):
        """
        Parameters
        ----------
        num_trees : int
        num_jobs : int
        names : list
        ranges : list
        colors : list
        verbose : bool or int
        enable_posterior : bool
        """
        scaler = MinMaxScaler(feature_range=(0, 100))
        rf = ensemble.RandomForestRegressor(n_estimators=num_trees,
                                            oob_score=True,
                                            verbose=verbose,
                                            n_jobs=num_jobs,
                                            max_features="sqrt",
                                            min_impurity_decrease=0.01)

        self.scaler = scaler
        self.rf = rf

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

    def _scaler_transform(self, y):
        if y.ndim == 1:
            y = y[:, None]
            return self.scaler.transform(y)[:, 0]

        return self.scaler.transform(y)

    def _scaler_inverse_transform(self, y):

        if y.ndim == 1:
            y = y[:, None]
            # return self.scaler.inverse_transform(y)[:, 0]

        return self.scaler.inverse_transform(y)

    def fit(self, x, y):
        """
        Fit the model.

        Follows scikit-learn convention.

        Parameters
        ----------
        x : `~numpy.ndarray`
        y : `~numpy.ndarray`
        """
        self._scaler_fit(y)
        self.rf.fit(x, self._scaler_transform(y))

        # Build the structures to quickly compute the posteriors
        if self.enable_posterior:
            data_leaves = self.rf.apply(x).T
            self.data_leaves = _as_smallest_udtype(data_leaves)
            self.data_weights = np.array(
                [_tree_weights(tree, len(y)) for tree in self.rf])
            self.data_y = y

    def predict(self, x):
        """
        Predict on values of ``x``

        Follows scikit-learn convention.

        Parameters
        ----------
        x : `~numpy.ndarray`
        """
        pred = self.rf.predict(x)
        return self._scaler_inverse_transform(pred)

    def get_params(self):
        return {"num_trees": self.num_trees, "num_jobs": self.num_jobs,
                "names": self.names, "ranges": self.ranges,
                "colors": self.colors, "verbose": self.verbose,
                "enable_posterior": self.enable_posterior}

    def trees_predict(self, x):

        if x.ndim > 1:
            raise ValueError("x.ndim must be 1")

        preds = np.array([i.predict(x[None, :])[0]
                          for i in self.rf.estimators_])
        return self._scaler_inverse_transform(preds)

    def predict_median(self, x):
        return self.predict_percentile(x, 50)

    def predict_percentile(self, x, percentile):

        if not self.enable_posterior:
            raise ValueError("Cannot compute posteriors with this model. "
                             "Set `enable_posterior` to True to enable "
                             "posterior computation.")

        # Find the leaves for the query points
        leaves_x = self.rf.apply(x)

        if len(x) > self.num_trees:
            # If there are many queries, it is faster to find points
            # using a cache
            return _posterior_percentile_cache(
                self.data_leaves, self.data_weights,
                self.data_y, leaves_x, percentile
            )
        else:
            # For few queries, it is faster if we just compute the posterior
            # for each element
            return _posterior_percentile_nocache(
                self.data_leaves, self.data_weights,
                self.data_y, leaves_x, percentile
            )

    def predict_posterior(self, x):
        leaves_x = self.rf.apply(x[None, :])[0]
        if not self.enable_posterior:
            raise ValueError("Cannot compute posteriors with this model. "
                             "Set `enable_posterior` to True to enable "
                             "posterior computation.")

        return _posterior(
            self.data_leaves, self.data_weights,
            self.data_y, leaves_x
        )


class Posterior(object):
    """
    Posteriors are represented as a collection of weighted samples
    """

    def __init__(self, samples, weights):
        """
        Parameters
        ----------
        samples : `~numpy.ndarray`
        weights : `~numpy.ndarray`
        """
        self.samples = samples
        self.weights = weights

    def print_percentiles(self, names):
        """
        Print the median and the +/- 1 sigma posterior values.

        Parameters
        ----------
        names

        Returns
        -------

        """
        posterior_ranges = self.data_ranges()
        for name_i, pred_range_i in zip(names, posterior_ranges):
            print("Prediction for {}: {:.3g} "
                  "[+{:.3g} -{:.3g}]".format(name_i, *pred_range_i))


    def data_ranges(self, percentiles=(50, 16, 84)):
        """
        Return posterior ranges.

        Parameters
        ----------
        posterior : `~numpy.ndarray`
        percentiles : tuple

        Returns
        -------
        ranges : `~numpy.ndarray`
        """
        values = wpercentile(self.samples, self.weights,
                             percentiles, axis=0)
        ranges = np.array(
            [values[0], values[2] - values[0], values[0] - values[1]])
        return ranges.T

    def plot_posterior_matrix(self, dataset):
        from .plot import plot_posterior_matrix
        return plot_posterior_matrix(self, dataset)

def _posterior(data_leaves, data_weights, data_y, query_leaves):
    weights_x = (query_leaves[:, None] == data_leaves) * data_weights
    weights_x = _as_smallest_udtype(weights_x.sum(0))

    # Remove samples with weight zero
    mask = weights_x != 0
    samples = data_y[mask]
    weights = weights_x[mask]

    return Posterior(samples, weights)


def _posterior_percentile_nocache(data_leaves, data_weights, data_y,
                                  query_leaves, percentile):
    values = []

    # Computing percentiles...
    for leaves_x_i in tqdm(query_leaves):
        posterior = _posterior(
            data_leaves, data_weights,
            data_y, leaves_x_i
        )
        samples = np.repeat(posterior.samples, posterior.weights, axis=0)
        value = np.percentile(samples, percentile, axis=0)
        values.append(value)

    return np.array(values)


def _posterior_percentile_cache(data_leaves, data_weights, data_y,
                                query_leaves, percentile):
    # Build a dictionary for fast access of the contents of the leaves.
    # Building cache...
    cache = [_build_leaves_cache(leaves_i, weights_i)
             for leaves_i, weights_i in zip(data_leaves, data_weights)]

    values = []
    # Check the contents of the leaves in leaves_x
    # Computing percentiles...
    for leaves_x_i in tqdm(query_leaves):
        data_elements = []
        for tree, leaves_x_i_j in enumerate(leaves_x_i):
            aux = cache[tree][leaves_x_i_j]
            data_elements.extend(aux)
        value = np.percentile(data_y[data_elements], percentile, axis=0)
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


def resample_posterior(posterior, num_draws):
    p = posterior.weights / posterior.weights.sum()
    indices = np.random.choice(len(posterior.samples), size=num_draws, p=p)

    new_weights = np.bincount(indices, minlength=len(posterior.samples))
    mask = new_weights != 0
    new_samples = posterior.samples[mask]
    new_weights = posterior.weights[mask]

    return Posterior(new_samples, new_weights)


def _as_smallest_udtype(arr):
    return arr.astype(_smallest_udtype(arr.max()))


def _smallest_udtype(value):
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]

    for dtype in dtypes:
        if value <= np.iinfo(dtype).max:
            return dtype

    raise ValueError("value is too large for any dtype")
