
import numpy as np

__all__ = [
    "wpercentile",
    "wmedian"
]


# def _wpercentile1d(data, weights, percentiles):

#     if data.ndim > 1 or weights.ndim > 1:
#         raise ValueError("data and weights must be one-dimensional arrays")

#     if data.shape != weights.shape:
#         raise ValueError("data and weights must have the same shape")

#     data = np.asarray(data)
#     weights = np.asarray(weights)
#     percentiles = np.asarray(percentiles)

#     sort_indices = np.argsort(data)
#     sorted_data = data[sort_indices]
#     sorted_weights = weights[sort_indices]

#     cumsum_weights = np.cumsum(sorted_weights)
#     sum_weights = cumsum_weights[-1]

#     pn = 100 * (cumsum_weights - 0.5*sorted_weights) / sum_weights

#     return np.interp(percentiles, pn, sorted_data)


# def wpercentile_unsafe(data, weights, percentiles, axis=None):
#     """
#     Compute percentiles of a weighted sample.
#     """

#     # TODO: This code might fail with esoteric shapes of data, weights and
#     # for some values of axis. Check this properly.

#     if axis is None:
#         data = np.ravel(data)
#         weights = np.ravel(weights)
#         return _wpercentile1d(data, weights, percentiles)

#     axis = np.atleast_1d(axis)

#     # Reshape the arrays for proper computation
#     # Move the requested axis to the final dimensions
#     dest_axis = list(range(len(axis)))
#     data2 = np.moveaxis(data, axis, dest_axis)

#     ndim = len(axis)
#     shape = data2.shape
#     newshape = (np.prod(shape[:ndim]), np.prod(shape[ndim:]))
#     newdata = np.reshape(data2, newshape)
#     newweights = np.reshape(weights, newshape[0])

#     result = np.apply_along_axis(_wpercentile1d, 0, newdata, newweights,
#                                  percentiles)

#     final_shape = (*np.shape(percentiles), *shape[ndim:])
#     return np.reshape(result, final_shape)


def wpercentile(data, weights, percentiles, axis=None):

    # This is a simple, error-free but pretty inefficient implementation of
    # wpercentile.

    data = np.repeat(data, weights, axis=axis)
    return np.percentile(data, percentiles, axis=axis)


def wmedian(data, weights, axis=None):
    """Compute the weighted median."""
    return wpercentile(data, weights, 50, axis)
