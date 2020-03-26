
from itertools import product

import numpy as np
from matplotlib.colors import to_rgba_array
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn import metrics, neighbors
from sklearn.preprocessing import MinMaxScaler

from models import resample_posterior
from wpercentile import wmedian
from utils import tqdm

__all__ = [
    'predicted_vs_real',
    'feature_importances',
    'stacked_feature_importances',
    'posterior_matrix'
]

POSTERIOR_MAX_SIZE = 10000


def predicted_vs_real(y_real, y_pred, names, ranges, alpha='auto'):

    num_plots = y_pred.shape[1]
    num_plot_rows = int(np.sqrt(num_plots))
    num_plot_cols = (num_plots - 1) // num_plot_rows + 1

    fig, axes = plt.subplots(num_plot_rows, num_plot_cols,
                             figsize=(5*num_plot_cols, 5*num_plot_rows),
                             squeeze=False)

    for dim, (ax, name_i, range_i) in enumerate(zip(axes.ravel(),
                                                    names, ranges)):

        current_real = y_real[:, dim]
        current_pred = y_pred[:, dim]

        if alpha == 'auto':
            # TODO: this is a quick fix. Check at some point in the future.
            aux, *_ = np.histogram2d(current_real, current_pred, bins=60)
            alpha_ = 1 / np.percentile(aux[aux > 0], 60)
        elif alpha == 'none':
            alpha_ = None
        else:
            alpha_ = alpha

        r2 = metrics.r2_score(current_real, current_pred)
        label = "$R^2 = {:.3f}$".format(r2)
        ax.plot(current_real, current_pred, '.', label=label, alpha=alpha_)

        ax.plot(range_i, range_i, '--', linewidth=3, color="C3", alpha=0.8)

        ax.axis("equal")
        ax.grid()
        ax.set_xlim(range_i)
        ax.set_ylim(range_i)
        ax.set_xlabel("Real {}".format(name_i), fontsize=18)
        ax.set_ylabel("Predicted {}".format(name_i), fontsize=18)
        ax.legend(loc="upper left", fontsize=14)

    fig.tight_layout()
    return fig


def feature_importances(forests, names, colors):

    num_plots = len(forests)
    num_plot_rows = (num_plots - 1) // 2 + 1
    num_plot_cols = 2

    fig, axes = plt.subplots(num_plot_rows, num_plot_cols,
                             figsize=(15, 3.5*num_plot_rows))

    for ax, forest_i, name_i, color_i in zip(axes.ravel(), forests,
                                             names, colors):
        ax.bar(
            np.arange(len(forest_i.feature_importances_)),
            forest_i.feature_importances_,
            label="Importance for {}".format(name_i),
            width=0.4,
            color=color_i
        )
        ax.set_xlabel("Feature index", fontsize=18)
        ax.legend(fontsize=16)
        ax.grid()

    fig.tight_layout()
    return fig


def stacked_feature_importances(importances, names, colors):

    bottoms = np.zeros(importances.shape[0])

    ind = np.arange(len(importances))

    fig = plt.figure()
    ax = fig.gca()
    for data_i, name_i, color_i in zip(importances.T, names, colors):
        ax.bar(ind, data_i, bottom=bottoms, color=color_i, label=name_i)
        bottoms += data_i

    ax.set_xlabel("Feature index", fontsize=18)
    ax.legend(fontsize=16)
    ax.grid()

    fig.tight_layout()
    return fig


def posterior_matrix(posterior, names, ranges, colors, soft_colors=None):

    samples, weights = posterior

    cmaps = [LinearSegmentedColormap.from_list("MyReds", [(1, 1, 1), c], N=256)
             for c in colors]

    ranges = np.array(ranges)

    if soft_colors is None:
        soft_colors = colors

    num_dims = samples.shape[1]

    fig, axes = plt.subplots(nrows=num_dims, ncols=num_dims,
                             figsize=(2 * num_dims, 2 * num_dims))
    fig.subplots_adjust(left=0.07, right=1-0.05,
                        bottom=0.07, top=1-0.05,
                        hspace=0.05, wspace=0.05)

    iterable = zip(axes.flat, product(range(num_dims), range(num_dims)))
    for ax, dims in tqdm(iterable, total=num_dims**2):
        # Flip dims.
        dims = [dims[1], dims[0]]

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.title.set_visible(False)
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
            ax.yaxis.set_visible(True)
            if names is not None:
                ax.set_ylabel(names[dims[1]], fontsize=18)
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_visible(True)
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_visible(True)
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_visible(True)
            if names is not None:
                ax.set_xlabel(names[dims[0]], fontsize=18)
        if ax.is_first_col() and ax.is_first_row():
            ax.yaxis.set_visible(False)
            ax.set_ylabel("")
        if ax.is_last_col() and ax.is_last_row():
            ax.yaxis.set_visible(False)

        if dims[0] < dims[1]:
            _plot_histogram2d(
                ax, posterior,
                color=colors[dims[0]],
                cmap=cmaps[dims[0]],
                dims=dims,
                ranges=ranges[dims]
            )
        elif dims[0] > dims[1]:
            _plot_samples(
                ax, posterior,
                color=soft_colors[dims[1]],
                dims=dims,
                ranges=ranges[dims]
            )
        else:
            histogram, bins = _histogram1d(
                samples[:, dims[:1]], weights,
                ranges=ranges[dims[:1]]
            )
            ax.bar(
                bins[:-1],
                histogram,
                color=soft_colors[dims[0]],
                width=bins[1]-bins[0]
            )

            kd_probs = histogram
            expected = wmedian(samples[:, dims[0]], weights)
            ax.plot(
                [expected, expected], [0, 1.1 * kd_probs.max()], '-',
                linewidth=1,
                color='#222222'
            )

            ax.axis([ranges[dims[0]][0], ranges[dims[0]][1],
                     0, 1.1 * kd_probs.max()])

    # fig.tight_layout(pad=0)
    return fig


def _plot_histogram2d(ax, posterior, color, cmap, dims, ranges):

    samples, weights = posterior
    # For efficiency, do not compute the kernel density
    # over all the samples of the posterior. Subsample first.
    if len(samples) > POSTERIOR_MAX_SIZE:
        samples, weights = resample_posterior(posterior, POSTERIOR_MAX_SIZE)

    locations, kd_probs, *_ = _kernel_density_joint(
        samples[:, dims],
        weights,
        ranges
    )
    ax.contour(
        locations[0], locations[1],
        kd_probs,
        colors=color,
        linewidths=0.5
    )

    # For the rest of the plot we use the complete posterior
    samples, weights = posterior
    histogram, grid_x, grid_y = _histogram2d(
        samples[:, dims], weights,
        ranges
    )
    ax.pcolormesh(grid_x, grid_y, histogram, cmap=cmap)

    expected = wmedian(samples[:, dims], weights, axis=0)
    ax.plot([expected[0], expected[0]], [ranges[1][0], ranges[1][1]],
            '-', linewidth=1, color='#222222')
    ax.plot([ranges[0][0], ranges[0][1]], [expected[1], expected[1]],
            '-', linewidth=1, color='#222222')
    ax.plot(expected[0], expected[1], '.', color='#222222')
    ax.axis('auto')
    ax.axis([ranges[0][0], ranges[0][1],
             ranges[1][0], ranges[1][1]])


def _plot_samples(ax, posterior, color, dims, ranges):

    # For efficiency, do not plot all the samples of the posterior.
    # Subsample first.
    if len(posterior.samples) > POSTERIOR_MAX_SIZE:
        posterior = resample_posterior(posterior, POSTERIOR_MAX_SIZE)

    samples, weights = posterior

    points_alpha = _weights_to_alpha(weights)

    current_colors = to_rgba_array(color)
    current_colors = np.tile(current_colors, (len(samples), 1))
    current_colors[:, 3] = points_alpha

    ax.scatter(
        x=samples[:, dims[0]],
        y=samples[:, dims[1]],
        s=100,
        c=current_colors,
        marker='.',
        linewidth=0
    )

    ax.axis([ranges[0][0], ranges[0][1],
             ranges[1][0], ranges[1][1]])


def _min_max_scaler(ranges, feature_range=(0, 100)):
    res = MinMaxScaler()
    res.data_max_ = ranges[:, 1]
    res.data_min_ = ranges[:, 0]
    res.data_range_ = res.data_max_ - res.data_min_
    res.scale_ = ((feature_range[1] - feature_range[0]) /
                  (ranges[:, 1] - ranges[:, 0]))
    res.min_ = -res.scale_ * res.data_min_
    res.n_samples_seen_ = 1
    res.feature_range = feature_range
    return res


def _kernel_density_joint(samples, weights, ranges, bandwidth=1/25):

    ndims = len(ranges)

    scaler = _min_max_scaler(ranges, feature_range=(0, 100))

    bandwidth = bandwidth * 100
    # step = 1.0

    kd = neighbors.KernelDensity(bandwidth=bandwidth)
    kd.fit(scaler.transform(samples), sample_weight=weights)

    grid_shape = [100] * ndims
    grid = np.indices(grid_shape)
    locations = np.reshape(grid, (ndims, -1)).T
    kd_probs = np.exp(kd.score_samples(locations))

    shape = (ndims, *grid_shape)
    locations = scaler.inverse_transform(locations)
    locations = np.reshape(locations.T, shape)
    kd_probs = np.reshape(kd_probs, grid_shape)
    return locations, kd_probs, kd


def _histogram1d(samples, weights, ranges, bins=20):

    assert len(ranges) == 1

    histogram, edges = np.histogram(
        samples[:, 0],
        bins=bins,
        range=ranges[0],
        weights=weights.astype(np.uint)
    )
    return histogram, edges


def _histogram2d(samples, weights, ranges, bins=20):

    assert len(ranges) == 2

    histogram, xedges, yedges = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=bins,
        range=ranges,
        weights=weights
    )
    grid_x, grid_y = np.meshgrid(xedges, yedges)

    return histogram.T, grid_x, grid_y


def _weights_to_alpha(weights):

    # Maximum weight (removing potential outliers)
    max_weight = np.percentile(weights, 98)
    return np.clip(weights / max_weight, 0, 1)
