import os
import json

import numpy as np
from sklearn import metrics, multioutput
import joblib

from .dataset import load_dataset, load_data_file
from .wrapper import RandomForestWrapper
from .plot import (plot_predicted_vs_real, plot_feature_importances,
                   plot_posterior_matrix)
from .wpercentile import wpercentile

__all__ = ['Retrieval', 'generate_example_data']


def train_model(dataset, num_trees, num_jobs, verbose=1):
    pipeline = RandomForestWrapper(num_trees, num_jobs,
                                   names=dataset.names,
                                   ranges=dataset.ranges,
                                   colors=dataset.colors,
                                   verbose=verbose)
    pipeline.fit(dataset.training_x, dataset.training_y)
    return pipeline


def test_model(model, dataset):
    if dataset.testing_x is None:
        return

    pred = model.predict(dataset.testing_x)
    r2scores = {name_i: metrics.r2_score(real_i, pred_i)
                for name_i, real_i, pred_i in
                zip(dataset.names, dataset.testing_y.T, pred.T)}
    print("Testing scores:")
    for name, values in r2scores.items():
        print("\tR^2 score for {}: {:.3f}".format(name, values))

    return pred, r2scores


def compute_feature_importance(model, dataset):
    regr = multioutput.MultiOutputRegressor(model, n_jobs=1)
    regr.fit(dataset.training_x, dataset.training_y)

    forests = [i.rf for i in regr.estimators_] + [model.rf]
    return np.array([forest_i.feature_importances_ for forest_i in forests])


class Retrieval(object):
    """
    A class for a trainable random forest model.
    """

    def __init__(self, training_dataset, model_path, data_file):
        """
        Parameters
        ----------
        training_dataset : str
            Path to the dataset metadata JSON file
        model_path : str
            Path to the output directory to create and populate
        data_file : str
            Path to the numpy pickle of the samples to predict on
        """
        self.training_dataset = training_dataset
        self.model_path = model_path
        self.data_file = data_file
        self.output_path = self.model_path

        self.dataset = None
        self.model = None
        self._feature_importance = None
        self._posterior = None
        self.oob = None
        self.pred = None

    def train(self, num_trees=1000, num_jobs=5, quiet=False):
        """
        Train the random forest on a set of observations.

        Parameters
        ----------
        num_trees : int
        num_jobs : int
        quiet : bool

        Returns
        -------
        r2scores : dict
            :math:`R^2` values for each parameter after training
        """
        # Loading dataset
        self.dataset = load_dataset(self.training_dataset)

        # Training model
        self.model = train_model(self.dataset, num_trees, num_jobs, not quiet)

        os.makedirs(self.model_path, exist_ok=True)
        model_file = os.path.join(self.model_path, "model.pkl")

        # Saving model
        joblib.dump(self.model, model_file)

        # Printing model information...
        print("OOB score: {:.4f}".format(self.model.rf.oob_score_))
        self.oob = self.model.rf.oob_score_

        pred, r2scores = test_model(self.model, self.dataset)
        self.pred = pred
        return r2scores

    def plot_correlations(self):
        """
        Plot training correlations.

        Returns
        -------
        fig, axes
        """
        fig, axes = plot_predicted_vs_real(self.dataset.testing_y, self.pred,
                                           self.dataset.names,
                                           self.dataset.ranges)
        fig.savefig(os.path.join(self.output_path, "predicted_vs_real.pdf"),
                    bbox_inches='tight')
        return fig, axes

    def feature_importance(self):
        """
        Compute feature importance.

        Returns
        -------
        feature_importances : `~numpy.ndarray`
        """
        if self._feature_importance is None:
            self._feature_importance = compute_feature_importance(self.model,
                                                                  self.dataset)
        return self._feature_importance

    def plot_feature_importance(self):
        """
        Plot the feature importances.

        Returns
        -------
        fig, axes
        """
        forests = self.feature_importance()
        fig, axes = plot_feature_importances(forests=forests,
                                             names=(self.dataset.names +
                                                    ["joint prediction"]),
                                             colors=(self.dataset.colors +
                                                     ["C0"]))

        fig.savefig(os.path.join(self.output_path, "feature_importances.pdf"),
                    bbox_inches='tight')
        return fig, axes

    def predict(self, quiet=False):
        """
        Predict values from the trained random forest.

        Parameters
        ----------
        plot_posterior : bool

        Returns
        -------
        preds : `~numpy.ndarray`
            ``N x M`` values where ``N`` is number of parameters, ``M`` is
            number of samples/trees (check out attributes of model for
            metadata)
        """
        if self._posterior is None:
            model_file = os.path.join(self.model_path, "model.pkl")
            # Loading random forest from '{}'...".format(model_file)
            model = joblib.load(model_file)

            # Loading data from '{}'...".format(data_file)
            data, _ = load_data_file(self.data_file, model.rf.n_features_)

            posterior = model.posterior(data[0])

            if not quiet:
                posterior_ranges = data_ranges(posterior)
                for name_i, pred_range_i in zip(model.names, posterior_ranges):
                    print("Prediction for {}: {:.3g} "
                          "[+{:.3g} -{:.3g}]".format(name_i, *pred_range_i))

            self._posterior = posterior

        return self._posterior

    def plot_posterior(self):
        """
        Plot the posterior distributions for each parameter.

        Returns
        -------
        fig, axes
        """
        model_file = os.path.join(self.model_path, "model.pkl")
        # Loading random forest from '{}'...".format(model_file)
        model = joblib.load(model_file)

        fig, axes = plot_posterior_matrix(self._posterior,
                                          names=model.names,
                                          ranges=model.ranges,
                                          colors=model.colors)
        os.makedirs(self.output_path, exist_ok=True)
        fig.savefig(os.path.join(self.output_path, "posterior_matrix.pdf"),
                    bbox_inches='tight')
        return fig, axes


def data_ranges(posterior, percentiles=(50, 16, 84)):
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
    values = wpercentile(posterior.samples, posterior.weights,
                         percentiles, axis=0)
    ranges = np.array(
        [values[0], values[2] - values[0], values[0] - values[1]])
    return ranges.T


def generate_example_data():
    """
    Generate an example dataset in the new directory ``linear_dataset``.

    Returns
    -------
    example_dir : str
        Path to the directory of the example data
    training_dataset : str
        Path to the dataset metadata JSON file
    samples_path : str
        Path to the numpy pickle of the samples to predict on
    """
    example_dir = 'linear_dataset'
    training_dataset = os.path.join(example_dir, 'example_dataset.json')
    samples_path = 'samples.npy'

    os.makedirs(example_dir, exist_ok=True)

    # Save the dataset metadata to a JSON file
    dataset = {
        "metadata": {
            "names": ["slope", "intercept"],
            "ranges": [[0, 1], [0, 1]],
            "colors": ["#F14532", "#4a98c9"],
            "num_features": 1000
        },
        "training_data": "training.npy",
        "testing_data": "testing.npy"
    }

    with open(training_dataset, 'w') as fp:
        json.dump(dataset, fp)

    # Generate fake training data
    npoints = 1000

    slopes = np.random.rand(npoints)
    ints = np.random.rand(npoints)
    x = np.linspace(0, 1, 1000)[:, np.newaxis]
    data = slopes * x + ints

    labels = np.vstack([slopes, ints])
    X = np.vstack([data, labels])

    # Split dataset into training and testing segments
    training = X[:, :int(0.8 * npoints)].T
    testing = X[:, int(-0.2 * npoints):].T

    np.save(os.path.join(example_dir, 'training.npy'), training)
    np.save(os.path.join(example_dir, 'testing.npy'), testing)

    # Generate a bunch of samples with a test value to "retrieve" with the
    # random forest:
    true_slope = 0.2
    true_intercept = 0.5

    samples = true_slope * x + true_intercept
    np.save(samples_path, samples.T)
    return training_dataset, example_dir, samples_path
