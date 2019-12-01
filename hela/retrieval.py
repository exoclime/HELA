import os
import json

import numpy as np
from sklearn import metrics, multioutput
import joblib

from .posteriors import PosteriorRandomForest
from .plot import plot_predicted_vs_real, plot_feature_importances

__all__ = ['Retrieval', 'generate_example_data', 'save_model', 'load_model']


def save_model(path, model, **kwargs):
    joblib.dump(model, path, **kwargs)


def load_model(path, **kwargs):
    joblib.load(path, **kwargs)


def train_model(dataset, num_trees, num_jobs, verbose=1):
    pipeline = PosteriorRandomForest(num_trees, num_jobs,
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

    def __init__(self):
        self.model = None
        self._feature_importance = None
        self.oob = None
        self.pred = None

    def train(self, dataset, num_trees=1000, num_jobs=5, quiet=False):
        """
        Train the random forest on a set of observations.

        Parameters
        ----------
        dataset : `~hela.Dataset`
        num_trees : int
        num_jobs : int
        quiet : bool

        Returns
        -------
        r2scores : dict
            :math:`R^2` values for each parameter after training
        """
        # Training model
        self.model = train_model(dataset, num_trees, num_jobs, not quiet)

        # saving model information...
        self.oob = self.model.rf.oob_score_

        pred, r2scores = test_model(self.model, dataset)
        self.pred = pred
        return r2scores

    def plot_predicted_vs_real(self, dataset):
        """
        Plot training correlations.

        Returns
        -------
        fig, axes
        """
        fig, axes = plot_predicted_vs_real(dataset, self)
        return fig, axes

    def feature_importance(self, dataset):
        """
        Compute feature importance.

        Parameters
        ----------
        dataset : `~hela.Dataset`

        Returns
        -------
        feature_importances : `~numpy.ndarray`
        """
        if self._feature_importance is None:
            self._feature_importance = compute_feature_importance(self.model,
                                                                  dataset)
        return self._feature_importance

    def plot_feature_importance(self, dataset):
        """
        Plot the feature importances.

        Parameters
        ----------
        dataset : `~hela.Dataset`

        Returns
        -------
        fig, axes
        """
        forests = self.feature_importance()
        fig, axes = plot_feature_importances(forests=forests,
                                             names=(dataset.names +
                                                    ["joint prediction"]),
                                             colors=(dataset.colors +
                                                     ["C0"]))
        return fig, axes

    def predict(self, x):
        """
        Predict values from the trained random forest.

        Parameters
        ----------
        x : `~numpy.ndarray`

        Returns
        -------
        preds : `~numpy.ndarray`
            ``N x M`` values where ``N`` is number of parameters, ``M`` is
            number of samples/trees (check out attributes of model for
            metadata)
        """
        posterior = self.model.predict_posterior(x)

        return posterior


def generate_example_data():
    """
    Generate an example dataset in the new directory ``linear_dataset``.

    Returns
    -------
    example_dir : str
        Path to the directory of the example data
    training_dataset : str
        Path to the dataset metadata JSON file
    samples : `~numpy.ndarray`
    """
    example_dir = 'linear_dataset'
    training_dataset = os.path.join(example_dir, 'example_dataset.json')

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

    slopes = np.random.uniform(size=npoints)
    ints = np.random.uniform(size=npoints)
    x = np.linspace(0, 1, 1000)[:, None]

    # Add correlated noise to parameters to introduce degeneracies
    noise_ints = np.random.normal(scale=0.15, size=npoints)
    noise_slopes = (np.abs(noise_ints) +
                    np.random.normal(scale=0.02, size=npoints))

    # Add also noise to data points (not strictly necessary)
    data = ((slopes + noise_slopes) * x + (ints + noise_ints) +
            np.random.normal(scale=0.01, size=(1000, npoints)))

    labels = np.vstack([slopes, ints])
    X = np.vstack([data, labels])

    # Split dataset into training and testing segments
    training = X[:, :int(0.8 * npoints)].T
    testing = X[:, int(0.8 * npoints):].T

    np.save(os.path.join(example_dir, 'training.npy'), training)
    np.save(os.path.join(example_dir, 'testing.npy'), testing)

    # Generate a bunch of samples with a test value to "retrieve" with the
    # random forest:
    true_slope = 0.7
    true_intercept = 0.5

    samples = true_slope * x + true_intercept
    return training_dataset, example_dir, samples.T[0]
