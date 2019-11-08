import os
import json

import numpy as np
from sklearn import metrics, multioutput
import joblib

from .dataset import load_dataset, load_data_file
from .models import Model
from .plot import predicted_vs_real, feature_importances, posterior_matrix
from .wpercentile import wpercentile

__all__ = ['RandomForest', 'generate_example_data']


def train_model(dataset, num_trees, num_jobs, verbose=1):
    pipeline = Model(num_trees, num_jobs,
                     names=dataset.names,
                     ranges=dataset.ranges,
                     colors=dataset.colors,
                     verbose=verbose)
    pipeline.fit(dataset.training_x, dataset.training_y)
    return pipeline


def test_model(model, dataset, output_path):
    if dataset.testing_x is None:
        return

    pred = model.predict(dataset.testing_x)
    r2scores = {name_i: metrics.r2_score(real_i, pred_i)
                for name_i, real_i, pred_i in
                zip(dataset.names, dataset.testing_y.T, pred.T)}
    print("Testing scores:")
    for name, values in r2scores.items():
        print("\tR^2 score for {}: {:.3f}".format(name, values))

    fig = predicted_vs_real(dataset.testing_y, pred, dataset.names,
                            dataset.ranges)
    fig.savefig(os.path.join(output_path, "predicted_vs_real.pdf"),
                bbox_inches='tight')
    return r2scores


def compute_feature_importance(model, dataset, output_path):
    regr = multioutput.MultiOutputRegressor(model, n_jobs=1)
    regr.fit(dataset.training_x, dataset.training_y)

    forests = [i.rf for i in regr.estimators_] + [model.rf]

    fig = feature_importances(
                forests=[i.rf for i in regr.estimators_] + [model.rf],
                names=dataset.names + ["joint prediction"],
                colors=dataset.colors + ["C0"])

    fig.savefig(os.path.join(output_path, "feature_importances.pdf"),
                bbox_inches='tight')
    return np.array([forest_i.feature_importances_ for forest_i in forests])


class RandomForest(object):
    """
    A class for a random forest.
    """
    def __init__(self, training_dataset, model_path, data_file):
        """
        Parameters
        ----------
        training_dataset
        model_path
        data_file
        """
        self.training_dataset = training_dataset
        self.model_path = model_path
        self.data_file = data_file
        self.output_path = self.model_path

        self.dataset = None
        self.model = None

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

        r2scores = test_model(self.model, self.dataset, self.model_path)

        return r2scores

    def feature_importance(self):
        """
        Compute feature importance.

        Returns
        -------
        feature_importances : `~numpy.ndarray`
        """
        self.model.enable_posterior = False
        return compute_feature_importance(self.model, self.dataset,
                                          self.model_path)

    def predict(self, plot_posterior=True):
        """
        Predict values from the trained random forest.

        Parameters
        ----------
        plot_posterior : bool

        Returns
        -------
        preds : `~numpy.ndarray`
            N x M values where N is number of parameters, M is number of
            samples/trees (check out attributes of model for metadata)
        """
        model_file = os.path.join(self.model_path, "model.pkl")
        # Loading random forest from '{}'...".format(model_file)
        model = joblib.load(model_file)

        # Loading data from '{}'...".format(data_file)
        data, _ = load_data_file(self.data_file, model.rf.n_features_)

        posterior = model.posterior(data[0])

        posterior_ranges = data_ranges(posterior)
        for name_i, pred_range_i in zip(model.names, posterior_ranges):
            print("Prediction for {}: {:.3g} "
                  "[+{:.3g} -{:.3g}]".format(name_i, *pred_range_i))

        if plot_posterior:
            # Plotting and saving the posterior matrix..."
            fig = posterior_matrix(posterior,
                                   names=model.names,
                                   ranges=model.ranges,
                                   colors=model.colors)
            os.makedirs(self.output_path, exist_ok=True)
            fig.savefig(os.path.join(self.output_path, "posterior_matrix.pdf"),
                        bbox_inches='tight')
        return posterior

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
    samples, weights = posterior
    values = wpercentile(samples, weights, percentiles, axis=0)
    ranges = np.array([values[0], values[2]-values[0], values[0]-values[1]])
    return ranges.T

def generate_example_data():
    """
    Generate an example dataset in the new directory ``linear_dataset``
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
    return example_dir, training_dataset, samples_path