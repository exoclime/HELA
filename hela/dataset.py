import os
import json

import numpy as np

__all__ = ["Dataset", "load_dataset", "load_data_file"]


class Dataset(object):
    """
    Class for a dataset used for training the random forest.
    """
    def __init__(self, training_x, training_y, testing_x, testing_y, names,
                 ranges, colors):
        """
        Parameters
        ----------
        training_x : `~numpy.ndarray`
        training_y : `~numpy.ndarray`
        testing_x : `~numpy.ndarray`
        testing_y : `~numpy.ndarray`
        names : list
        ranges : list
        colors : list
        """
        self.training_x = training_x
        self.training_y = training_y
        self.testing_x = testing_x
        self.testing_y = testing_y
        self.names = names
        self.ranges = ranges
        self.colors = colors


def load_data_file(data_file, num_features):
    data = np.load(data_file)

    if data.ndim == 1:
        data = data[None, :]

    x = data[:, :num_features]
    y = data[:, num_features:]

    return x, y


def load_dataset(dataset_file):
    """
    Load a dataset from a JSON file.

    Parameters
    ----------
    dataset_file

    Returns
    -------

    """
    with open(dataset_file, "r") as f:
        dataset_info = json.load(f)

    metadata = dataset_info["metadata"]

    base_path = os.path.dirname(dataset_file)

    # Load training data
    training_file = os.path.join(base_path, dataset_info["training_data"])
    # Loading training data from '{}'...".format(training_file)
    training_x, training_y = load_data_file(training_file,
                                            metadata["num_features"])
    # TODO: slice training_x (data) and training_y (params) to the same length
    # but something smaller for fast docs

    # Optionally, load testing data
    testing_x, testing_y = None, None
    if dataset_info["testing_data"] is not None:
        testing_file = os.path.join(base_path, dataset_info["testing_data"])
        # Loading testing data from '{}'...".format(testing_file)
        testing_x, testing_y = load_data_file(testing_file,
                                              metadata["num_features"])

    return Dataset(training_x, training_y, testing_x, testing_y,
                   metadata["names"], metadata["ranges"], metadata["colors"])
