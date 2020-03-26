
from collections import namedtuple
from multiprocessing import Pool
from typing import Optional
import logging

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from models import _tree_weights
from utils import tqdm

__all__ = [
    'importances_per_output'
]

LOGGER = logging.getLogger(__name__)


def importances_per_output(
        forest: RandomForestRegressor,
        x: np.ndarray,
        y: np.ndarray
        ) -> np.ndarray:
    """
    Compute the feature importances with a breakdown per output variable (i.e.,
    per label).

    Parameters
    ----------
    forest : RandomForestRegressor
        A trained random forest from which feature importances will be
        extracted.
    x : array-like of shape (n_samples, n_features)
        The input samples of the training dataset used to train `forest`.
    y : array-like of shape (n_samples, n_outputs)

    Returns
    -------
    importances : np.ndarray of shape (n_features, n_outputs)
        The feature importances splitted according to the contributions to each
        output variable. Summing along the columns of this array will provide
        the global feature importances contained in
        forest.feature_importances_.
    """

    impurities = impurities_per_output(forest, x, y)
    return importances_from_impurities(forest, impurities)


def importances_from_impurities(
        forest: RandomForestRegressor,
        impurities: np.ndarray
        ) -> np.ndarray:

    LOGGER.info("Computing importances from impurities...")
    importances_ = []
    for tree, impurities_i in tqdm(zip(forest, impurities), total=len(forest)):
        importance_i = _tree_importances_per_output(tree, impurities_i)
        importances_.append(importance_i)

    # importances = np.array(importances)
    importances = np.mean(importances_, axis=0)

    return importances / importances.sum()


# Global variables to avoid pickling very large arrays with multiprocessing
_X = None
_Y = None


def impurities_per_output(
        forest: RandomForestRegressor,
        x: np.ndarray,
        y: np.ndarray
        ) -> np.ndarray:

    LOGGER.info("Computing impurities...")
    global _X
    global _Y
    _X = x
    _Y = y
    with Pool(forest.n_jobs) as pool:
        impurities = list(tqdm(
            pool.imap(_tree_impurities_per_output, forest),
            total=len(forest)
        ))

    return np.array(impurities)


def _tree_importances_per_output(
        tree: DecisionTreeRegressor,
        impurities: np.ndarray
        ) -> np.ndarray:

    tree_ = tree.tree_
    importances = np.zeros((tree_.n_features, tree_.n_outputs))

    Node = namedtuple(
        "Node",
        ['left', 'right', 'weighted_n_node_samples', 'feature', 'impurities']
    )

    for node in zip(tree_.children_left, tree_.children_right,
                    tree_.weighted_n_node_samples, tree_.feature, impurities):
        node = Node(*node)
        if node.left == -1:
            continue

        left = Node(
            None, None,
            tree_.weighted_n_node_samples[node.left],
            tree_.feature[node.left],
            impurities[node.left]
        )
        right = Node(
            None, None,
            tree_.weighted_n_node_samples[node.right],
            tree_.feature[node.right], impurities[node.right]
        )

        importances[node.feature] += (
            node.weighted_n_node_samples * node.impurities -
            left.weighted_n_node_samples * left.impurities -
            right.weighted_n_node_samples * right.impurities
        )

    return importances / importances.sum()


def _tree_impurities_per_output(tree: DecisionTreeRegressor) -> np.ndarray:

    assert _X is not None and _Y is not None

    tree_ = tree.tree_
    x, y = _X, _Y

    impurities = np.zeros((tree.tree_.node_count, y.shape[1]))
    weights = _tree_weights(tree, len(x))

    def __tree_impurities_per_output(node_idx, x, y, weights):

        imp = _wvariance(y, weights, axis=0)
        impurities[node_idx] = imp

        if tree_.children_left[node_idx] == -1:
            return

        feature = tree_.feature[node_idx]
        threshold = tree_.threshold[node_idx]
        mask = x[:, feature] <= threshold
        __tree_impurities_per_output(
            tree_.children_left[node_idx],
            x[mask],
            y[mask],
            weights[mask]
        )
        mask = np.logical_not(mask)
        __tree_impurities_per_output(
            tree_.children_right[node_idx],
            x[mask],
            y[mask],
            weights[mask]
        )

    __tree_impurities_per_output(0, x, y, weights)
    return impurities


def _wvariance(
        data: np.ndarray,
        weights: np.ndarray,
        axis: Optional[int] = None
        ) -> np.ndarray:

    avg = np.average(data, weights=weights, axis=axis)
    return np.average((data - avg)**2, weights=weights, axis=axis)
