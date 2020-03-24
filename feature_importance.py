
from collections import namedtuple
from multiprocessing import Pool

import numpy as np

from models import _tree_weights

_x = None
_y = None


def compute_impurities_per_output(forest, x, y):

    # impurities = []
    # for tree in tqdm(forest):
    #     impurities_i = tree_compute_impurities_per_output(tree, x, y)
    #     impurities.append(impurities_i)
    global _x
    global _y
    _x = x
    _y = y
    with Pool(8) as p:
        impurities = p.map(tree_compute_impurities_per_output,
                           forest.estimators_)

    return impurities


def compute_importance_per_output(forest, impurities):

    importances = []
    for tree, impurities_i in zip(forest, impurities):
        importance_i = tree_compute_importance_per_output(tree, impurities_i)
        importances.append(importance_i)

    # importances = np.array(importances)
    importances = np.mean(importances, axis=0)

    return importances / importances.sum()


def tree_compute_impurities_per_output(tree):

    tree_ = tree.tree_
    x, y = _x, _y
    impurities = np.zeros((tree.tree_.node_count, y.shape[1]))
    weights = _tree_weights(tree, len(x))

    def _tree_compute_impurities_per_output(node_idx, x, y, weights):

        imp = _wvariance(y, weights, axis=0)
        impurities[node_idx] = imp

        if tree_.children_left[node_idx] == -1:
            return

        feature = tree_.feature[node_idx]
        threshold = tree_.threshold[node_idx]
        mask = x[:, feature] <= threshold
        _tree_compute_impurities_per_output(
            tree_.children_left[node_idx],
            x[mask],
            y[mask],
            weights[mask]
        )
        mask = np.logical_not(mask)
        _tree_compute_impurities_per_output(
            tree_.children_right[node_idx],
            x[mask],
            y[mask],
            weights[mask]
        )

    _tree_compute_impurities_per_output(0, x, y, weights)
    return impurities


def tree_compute_importance_per_output(tree, impurities):

    tree_ = tree.tree_
    importances = np.zeros((tree_.n_features, tree_.n_outputs))

    Node = namedtuple("Node", ['left', 'right', 'weighted_n_node_samples',
                      'feature', 'impurities'])

    for node in zip(tree_.children_left,
                    tree_.children_right,
                    tree_.weighted_n_node_samples,
                    tree_.feature,
                    impurities):
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


def _wvariance(data, weights, axis=None):
    avg = np.average(data, weights=weights, axis=axis)
    return np.average((data - avg)**2, weights=weights, axis=axis)
