
import numpy as np

from sklearn import ensemble
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    "Model"
]


def _generate_sample_indices(random_state, n_samples):
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices


def _tree_weights(tree, n_samples):
    indices = _generate_sample_indices(tree.random_state, n_samples)
    return np.bincount(indices, minlength=n_samples)


class Model:
    
    def __init__(self, num_trees, num_jobs,
                 names, ranges, colors,
                 verbose=1):
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
        self.data_leaves = None
        # self.leaf_content = None
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
        self._scaler_fit(y)
        self.rf.fit(x, self._scaler_transform(y))
        
        # Build the structures to quickly compute the posteriors
        self.data_leaves = self.rf.apply(x).T
        
        # This could help to make prediction faster, but makes pickling the
        # model much slower.
        # self.leaf_content = [_unique_indices(leaves_i) for leaves_i in leaves.T]
        
        self.data_weights = np.array([_tree_weights(tree, len(y)) for tree in self.rf])
        self.data_y = y
    
    def predict(self, x):
        pred = self.rf.predict(x)
        return self._scaler_inverse_transform(pred)
    
    def get_params(self, _=True):
        return {"num_trees": self.num_trees, "num_jobs": self.num_jobs,
                "names": self.names, "ranges": self.ranges,
                "colors": self.colors,
                "verbose": self.verbose}
    
    def trees_predict(self, x):
        
        if x.ndim > 1:
            raise ValueError("x.ndim must be 1")
        
        preds = np.array([tree.predict(x[None, :])[0] for tree in self.rf])
        return self._scaler_inverse_transform(preds)
    
    def posterior(self, x):
        
        if x.ndim > 1:
            raise ValueError("x.ndim must be 1")
        
        leaves_x = self.rf.apply(x[None, :])[0]
        
        weights_x = np.zeros(len(self.data_y), dtype=self.data_weights.dtype)
        
        for leaf_x, leaves_i, weights_i in zip(leaves_x, self.data_leaves, self.data_weights):
            indices = np.argwhere(leaves_i == leaf_x)
            weights_x[indices] += weights_i[indices]
        
        return self.data_y, weights_x

