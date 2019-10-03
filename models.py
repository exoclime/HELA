
import numpy as np
from collections import namedtuple

from sklearn import ensemble
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    "Model",
    "Posterior",
    "resample_posterior"
]

# Posteriors are represented as a collection of weighted samples
Posterior = namedtuple("Posterior", ["samples", "weights"])

def resample_posterior(posterior, num_draws):
    
    p = posterior.weights / posterior.weights.sum()
    indices = np.random.choice(len(posterior.samples), size=num_draws, p=p)
    
    new_weights = np.bincount(indices, minlength=len(posterior.samples))
    mask = new_weights != 0
    new_samples = posterior.samples[mask]
    new_weights = posterior.weights[mask]
    
    return Posterior(new_samples, new_weights)




class Model:
    
    def __init__(self, num_trees, num_jobs,
                 names, ranges, colors, enable_posterior=True,
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
        self._scaler_fit(y)
        self.rf.fit(x, self._scaler_transform(y))
        
        # Build the structures to quickly compute the posteriors
        if self.enable_posterior:
            data_leaves = self.rf.apply(x).T
            self.data_leaves = data_leaves.astype(_smallest_dtype(data_leaves.max()))
            self.data_weights = np.array([_tree_weights(tree, len(y)) for tree in self.rf])
            self.data_y = y
    
    def predict(self, x):
        pred = self.rf.predict(x)
        return self._scaler_inverse_transform(pred)
    
    def get_params(self, deep=True):
        return {"num_trees": self.num_trees, "num_jobs": self.num_jobs,
                "names": self.names, "ranges": self.ranges,
                "colors": self.colors, "enable_posterior": self.enable_posterior,
                "verbose": self.verbose}
    
    def trees_predict(self, x):
        
        if x.ndim > 1:
            raise ValueError("x.ndim must be 1")
        
        preds = np.array([tree.predict(x[None, :])[0] for tree in self.rf])
        return self._scaler_inverse_transform(preds)
    
    def posterior(self, x):
        
        if not self.enable_posterior:
            raise ValueError("Cannot compute posteriors with this model. Set `enable_posterior` to True to enable posterior computation.")
        
        if x.ndim > 1:
            raise ValueError("x.ndim must be 1")
        
        leaves_x = self.rf.apply(x[None, :])[0]
        
        weights_x = np.zeros(len(self.data_y), dtype=self.data_weights.dtype)
        
        for leaf_x, leaves_i, weights_i in zip(leaves_x, self.data_leaves, self.data_weights):
            indices = np.argwhere(leaves_i == leaf_x)
            weights_x[indices] += weights_i[indices]
        
        # Remove samples with weight zero
        mask = weights_x != 0
        samples = self.data_y[mask]
        weights = weights_x[mask]
        
        return Posterior(samples, weights)


def _generate_sample_indices(random_state, n_samples):
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices


def _tree_weights(tree, n_samples):
    indices = _generate_sample_indices(tree.random_state, n_samples)
    return np.bincount(indices, minlength=n_samples)


def _smallest_dtype(n):
    
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    
    for dtype in dtypes:
        if n <= np.iinfo(dtype).max:
            return dtype
    
    raise ValueError("n is too large for any dtype")
