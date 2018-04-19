
import numpy as np

from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler

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
            return self.scaler.inverse_transform(y)[:, 0]
        
        return self.scaler.inverse_transform(y)
    
    def fit(self, x, y):
        self._scaler_fit(y)
        self.rf.fit(x, self._scaler_transform(y))
    
    def predict(self, x):
        pred = self.rf.predict(x)
        return self._scaler_inverse_transform(pred)
    
    def get_params(self, deep=True):
        return {"num_trees": self.num_trees, "num_jobs": self.num_jobs,
                "names": self.names, "ranges": self.ranges,
                "colors": self.colors,
                "verbose": self.verbose}
    
    def trees_predict(self, x):
        
        if x.ndim > 1:
            raise ValueError("x.ndim must be 1")
        
        preds = np.array([i.predict(x[None, :])[0] for i in self.rf.estimators_])
        return self._scaler_inverse_transform(preds)

