import warnings
import numpy as np
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sksurv.metrics import concordance_index_censored

from models.base_model import BaseModel


def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y['Status'], y['Survival_in_days'],
                                        prediction)
    return result[0]


class Optimize(BaseModel):
    def __init__(self, estimator, param_grid, n_splits=100,
                 test_size=0.5, random_state=0, n_jobs=4, refit=False):
        super().__init__(model=estimator)
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.refit = refit

    def fit(self, x_train, y_train):
        cv = ShuffleSplit(n_splits=self.n_splits,
                          test_size=self.test_size,
                          random_state=self.random_state)

        gs = GridSearchCV(self.model, self.param_grid,
                          scoring=score_survival_model,
                          n_jobs=self.n_jobs, refit=self.refit, cv=cv)

        warnings.filterwarnings("ignore", category=FutureWarning)
        res = gs.fit(x_train, y_train)
        return res

    @property
    def name(self):
        return f'Optimize {self.model.name}'
