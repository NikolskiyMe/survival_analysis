import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, ShuffleSplit, \
    GridSearchCV
from sklearn.utils._testing import ignore_warnings
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM

from utils.base import BaseMLClass


@ignore_warnings(category=ConvergenceWarning)
def score_survival_model(model, x, y):
    prediction = model.predict(x)
    result = concordance_index_censored(y['Status'], y['Survival_in_days'], prediction)
    return result[0]


class SSVMModel(BaseMLClass):

    @ignore_warnings(category=ConvergenceWarning)
    def __init__(self, x, y):

        estimator = FastSurvivalSVM(max_iter=1000, tol=1e-5, random_state=0)
        param_grid = {'alpha': 2. ** np.arange(-12, 13, 2)}
        cv = ShuffleSplit(n_splits=100, test_size=0.5, random_state=0)
        gcv = GridSearchCV(estimator, param_grid, scoring=score_survival_model,
                           n_jobs=4, refit=False, cv=cv)
        self.model = gcv.fit(x, y)

    def get_score(self):
        return round(self.model.best_score_, 3)
