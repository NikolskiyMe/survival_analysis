import warnings

import numpy as np
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sksurv.metrics import concordance_index_censored


def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y['Status'], y['Survival_in_days'], prediction)
    return result[0]


def optimize(estimator, param_grid, n_splits=100, test_size=0.5, random_state=0, n_jobs=4, refit=False):
    cv = ShuffleSplit(n_splits=n_splits,
                      test_size=test_size,
                      random_state=random_state)

    model = GridSearchCV(estimator, param_grid, scoring=score_survival_model,
                       n_jobs=n_jobs, refit=refit, cv=cv)

    # model.name = estimator.name

    return model



