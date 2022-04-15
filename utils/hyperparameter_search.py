import warnings

import numpy as np
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sksurv.metrics import concordance_index_censored


def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y['death'], y['surv_time'], prediction)
    return result[0]


def grid_search(estimator, X, y):
    param_grid = {'alpha': 2. ** np.arange(-12, 13, 2)}
    cv = ShuffleSplit(n_splits=100, test_size=0.5, random_state=0)
    gcv = GridSearchCV(estimator, param_grid, scoring=score_survival_model,
                       n_jobs=4, refit=False, cv=cv)

    warnings.filterwarnings("ignore", category=FutureWarning)
    gcv = gcv.fit(X, y)

    print(f'{round(gcv.best_score_, 3)} {gcv.best_params_}')


def optimize(*args, **kwargs):
    print('OPTIMIZE OK')
