from sksurv.metrics import (
    concordance_index_censored,
    brier_score
)

from sklearn.metrics import mean_absolute_error


def get_cindex(event_indicator, event_time, estimate):
    return concordance_index_censored(event_indicator=event_indicator,
                                      event_time=event_time,
                                      estimate=estimate)


def get_bscore(survival_train, survival_test, estimate, times):
    return brier_score(survival_train=survival_train,
                       survival_test=survival_test,
                       estimate=estimate, times=times)


def get_mae(y_true, y_pred):
    return mean_absolute_error(y_true=y_true, y_pred=y_pred)
