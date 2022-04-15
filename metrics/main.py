from sksurv.metrics import (
    concordance_index_censored,
    brier_score,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

import numpy as np


class MyCIndex:
    def __init__(self, tied_tol=1e-8):
        self.tied_tol = tied_tol

    def __call__(self, y_test, y_pred):
        event_indicator = [y[0] for y in y_test]
        event_time = [y[1] for y in y_test]
        estimate = y_pred[:1800]

        score = concordance_index_censored(event_indicator=event_indicator,
                                           event_time=event_time,
                                           estimate=estimate,
                                           tied_tol=self.tied_tol)

        return score[0]

    @property
    def name(self):
        return 'C-index censored'


class MyBrierScore:
    def __init__(self, times):
        self.times = times

    def __call__(self, survival_train, survival_test, surv_func):
        estimate = [f(self.times) for f in surv_func]
        times, score = brier_score(survival_train=survival_train,
                                   survival_test=survival_test,
                                   estimate=estimate,
                                   times=self.times)

        return score[0]

    @property
    def name(self):
        return 'Brier score'


class MyCIndexIPCW:
    def __init__(self, tau=None, tied_tol=1e-08):
        self.tau = tau
        self.tied_tol = tied_tol

    def __call__(self, survival_train, survival_test, estimate):
        score = concordance_index_ipcw(survival_train=survival_train,
                                       survival_test=survival_test,
                                       estimate=estimate,
                                       tau=self.tau,
                                       tied_tol=self.tied_tol)
        return score[0]

    @property
    def name(self):
        return 'C-index ipcw'


class MyCumulativeDynamicAuc:
    def __init__(self, times, tied_tol=1e-08):
        self.times = times
        self.tied_tol = tied_tol

    def __call__(self, survival_train, survival_test, surv_func):
        estimate = [f(self.times) for f in surv_func]
        score = cumulative_dynamic_auc(survival_train,
                                       survival_test,
                                       estimate,
                                       self.times,
                                       self.tied_tol)
        return score

    @property
    def name(self):
        return 'Cumulative dynamic auc'


class MyIntegratedBrierScore:
    def __init__(self, times, tied_tol=1e-08):
        self.times = times
        self.tied_tol = tied_tol

    def __call__(self, survival_train, survival_test, func_surv):

        times = np.arange(self.times[0], self.times[1])
        estimate = np.asarray([[fn(t) for t in times] for fn in func_surv])

        score = integrated_brier_score(survival_train=survival_train[:720],
                                       survival_test=survival_test[:720],
                                       estimate=estimate[:720],
                                       times=times)

        return score

    @property
    def name(self):
        return 'Integrated brier score'

