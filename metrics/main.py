from sksurv.metrics import (
    concordance_index_censored,
)


class MyCIndex:
    def __init__(self, n_samples, tied_tol=1e-8):
        self.n_samples = n_samples
        self.tied_tol = tied_tol

    def __call__(self, y_test, y_pred):

        # ToDo: Куда n_samples и что передавать в __call__ ?

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
    def __init__(self, n_train_samples, n_samples, n_times):
        self.n_train_samples = n_train_samples
        self.n_samples = n_samples
        self.n_times = n_times

    def __call__(self):
        pass


class MyCIndexIPCW:
    def __init__(self, n_train_samples, n_samples, tau=None, tied_tol=1e-08):
        self.n_train_samples = n_train_samples
        self.n_samples = n_samples

    def __call__(self):
        pass


class MyCumulativeDynamicAuc:
    def __init__(self, n_train_samples, n_samples, n_times, tied_tol=1e-08):
        self.n_train_samples = n_train_samples
        self.n_samples = n_samples
        self.n_times = n_times

    def __call__(self):
        pass


class MyIntegratedBrierScore:
    def __init__(self, n_train_samples, n_samples, n_times):
        self.n_train_samples = n_train_samples
        self.n_samples = n_samples
        self.n_times = n_times

    def __call__(self):
        pass
