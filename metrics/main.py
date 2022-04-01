from sksurv.metrics import (
    concordance_index_censored,
    brier_score,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score
)


def c_index_censored(pred_risks, true_times, true_events):
    score, _, _, _, _ = concordance_index_censored(true_events, true_times,
                                                   pred_risks)
    return score


class Score:
    def __init__(self, survival_train, survival_test, estimate):
        self.survival_train = survival_train
        self.survival_test = survival_test
        self.estimate = estimate


class BrierScore(Score):
    def __init__(self, survival_train, survival_test, estimate):
        super().__init__(survival_train, survival_test, estimate)

    def __call__(self, times):
        _, score = brier_score(self.survival_train,
                               self.survival_test,
                               self.estimate,
                               times)
        return score[0]

    @property
    def name(self):
        return self.__class__.__name__


class CIndexIpcw(Score):
    def __init__(self, survival_train, survival_test, estimate):
        super().__init__(survival_train, survival_test, estimate)

    def __call__(self):
        score, _, _, _, _ = concordance_index_ipcw(self.survival_train,
                                                   self.survival_test,
                                                   self.estimate)
        return score

    @property
    def name(self):
        return self.__class__.__name__


class CumulativeDynmicAuc(Score):
    def __init__(self, survival_train, survival_test, estimate):
        super().__init__(survival_train, survival_test, estimate)

    def __call__(self, times):
        _, score = cumulative_dynamic_auc(self.survival_train,
                                          self.survival_test,
                                          self.estimate,
                                          times)
        return score

    @property
    def name(self):
        return self.__class__.__name__


class IntegratedBrierScore(Score):
    def __init__(self, survival_train, survival_test, estimate):
        super().__init__(survival_train, survival_test, estimate)

    def __call__(self, times):
        score = integrated_brier_score(self.survival_train,
                                       self.survival_test,
                                       self.estimate,
                                       times)
        return score

    @property
    def name(self):
        return self.__class__.__name__
