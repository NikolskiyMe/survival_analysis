from abc import ABC, abstractmethod

from sksurv.metrics import (
    concordance_index_censored,
    brier_score,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
    as_concordance_index_ipcw_scorer,
    as_cumulative_dynamic_auc_scorer,
    as_integrated_brier_score_scorer
)


class ScoreBaseA(ABC):
    def __init__(self, survival_train, survival_test, estimate):
        self.survival_train = survival_train
        self.survival_test = survival_test
        self.estimate = estimate

    @abstractmethod
    def score(self):
        pass

    @abstractmethod
    def name(self):
        pass


class ScoreBaseB(ABC):
    def __init__(self, estimator):
        self.estimator = estimator

    @abstractmethod
    def score(self):
        pass

    @abstractmethod
    def name(self):
        pass


class BrierScore(ScoreBaseA):
    def __init__(self, survival_train=None, survival_test=None, estimate=None, times=500):
        estimate = [fn(times) for fn in estimate]
        super().__init__(survival_train, survival_test, estimate)
        self.times = times

    @property
    def score(self):
        _, metric = brier_score(self.survival_train,
                                self.survival_test,
                                self.estimate,
                                self.times)
        return metric

    @property
    def name(self):
        return 'Brier Score'


class ConcordanceIndexCensored:
    def __init__(self, event_indicator, event_time, estimate, tied_tol=1e-08):
        self.event_indicator = event_indicator
        self.event_time = event_time
        self.estimate = estimate
        self.tied_tol = tied_tol

    @property
    def score(self):
        metric = concordance_index_censored(self.event_indicator,
                                            self.event_time,
                                            self.estimate,
                                            self.tied_tol)
        return metric

    @property
    def name(self):
        return 'C-Index censored'


class ConcordanceIndexIpcw(ScoreBaseA):
    def __init__(self, survival_train, survival_test, estimate, tau=None, tied_tol=1e-08):
        estimate = [fn(500) for fn in estimate]
        super().__init__(survival_train, survival_test, estimate)
        self.tau = tau
        self.tied_tol = tied_tol

    @property
    def score(self):
        metric, _, _, _, _ = concordance_index_ipcw(self.survival_train,
                                                    self.survival_test,
                                                    self.estimate,
                                                    self.tau,
                                                    self.tied_tol)
        return [metric]  # ToDo: это костыль! Разобраться почему в BS лист, а тут число

    @property
    def name(self):
        return 'C-Index ipcw'


class CumulativeDynamicAuc(ScoreBaseA):
    def __init__(self, survival_train, survival_test, estimate, times, tied_tol=1e-08):
        super().__init__(survival_train, survival_test, estimate)
        self.times = times
        self.tied_tol = tied_tol

    @property
    def score(self):
        metric = cumulative_dynamic_auc(self.survival_train,
                                        self.survival_test,
                                        self.estimate,
                                        self.times,
                                        self.tied_tol)
        return metric

    @property
    def name(self):
        return 'Cumulative dynamic auc'


class IntegratedBrierScore(ScoreBaseA):
    def __init__(self, survival_train, survival_test, estimate, times):
        super().__init__(survival_train, survival_test, estimate)
        self.times = times

    @property
    def score(self):
        metric = integrated_brier_score(self.survival_train,
                                        self.survival_test,
                                        self.estimate,
                                        self.times)
        return metric

    @property
    def name(self):
        return 'Integrated Brier Score'


# ToDo: Все, что ниже - переписать,
class AsConcordanceIndexIpcwScorer(ScoreBaseB):
    def __init__(self, estimator, tau=None, tied_tol=1e-08):
        super().__init__(estimator)
        self.tau = tau
        self.tied_tol = tied_tol

    @property
    def score(self):
        metric = as_concordance_index_ipcw_scorer(self.estimator,
                                                  self.tau,
                                                  self.tied_tol)
        return metric


class AsCumulativeDynamicAucScorer(ScoreBaseB):
    def __init__(self, estimator, times, tied_tol=1e-08):
        super().__init__(estimator)
        self.times = times
        self.tied_tol = tied_tol

    @property
    def score(self):
        metric = as_cumulative_dynamic_auc_scorer(self.estimator,
                                                  self.times,
                                                  self.tied_tol)
        return metric


class AsIntegratedBrierScoreScorer(ScoreBaseB):
    def __init__(self, estimator, times):
        super().__init__(estimator)
        self.times = times

    @property
    def score(self):
        metric = as_integrated_brier_score_scorer(self.estimator,
                                                  self.times)
        return metric

