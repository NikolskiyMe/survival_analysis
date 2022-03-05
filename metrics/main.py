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

from sklearn.metrics import mean_absolute_error

from utils.errors import ParameterError


class ScoreBaseA:
    def __init__(self, survival_train, survival_test, estimate):
        self.survival_train = survival_train
        self.survival_test = survival_test
        self.estimate = estimate


class ScoreBaseB:
    def __init__(self, estimator):
        self.estimator = estimator


class BrierScore(ScoreBaseA):
    def __init__(self, survival_train, survival_test, estimate, times):
        super().__init__(survival_train, survival_test, estimate)
        self.times = times


class ConcordanceIndexCensored:
    def __init__(self, event_indicator, event_time, estimate, tied_tol):
        self.event_indicator = event_indicator
        self.event_time = event_time
        self.estimate = estimate
        self.tied_tol = tied_tol


class ConcordanceIndexIpcw(ScoreBaseA):
    def __init__(self, survival_train, survival_test, estimate, tau, tied_tol):
        super().__init__(survival_train, survival_test, estimate)
        self.tau = tau
        self.tied_tol = tied_tol


class CumulativeDynamicAuc(ScoreBaseA):
    def __init__(self, survival_train, survival_test, estimate, times, tied_tol):
        super().__init__(survival_train, survival_test, estimate)
        self.times = times
        self.tied_tol = tied_tol


class IntegratedBrierScore(ScoreBaseA):
    def __init__(self, survival_train, survival_test, estimate, times):
        super().__init__(survival_train, survival_test, estimate)
        self.times = times


class AsConcordanceIndexIpcwScorer(ScoreBaseB):
    def __init__(self, estimator, tau, tied_tol):
        super().__init__(estimator)
        self.tau = tau
        self.tied_tol = tied_tol


class AsCumulativeDynamicAucScorer(ScoreBaseB):
    def __init__(self, estimator, times, tied_tol):
        super().__init__(estimator)
        self.times = times
        self.tied_tol = tied_tol


class AsIntegratedBrierScoreScorer(ScoreBaseB):
    def __init__(self, estimator, times):
        super().__init__(estimator)
        self.times = times


"""
# estimate - оценка
class Score:

    def __init__(self, estimate=None):
        self.estimate = estimate

    def drop_estimate(self):
        self.estimate = None

    # ToDo: как передавать тип оценки
    def __call__(self, m_type=1, *args):
        if args == ():
            raise ParameterError('Недостаточно параметров')

        if self.estimate == 'mae':
            try:
                score = mean_absolute_error(y_true=args[0], y_pred=args[1])
            except BaseException:
                raise ParameterError('Неверные параметры для mae')
            else:
                return score

        if m_type == 1:
            try:
                # ToDo: как передавать параметры?
                score = brier_score(survival_train=args[0],
                                    survival_test=args[1],
                                    estimate=self.estimate, times=args[3])
            except BaseException:  # ToDo: Какое исключение?
                raise ParameterError('Неверные параметры для brier_score')
            else:
                return score

        elif m_type == 2:
            try:
                score = concordance_index_censored(event_indicator=args[0],
                                                   event_time=args[1],
                                                   estimate=self.estimate)
            except BaseException:
                raise ParameterError('Неверные параметры для cindex')
            else:
                return score
"""
