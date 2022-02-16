from sksurv.metrics import (
    concordance_index_censored,
    brier_score
)

from sklearn.metrics import mean_absolute_error

from utils.errors import ParameterError


"""
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

