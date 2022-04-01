import time

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import brier_score

from metrics.main import c_index_censored, Score
from .info import Info
from .plt_helper import draw_function
from .report_generation import get_report

experiment_num = 0


class Experiment:
    """
    Данный класс предоставляет методы для выбора метрик и моделей из
    списка(словаря)
    """
    def __init__(self, model, x, y):
        self.x = x
        self.y = y
        self.model = model
        glob = globals()
        glob['experiment_num'] += 1

    @property
    def get_res(self):

        x_train, x_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=0.5,
                                                            random_state=1)

        print(f'{self.model.name} is fitting ...')
        start_ts = time.time()
        est = self.model.fit(x_train, y_train)
        end_ts = time.time()
        tm = end_ts - start_ts

        print(f'Fit {self.model.name}: OK')
        print(f'Fit time for {self.model.name}: {round(tm, 3)} sec.')

        variables = [i for i in dir(est) if not callable(i)]

        chf_funcs, surv_funcs = None, None
        if 'predict_cumulative_hazard_function' in variables:
            chf_funcs = est.predict_cumulative_hazard_function(self.x)
        if 'predict_survival_function' in variables:
            surv_funcs = est.predict_survival_function(self.x)
        y_pred = est.predict(x_test)


        # draw_function(chf_funcs)  # cumulative hazard function
        # draw_function(surv_funcs)  # survival_function

        return chf_funcs, surv_funcs, y_pred, y_train, y_test
