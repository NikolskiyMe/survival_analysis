import time

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import brier_score

from metrics.base import c_index_censored
from .plt_helper import draw_function
from .report_generation import get_report

experiment_num = 0


class Experiment:
    """
    Данный класс предоставляет методы для выбора метрик и моделей из
    списка(словаря)
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        glob = globals()
        glob['experiment_num'] += 1

    def start(self, models=None, metrics=None):
        if models is None:
            models = []
        if metrics is None:
            metrics = []

        results = {m.__name__: {s.__name__: [] for s in metrics} for m in models}

        x_train, x_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=0.1,
                                                            random_state=1)
        for model in models:
            print(f'{model.__name__} is fitting ...')
            # ToDo: перетащить время в отчет
            start_ts = time.time()
            est = model().fit(x_train, y_train)
            end_ts = time.time()
            tm = end_ts - start_ts

            print(f'Fit {model.__name__}: OK')
            print(f'Fit time for {model.__name__}: {tm}')

            chf_funcs = est.predict_cumulative_hazard_function(self.x)
            surv_funcs = est.predict_survival_function(self.x)

            y = est.predict(x_test)

            print(y_train)
            true_times = [y[1] for y in y_train]
            true_events = [y[0] for y in y_train]
            pred_risks = [fn(800) for fn in surv_funcs][:3240]

            print(len(true_times), len(true_events), len(pred_risks))

            cindex = c_index_censored(pred_risks, true_times, true_events)
            print(cindex)

            """
            for metric in metrics:
                m = metric(self.y, self.y, surv_funcs)
                print(f'{m.name} for {model.__name__}: {m.score}')

                # draw_function(chf_funcs)  # cumulative hazard function
                # draw_function(surv_funcs)  # survival_function

                results[model.__name__][metric.__name__].append(m.score[0])
            """

        # ToDo: добавить в словарь параметры модели

        glob = globals()
        e_n = glob['experiment_num']
        # get_report(f'Experiment_{e_n}', results)
