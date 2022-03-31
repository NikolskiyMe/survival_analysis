from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import brier_score

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

        results = {}
        results['RSF'] = []

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,
                                                             test_size=0.1,
                                                            random_state=1)
        for model in models:
            print(f'{model.__name__} is fitting ...')
            # ToDo: добавить tqdm для замера времени
            est = model().fit(x_train, y_train)
            print(f'Fit {model.__name__}: OK')

            chf_funcs = est.predict_cumulative_hazard_function(self.x)

            surv_funcs = est.predict_survival_function(self.x)

            y = est.predict(self.x)

            for metric in metrics:
                m = metric(self.y, self.y, surv_funcs, 900)
                print(f'{m.name}: {m.score}')

                # ToDo: перенести метрики в отчет
                # draw_function(chf_funcs)  # cumulative hazard function
                # draw_function(surv_funcs)  # survival_function

                # results[model.name].append(metric.score) - для чисел

        glob = globals()
        e_n = glob['experiment_num']
        get_report(f'Experiment_{e_n}', results)
