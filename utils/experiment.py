from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

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
            # est = GradientBoostingSurvivalAnalysis().fit(x_train, y_train)
            # print(type(est))
            est = model().fit(x_train, y_train)
            print('ok')
            chf_funcs = est.predict_cumulative_hazard_function(self.x)
            surv_funcs = est.predict_survival_function(self.x)

            # draw_function(chf_funcs)
            # draw_function(surv_funcs)

            for metric in metrics:
                pass

                # results[model.name].append(metric.score)

        glob = globals()
        e_n = glob['experiment_num']
        get_report(f'Experiment_{e_n}', results)
