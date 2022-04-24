import time
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from metrics import MyCIndex
from utils.plt_helper import draw_function

clear = lambda: os.system('clear')


class ModelsResult:
    def __init__(self, model_name=None, model_params=None):
        self._model_name = model_name
        self._model_params = model_params
        self._model_time = None

        self._score = []

    def add_score(self, new_score=None):
        self._score.append(new_score)

    @property
    def model_time(self):
        return self._model_time

    @model_time.setter
    def model_time(self, new_value):
        self._model_time = new_value

    def __str__(self):
        res_str = ''
        res_str += f'[{self._model_name}] fitted for {self._model_time} sec.'
        for name, value in self._score:
            res_str += f'\n\t{name}: {value}'
        return res_str


class Experiment:
    def __init__(self, test_size=None, num_of_repeat=1):
        self.test_size = test_size
        self.num_of_repeat = num_of_repeat

    def run(self, X, y, models, metrics) -> list[ModelsResult]:
        report_res = []

        for num_experiment in range(self.num_of_repeat):

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
            for model in models:

                model_res = ModelsResult(model.name)

                start_ts = time.time()
                est = model.fit(X_train, y_train)
                end_ts = time.time()
                tm = end_ts - start_ts

                model_res.model_time = tm

                variables = [i for i in dir(est) if not callable(i)]

                chf_func, surv_func = None, None

                if 'predict_cumulative_hazard_function' in variables:
                    chf_func = est.predict_cumulative_hazard_function(X)
                if 'predict_survival_function' in variables:
                    surv_func = est.predict_survival_function(X)

                if model.name.startswith("Optimize "):
                    y_pred = None
                else:
                    y_pred = est.predict(X_test)

                # Это костыль
                for metric in metrics:
                    if model.name.startswith("Optimize "):
                        res = est.best_score_
                        model_res.add_score((metric.name, res))
                        break
                    if metric.name == 'C-index censored':
                        res = metric(y_test, y_pred)
                        model_res.add_score((metric.name, res))
                    elif metric.name == 'C-index ipcw':
                        res = metric(y_train, y_test, y_pred)
                        model_res.add_score((metric.name, res))

                report_res.append(model_res)

        return report_res


class ExperimentCV:
    def __init__(self, n_splits=6, random_state=42, shuffle=True):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

    def run(self, X, y, models, metrics):
        report_res = []

        cv = KFold(n_splits=self.n_splits,
                   random_state=self.random_state,
                   shuffle=self.shuffle)

        for model in models:
            sum_time = 0

            metric_res = {}
            for metric in metrics:
                metric_res[metric.name] = []

            model_res = ModelsResult(model.name)

            for train_index, test_index in cv.split(y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                start_ts = time.time()
                est = model.fit(X_train, y_train)
                end_ts = time.time()
                tm = end_ts - start_ts
                sum_time += tm

                if model.name.startswith("Optimize "):
                    pass
                else:
                    y_pred = est.predict(X_test)

                for metric in metrics:
                    if model.name.startswith("Optimize "):
                        res = est.best_score_
                        print(f'| {metric.name}: {res}')
                        metric_res[metric.name].append(res)
                        break
                    if metric.name == 'C-index censored':
                        res = metric(y_test, y_pred)
                        metric_res[metric.name].append(res)

                for k, v in metric_res.items():
                    metric_res[k] = np.mean(metric_res[k])

            model_res.model_time = sum_time

        return report_res
