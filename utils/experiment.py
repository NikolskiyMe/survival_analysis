import time
import os

import numpy as np
from scipy.stats import sem, t
from sklearn.model_selection import train_test_split, KFold

clear = lambda: os.system('clear')


def mean(lst: list):
    if len(lst) != 0:
        return sum(lst) / len(lst)
    return 0


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


class MetricResult:
    """
    Класс для значений метрик
    """
    def __init__(self, value=0.0, confidence_interval=None):
        self._value: float = value  # Значение
        self._confidence_interval: tuple = confidence_interval  # Доверительный интервал

    def __repr__(self):
        return f'Metric result'

    def __str__(self):
        return f'  Value: {self._value}\n' \
               f'  Confidence_interval:\n' \
               f'      m: {self._confidence_interval[0]}\n' \
               f'      m-h: {self._confidence_interval[1]}\n' \
               f'      m+h {self._confidence_interval[2]}\n'


class ModelsResult:
    """
    Класс для результатов эксперимента каждого метода из списка
    """
    def __init__(self, model_name=None, model_params=None):
        self._model_name: str = model_name  # Название модели
        self._model_params: dict = model_params  # Параметры модели
        self._model_time: float = 0.0  # Время обучения

        self._score: dict = {}  # Словарь экземпляров MetricResult()

    @property
    def scores(self):
        return self._score

    @scores.setter
    def scores(self, new_score):
        self._score = new_score

    @property
    def time(self):
        return self._model_time

    @time.setter
    def time(self, new_value):
        self._model_time = new_value

    def __repr__(self):
        return f'Model result of {self._model_name}'

    def __str__(self):
        res_str = ''
        res_str += f'---[ {self._model_name} ]--- fitted for {self._model_time} sec.\n'
        for score_name, score_val in self._score.items():
            res_str += f'\n <{score_name}>\n'
            res_str += f'{score_val}\n'
        return res_str


class Experiment:
    def __init__(self, test_size=None, num_of_repeat=1):
        self.test_size = test_size
        self.num_of_repeat = num_of_repeat

    def run(self, X, y, models, metrics) -> dict:
        print("=== Experiment with repeat START ======================")

        # Итоговый словарь
        report_res = {model.name: ModelsResult(model_name=model.name)
                      for model in models}

        for model in models:
            # Суммарное время обучения на повторах
            sum_time = 0

            # Словарь метрик для каждой модели
            metric_res_tmp = {metric.name: [] for metric in metrics}
            metric_res = {}

            for num_experiment in range(self.num_of_repeat):
                X_train, X_test, \
                y_train, y_test = train_test_split(X, y, test_size=self.test_size)

                print(f'  === Fitting {model.name}... ===')
                start_ts = time.time()
                est = model.fit(X_train, y_train)
                end_ts = time.time()
                tm = end_ts - start_ts
                sum_time += tm
                print(f'  === Fitting {model.name} OK ===')

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
                        metric_res_tmp[metric.name].append(res) if res else ...
                        break
                    if metric.name == 'C-index censored':
                        res = metric(y_test, y_pred)
                        metric_res_tmp[metric.name].append(res) if res else ...
                    elif metric.name == 'C-index ipcw':
                        res = metric(y_train, y_test, y_pred)
                        metric_res_tmp[metric.name].append(res) if res else ...

            # Считаем среднее для каждой метрики
            for k, v in metric_res_tmp.items():
                metric_ci = mean_confidence_interval(metric_res_tmp[k])
                metric_val = mean(metric_res_tmp[k])
                metric_res[k] = MetricResult(value=metric_val,
                              confidence_interval=metric_ci)

            report_res[model.name].scores = metric_res
            report_res[model.name].time = sum_time

        print("=== Experiment with repeat OK. ======================\n")
        return report_res


class ExperimentCV:
    def __init__(self, n_splits=6, random_state=42, shuffle=True):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

    def run(self, X, y, models, metrics):
        print("=== Experiment with cross-validation START ===")

        # итоговый словарь
        report_res = {model.name: ModelsResult(model_name=model.name)
                      for model in models}

        cv = KFold(n_splits=self.n_splits,
                   random_state=self.random_state,
                   shuffle=self.shuffle)

        for model in models:
            sum_time = 0  # Суммарное время обучения на фолдах

            # Словарь метрик для каждой модели
            metric_res = {metric.name: [] for metric in metrics}

            for train_index, test_index in cv.split(y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                start_ts = time.time()
                est = model.fit(X_train, y_train)
                end_ts = time.time()
                tm = end_ts - start_ts
                sum_time += tm

                variables = [i for i in dir(est) if not callable(i)]

                chf_func, surv_func = None, None

                if 'predict_cumulative_hazard_function' in variables:
                    chf_func = est.predict_cumulative_hazard_function(X)
                if 'predict_survival_function' in variables:
                    surv_func = est.predict_survival_function(X)

                if model.name.startswith("Optimize "):
                    pass
                else:
                    y_pred = est.predict(X_test)

                for metric in metrics:
                    if model.name.startswith("Optimize "):
                        res = est.best_score_
                        metric_res[metric.name].append(res) if res else ...
                        break
                    elif metric.name == 'C-index censored':
                        res = metric(y_test, y_pred)
                        metric_res[metric.name].append(res) if res else ...
                    elif metric.name == 'C-index ipcw':
                        res = metric(y_train, y_test, y_pred)
                        metric_res[metric.name].append(res) if res else ...

            # Считаем среднее для каждой метрики
            for k, v in metric_res.items():
                metric_res[k] = mean(metric_res[k])

            report_res[model.name].scores = metric_res
            report_res[model.name].time = sum_time

        print("=== Experiment with cross-validation OK. ===\n")
        return report_res
