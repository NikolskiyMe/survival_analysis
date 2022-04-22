import time

import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from metrics import MyCIndex
from utils.plt_helper import draw_function

import os

clear = lambda: os.system('clear')


class Experiment:
    def __init__(self, test_size=None, num_of_repeat=1):
        self.test_size = test_size
        self.num_of_repeat = num_of_repeat

    def run(self, X, y, models, metrics) -> dict:
        report_res = {}
        metric_result = {}

        print('START.')

        for num_experiment in range(self.num_of_repeat):

            print(f'[{num_experiment + 1}/{self.num_of_repeat}]')

            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=self.test_size)
            for model in models:
                if metric_result.get(model.name) is None:
                    metric_result[model.name] = 0

                print(f'>>> Fitting {model.name} ...')
                start_ts = time.time()
                est = model.fit(X_train, y_train)
                end_ts = time.time()
                tm = end_ts - start_ts
                print(f'>>> Fitting {model.name}: OK')

                print(f'| Time: {tm}')

                # model_key = (model.name, model_params, tm)
                # report_res[model_key] = {}

                variables = [i for i in dir(est) if not callable(i)]

                chf_func, surv_func = None, None

                if 'predict_cumulative_hazard_function' in variables:
                    chf_func = est.predict_cumulative_hazard_function(X)
                if 'predict_survival_function' in variables:
                    surv_func = est.predict_survival_function(X)
                    # draw_function(surv_func, est)

                if model.name.startswith("Optimize "):
                    pass
                else:
                    y_pred = est.predict(X_test)

                # draw_function(chf_func)  # cumulative hazard function
                # draw_function(surv_func)  # survival_function

                # Это костыль
                for metric in metrics:
                    if model.name.startswith("Optimize "):
                        res = est.best_score_
                        print(f'| {metric.name}: {res}')
                        metric_result[model.name] += res
                        break
                    res = []
                    if metric.name == 'C-index censored':
                        print('    >>> C-index calculating ...')
                        res = metric(y_test, y_pred)
                        print('    >>> C-index calculating: OK')
                    elif metric.name == 'Brier score':
                        print('    >>> Brier score calculating ...')
                        res = metric(y_train, y_test, surv_func)
                        print('    >>> Brier score calculating: OK')
                    elif metric.name == 'C-index ipcw':
                        print('    >>> C-index ipcw calculating ...')
                        res = metric(y_train, y_test, y_pred)
                        print('    >>> C-index ipcw calculating: OK')
                    elif metric.name == 'Cumulative dynamic auc':
                        print('    >>> Cumulative dynamic auc calculating ...')
                        res = metric(y_train, y_test, surv_func)
                        print('    >>> Cumulative dynamic auc calculating: OK')
                    elif metric.name == 'Integrated brier score':
                        print('    >>> Integrated brier score calculating ...')
                        res = metric(y_train, y_test, surv_func)
                        print('    >>> Integrated brier score calculating: OK')

                    # report_res[model_key][metric.name] = res
                    metric_result[model.name] += res
                    print(f'| {metric.name}: {res}')

        print('DONE.\n')

        for k, v in metric_result.items():
            print(f'{k}: {v / self.num_of_repeat}')

        return report_res


class ExperimentCV:
    def __init__(self, n_splits=6, random_state=42, shuffle=True):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

    def run(self, X, y, models, metrics):
        time.sleep(3)

        cv = KFold(n_splits=self.n_splits,
                   random_state=self.random_state,
                   shuffle=self.shuffle)

        result = []

        for model in models:
            results = []
            sum_time = 0

            for train_index, test_index in cv.split(y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                start_ts = time.time()
                est = model.fit(X_train, y_train)
                end_ts = time.time()
                tm = end_ts - start_ts
                sum_time += tm

                time.sleep(0.2)
                clear()
                print(f'Score results: {results}')

                if model.name.startswith("Optimize "):
                    pass
                else:
                    y_pred = est.predict(X_test)

                for metric in metrics:
                    if model.name.startswith("Optimize "):
                        res = est.best_score_
                        print(f'| {metric.name}: {res}')
                        results.append(res)
                        break
                    if metric.name == 'C-index censored':
                        print('    >>> C-index calculating ...')
                        res = metric(y_test, y_pred)
                        print('    >>> C-index calculating: OK')
                        results.append(res)

            clear()
            score = np.mean(results)
            tmp = (model.name, score)
            result.append(tmp)
            print(result)

        return result
