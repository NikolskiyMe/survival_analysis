import time

from sklearn.model_selection import train_test_split


class Experiment:
    def __init__(self, test_size=None, num_of_repeat=1):
        self.test_size = test_size
        self.num_of_repeat = num_of_repeat

    def hyperparameters_search(self, *args, **kwargs):
        pass

    def run(self, X, y, models, metrics) -> dict:
        report_res = {}
        print('START.\n')

        for num_experiment in range(self.num_of_repeat):

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
            for model in models:
                model_params = str(model.__dict__['model'].__dict__)

                print(f'>>> Fitting {model.name} ...')
                start_ts = time.time()
                est = model.fit(x_train, y_train)
                end_ts = time.time()
                tm = end_ts - start_ts
                print(f'>>> Fitting {model.name}: OK')

                model_key = (model.name, model_params, tm)
                report_res[model_key] = {}

                variables = [i for i in dir(est) if not callable(i)]

                chf_func, surv_func = None, None

                if 'predict_cumulative_hazard_function' in variables:
                    chf_func = est.predict_cumulative_hazard_function(X)
                if 'predict_survival_function' in variables:
                    surv_func = est.predict_survival_function(X)

                y_pred = est.predict(x_test)

                # draw_function(chf_func)  # cumulative hazard function
                # draw_function(surv_func)  # survival_function

                for metric in metrics:
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

                    report_res[model_key][metric.name] = res
                    print(f'{metric.name}-{num_experiment}: {res}')

            print('DONE.')

        return report_res
