import time

from sklearn.model_selection import train_test_split


class Experiment:
    def __init__(self, X, y, test_size=0.5, random_state=1):
        self.X = X
        self.y = y

        self._cross_validation_is_setted = False
        self._hyperparameters_search_is_setted = False

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)

        n_censored = y.shape[0] - y['Status'].sum()
        print('\nDATASET:')
        print(f'>>> Number of observations: {y.shape[0]}')
        print('>>> %.1f%% of records are censored' % (n_censored / y.shape[0] * 100))
        print()

    def hyperparameters_search(self, *args, **kwargs):
        pass

    def run(self, models, metrics) -> dict:
        report_res = {}
        print('START.\n')

        for model in models:

            model_params = str(model.__dict__['model'].__dict__)

            print(f'>>> Fitting {model.name} ...')
            start_ts = time.time()
            est = model.fit(self.x_train, self.y_train)
            end_ts = time.time()
            tm = end_ts - start_ts
            print(f'>>> Fitting {model.name}: OK\n')

            model_key = (model.name, model_params, tm)
            report_res[model_key] = {}

            variables = [i for i in dir(est) if not callable(i)]

            chf_funcs, surv_funcs = None, None

            if 'predict_cumulative_hazard_function' in variables:
                chf_funcs = est.predict_cumulative_hazard_function(self.X)
            if 'predict_survival_function' in variables:
                surv_funcs = est.predict_survival_function(self.X)

            y_pred = est.predict(self.x_test)

            # draw_function(chf_funcs)  # cumulative hazard function
            # draw_function(surv_funcs)  # survival_function

            for metric in metrics:
                res = metric(self.y_test, y_pred)
                report_res[model_key][metric.name] = res

        print('DONE.')

        return report_res
