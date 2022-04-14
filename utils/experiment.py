import time

from sklearn.model_selection import train_test_split

from utils.report_generation import get_report


class Experiment:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self._models = None
        self._metrics = None

        self._cross_validation_is_setted = False
        self._hyperparameters_search_is_setted = False

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models):
        self._models = models

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    def cross_validation(self, test_size=0.5, random_state=1):
        pass

    def tts(self, test_size=0.5, random_state=1):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def hyperparameters_search(self, *args, **kwargs):
        pass

    def run(self, in_report=True):
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        report_res = {}

        for model in self.models:

            print(model.__dict__['model'])
            model_params = str(model.__dict__['model'].__dict__)

            start_ts = time.time()
            est = model.fit(self.x_train, self.y_train)
            end_ts = time.time()
            tm = end_ts - start_ts

            print(f'>>> Fit {model.name}: OK\n')
            print(f'>>> Fit time for {model.name}: {round(tm, 3)} sec.')

            if in_report:
                model_key = (model.name, model_params, round(tm, 3))
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

            for metric in self._metrics:
                res = metric(self.y_test, y_pred)
                print(f'>>> CindexCensored: {res}')
                if in_report:
                    report_res[model_key][metric.name] = round(res, 4)

            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        get_report('Add params', report_res)
