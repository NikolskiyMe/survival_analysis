from sklearn.model_selection import train_test_split


class Experiment:
    """
    Данный класс предоставляет методы для выбора метрик и моделей из
    списка(словаря)
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

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
            est = model.fit(self.x, self.y)
            print(self.x)
            pred = model.predict(x_train)
            for metric in metrics:
                survs = est.predict_survival_function(self.x)
                preds = [fn(1825) for fn in survs]

                metric = metric(y_train, y_test, preds)

                print('!!!!!!!!!!!!' + metric.score + '!!!!!!!!')
                results[model.name].append(metric.score)