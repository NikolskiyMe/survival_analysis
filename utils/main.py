from sklearn.model_selection import train_test_split


class Experiment:
    """
    Данный класс предоставляет методы для выбора метрик и моделей из
    списка(словаря)
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def start(self, models, metrics):
        results = {}
        for model in models:
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,
                                                                test_size=0.1,
                                                                random_state=1)
            model.fit(x_train, y_train)

            model.predict(x_train)
            for metric in metrics:
                if metric in group1:
                    "тут передаются y_train и y_test, заносим в таблицу"
                    result = metric(y_train, y_test)
                    pass
                elif metric in group2:
                    "тут предсказываются ф-ии, отрисовать их в отчет"
                    pass
                else:
                    "тут cindex censored"
                    pass
                result = metric
                results[model.name] = result
