from sklearn.model_selection import train_test_split

from sksurv.ensemble import RandomSurvivalForest


class RandomSF:
    def __init__(self, x, y, *params):
        test_size = 0.2
        random_state = 1
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size,
            random_state=random_state)
        self.model = RandomSurvivalForest(params)

    def set_params(self, n):
        self.model.set_params(n_estimators=n)

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def predict_survival_function(self):
        self.model.predict_survival_function(self.x_test)

    # ToDo: @use_metrics([...])
    def get_score(self):
        cindex = self.model.score(self.x_test, self.y_test)

        return round(cindex, 3)