from sklearn.model_selection import train_test_split


class BaseMLClass:
    def __init__(self, model, x_train, x_test, y_train, y_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def get_score(self):
        self.model.fit(self.x_train, self.y_train)
        cindex = self.model.score(self.x_test, self.y_test)

        return cindex
