import inspect
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class BaseModel(BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit(self, x_train, y_train):
        res = self.model.fit(x_train, y_train)
        return res

    def predict(self, x):
        self.model.predict(x)

    def name(self):
        return self.__class__.__name__

    def params(self):
        return self.params

