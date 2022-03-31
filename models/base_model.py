from abc import ABC, abstractmethod


class BaseModel:
    """
    Базовый класс для всех моделей
    """
    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def name(self):
        return self.__class__.__name__
