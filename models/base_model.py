from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Абстрактный базовый класс для всех моделей
    """
    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

