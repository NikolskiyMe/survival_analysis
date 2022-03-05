from sksurv.svm import NaiveSurvivalSVM

from models.base_model import BaseModel


class NaiveSurvivalSVMModel(BaseModel):
    def __init__(self,
                 penalty='l2',
                 loss='squared_hinge',
                 dual=False,
                 tol=1e-4,
                 alpha=1.0,
                 verbose=0,
                 random_state=None,
                 max_iter=1000):
        self.model = NaiveSurvivalSVM(self,
                                      penalty,
                                      loss,
                                      dual,
                                      tol,
                                      alpha,
                                      verbose,
                                      random_state,
                                      max_iter)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        self.model.predict(x)
