from sksurv.svm import FastSurvivalSVM

from models.base_model import BaseModel


class FastSurvivalSVMModel(BaseModel):
    def __init__(self,
                 alpha=1,
                 rank_ratio=1.0,
                 fit_intercept=False,
                 max_iter=20,
                 verbose=False,
                 tol=None,
                 optimizer=None,
                 random_state=None,
                 timeit=False):
        self.model = FastSurvivalSVM(self,
                                     alpha,
                                     rank_ratio,
                                     fit_intercept,
                                     max_iter,
                                     verbose,
                                     tol,
                                     optimizer,
                                     random_state,
                                     timeit)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        self.model.predict(x)
