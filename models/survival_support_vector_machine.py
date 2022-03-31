from sksurv.svm import (
    FastKernelSurvivalSVM,
    FastSurvivalSVM,
    HingeLossSurvivalSVM,
    MinlipSurvivalAnalysis,
    NaiveSurvivalSVM
)

from models.base_model import BaseModel


class FastKernelSurvivalSVMModel(BaseModel):
    def __init__(self,
                 alpha=1,
                 rank_ratio=1.0,
                 fit_intercept=False,
                 kernel="rbf",
                 gamma=None,
                 degree=3,
                 coef0=1,
                 kernel_params=None,
                 max_iter=20,
                 verbose=False,
                 tol=None,
                 optimizer=None,
                 random_state=None,
                 timeit=False):
        self.model = FastKernelSurvivalSVM(self,
                                           alpha,
                                           rank_ratio,
                                           fit_intercept,
                                           kernel,
                                           gamma,
                                           degree,
                                           coef0,
                                           kernel_params,
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
        self.model = FastSurvivalSVM(alpha,
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


class HingeLossSurvivalSVMModel(BaseModel):
    def __init__(self,
                 solver="ecos",
                 alpha=1.0,
                 kernel="linear",
                 gamma=None,
                 degree=3,
                 coef0=1,
                 kernel_params=None,
                 pairs="all",
                 verbose=False,
                 timeit=None,
                 max_iter=None):
        self.model = HingeLossSurvivalSVM(solver,
                                          alpha,
                                          kernel,
                                          gamma,
                                          degree,
                                          coef0,
                                          kernel_params,
                                          pairs,
                                          verbose,
                                          timeit,
                                          max_iter)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        self.model.predict(x)


class MinlipSurvivalAnalysisModel(BaseModel):
    def __init__(self,
                 solver="ecos",
                 alpha=1.0,
                 kernel="linear",
                 gamma=None,
                 degree=3,
                 coef0=1,
                 kernel_params=None,
                 pairs="nearest",
                 verbose=False,
                 timeit=None,
                 max_iter=None):
        self.model = MinlipSurvivalAnalysis(solver,
                                            alpha,
                                            kernel,
                                            gamma,
                                            degree,
                                            coef0,
                                            kernel_params,
                                            pairs,
                                            verbose,
                                            timeit,
                                            max_iter)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        self.model.predict(x)


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
        self.model = NaiveSurvivalSVM(penalty,
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