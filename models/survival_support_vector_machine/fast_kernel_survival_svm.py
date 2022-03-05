from sksurv.svm import FastKernelSurvivalSVM

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
