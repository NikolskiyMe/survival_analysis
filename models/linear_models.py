from sksurv.linear_model import (
    CoxPHSurvivalAnalysis,
    CoxnetSurvivalAnalysis,
    IPCRidge
)

from models.base_model import BaseModel


class CoxPHSurvivalAnalysisModel(BaseModel):
    def __init__(self,
                 alpha=0,
                 ties="breslow",
                 n_iter=100,
                 tol=1e-9,
                 verbose=0):
        self.model = CoxPHSurvivalAnalysis(alpha,
                                           ties,
                                           n_iter,
                                           tol,
                                           verbose)

    def fit(self, x_train, y_train):
        return self.model.fit(x_train, y_train)

    def predict(self, x):
        self.model.predict(x)


class CoxnetSurvivalAnalysisModel(BaseModel):
    def __init__(self,
                 n_alphas=100,
                 alphas=None,
                 alpha_min_ratio="auto",
                 l1_ratio=0.5,
                 penalty_factor=None,
                 normalize=False,
                 copy_X=True,
                 tol=1e-7,
                 max_iter=100000,
                 verbose=False,
                 fit_baseline_model=False
                 ):
        self.model = CoxnetSurvivalAnalysis(n_alphas,
                                            alphas,
                                            alpha_min_ratio,
                                            l1_ratio,
                                            penalty_factor,
                                            normalize,
                                            copy_X,
                                            tol,
                                            max_iter,
                                            verbose,
                                            fit_baseline_model
                                            )

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        self.model.predict(x)


class IPCRidgeModel(BaseModel):
    def __init__(self,
                 alpha=1.0,
                 fit_intercept=True,
                 normalize=False,
                 copy_X=True,
                 max_iter=None,
                 tol=1e-3,
                 solver="auto"):
        self.model = IPCRidge(self,
                              alpha,
                              fit_intercept,
                              normalize,
                              copy_X,
                              max_iter,
                              tol,
                              solver)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        self.model.predict(x)