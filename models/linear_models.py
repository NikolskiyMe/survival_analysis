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
        model = CoxPHSurvivalAnalysis(alpha,
                                      ties,
                                      n_iter,
                                      tol,
                                      verbose)
        super().__init__(model)

    @property
    def name(self):
        return 'CoxPHSurvivalAnalysis'


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
                 fit_baseline_model=False):
        model = CoxnetSurvivalAnalysis(n_alphas,
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
        super().__init__(model)

    @property
    def name(self):
        return 'CoxnetSurvivalAnalysis'


class IPCRidgeModel(BaseModel):
    def __init__(self,
                 alpha=1.0,
                 fit_intercept=True,
                 normalize=False,
                 copy_X=True,
                 max_iter=None,
                 tol=1e-3,
                 solver="auto"):
        model = IPCRidge(alpha,
                         fit_intercept,
                         normalize,
                         copy_X,
                         max_iter,
                         tol,
                         solver)
        super().__init__(model)

    @property
    def name(self):
        return 'IPCRidge'
