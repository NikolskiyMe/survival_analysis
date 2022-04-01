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
        model = FastKernelSurvivalSVM(alpha,
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
        super().__init__(model)

    @property
    def name(self):
        return 'FastKernelSurvivalSVM'


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
        model = FastSurvivalSVM(alpha,
                                rank_ratio,
                                fit_intercept,
                                max_iter,
                                verbose,
                                tol,
                                optimizer,
                                random_state,
                                timeit)
        super().__init__(model)

    @property
    def name(self):
        return 'FastSurvivalSVM'


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
        model = HingeLossSurvivalSVM(solver,
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
        super().__init__(model)

    @property
    def name(self):
        return 'HingeLossSurvivalSVM'


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
        model = MinlipSurvivalAnalysis(solver,
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
        super().__init__(model)

    @property
    def name(self):
        return 'MinlipSurvivalAnalysis'


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
        model = NaiveSurvivalSVM(penalty,
                                 loss,
                                 dual,
                                 tol,
                                 alpha,
                                 verbose,
                                 random_state,
                                 max_iter)
        super().__init__(model)

    @property
    def name(self):
        return 'NaiveSurvivalSVM'
