from sksurv.svm import MinlipSurvivalAnalysis

from models.base_model import BaseModel


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
        self.model = MinlipSurvivalAnalysis(self,
                                            solver,
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
