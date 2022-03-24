from sksurv.linear_model import CoxPHSurvivalAnalysis

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
