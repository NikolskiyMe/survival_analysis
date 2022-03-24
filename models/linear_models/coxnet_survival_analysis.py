from sksurv.linear_model import CoxnetSurvivalAnalysis

from models.base_model import BaseModel


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
