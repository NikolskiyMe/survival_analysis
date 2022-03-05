from sksurv.linear_model import IPCRidge

from models.base_model import BaseModel


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
