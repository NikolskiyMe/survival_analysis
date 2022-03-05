from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

from models.base_model import BaseModel


class ComponentwiseGradientBoostingSurvivalAnalysisModel(BaseModel):
    def __init__(self,
                 loss="coxph",
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 dropout_rate=0,
                 random_state=None,
                 verbose=0):
        self.model = ComponentwiseGradientBoostingSurvivalAnalysis(self,
                                                                   loss,
                                                                   learning_rate,
                                                                   n_estimators,
                                                                   subsample,
                                                                   dropout_rate,
                                                                   random_state,
                                                                   verbose)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        self.model.predict(x)

