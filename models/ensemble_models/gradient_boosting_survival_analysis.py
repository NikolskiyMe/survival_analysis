from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from models.base_model import BaseModel


class GradientBoostingSurvivalAnalysisModel(BaseModel):
    def __init__(self,
                 learning_rate=0.1,
                 n_estimators=100,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_split=None,
                 min_impurity_decrease=0.,
                 random_state=None,
                 max_features=None,
                 max_leaf_nodes=None,
                 subsample=1.0,
                 dropout_rate=0.0,
                 verbose=0,
                 ccp_alpha=0.0):
        self.model = GradientBoostingSurvivalAnalysis(learning_rate,
                                                      n_estimators,
                                                      criterion,
                                                      min_samples_split,
                                                      min_samples_leaf,
                                                      min_weight_fraction_leaf,
                                                      max_depth,
                                                      min_impurity_split,
                                                      min_impurity_decrease,
                                                      random_state,
                                                      max_features,
                                                      max_leaf_nodes,
                                                      subsample,
                                                      dropout_rate,
                                                      verbose,
                                                      ccp_alpha)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        self.model.predict(x)

    def __repr__(self):
        return 'БУСТИНГ'
