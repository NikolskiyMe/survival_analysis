from sksurv.ensemble import (
    ComponentwiseGradientBoostingSurvivalAnalysis,
    ExtraSurvivalTrees,
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest
)

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


class ExtraSurvivalTreesModel(BaseModel):
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=6,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 max_samples=None):
        self.model = ExtraSurvivalTrees(self, n_estimators,
                                        max_depth,
                                        min_samples_split,
                                        min_samples_leaf,
                                        min_weight_fraction_leaf,
                                        max_features,
                                        max_leaf_nodes,
                                        bootstrap,
                                        oob_score,
                                        n_jobs,
                                        random_state,
                                        verbose,
                                        warm_start,
                                        max_samples)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        self.model.predict(x)


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


class RandomSurvivalForestModel(BaseModel):
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=6,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 max_samples=None):
        self.model = RandomSurvivalForest(n_estimators,
                                          max_depth,
                                          min_samples_split,
                                          min_samples_leaf,
                                          min_weight_fraction_leaf,
                                          max_features,
                                          max_leaf_nodes,
                                          bootstrap,
                                          oob_score,
                                          n_jobs,
                                          random_state,
                                          verbose,
                                          warm_start,
                                          max_samples)

    def fit(self, x_train, y_train):
        return self.model.fit(x_train, y_train)

    def predict(self, x):
        self.model.predict(x)


