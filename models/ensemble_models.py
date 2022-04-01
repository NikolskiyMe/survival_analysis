from sksurv.ensemble import (
    ComponentwiseGradientBoostingSurvivalAnalysis,
    ExtraSurvivalTrees,
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest
)

from models.base_model import BaseModel


class ComponentwiseGradientBoostingSurvivalAnalysisModel(BaseModel):
    def __init__(self, loss="coxph", learning_rate=0.1,
                 n_estimators=100, subsample=1.0, dropout_rate=0,
                 random_state=None, verbose=0):
        model = ComponentwiseGradientBoostingSurvivalAnalysis(loss,
                                                              learning_rate,
                                                              n_estimators,
                                                              subsample,
                                                              dropout_rate,
                                                              random_state,
                                                              verbose)
        super().__init__(model)

    @property
    def name(self):
        return 'ComponentwiseGradientBoostingSurvivalAnalysis'


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
        model = ExtraSurvivalTrees(n_estimators,
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
        super().__init__(model)

    @property
    def name(self):
        return 'ExtraSurvivalTrees'


class GradientBoostingSurvivalAnalysisModel(BaseModel):
    def __init__(self,
                 loss="coxph",
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
        model = GradientBoostingSurvivalAnalysis(loss,
                                                 learning_rate,
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
        super().__init__(model)

    @property
    def name(self):
        return 'GradientBoostingSurvivalAnalysis'


"""
class GBSA:
    def __init__(self,
                 loss="coxph",
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
        params = [loss,
                  learning_rate,
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
                  ccp_alpha]
        self.params = {}
"""


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
        model = RandomSurvivalForest(n_estimators,
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
        super().__init__(model)

    @property
    def name(self):
        return 'RandomSurvivalForest'
