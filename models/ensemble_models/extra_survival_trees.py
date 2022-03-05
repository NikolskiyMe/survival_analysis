from sksurv.ensemble import ExtraSurvivalTrees

from models.base_model import BaseModel


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

