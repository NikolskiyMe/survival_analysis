from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

n_estimators = [i * 5 for i in range(1, 21)]

estimators = {
    "no regularization": GradientBoostingSurvivalAnalysis(
        learning_rate=1.0, max_depth=1, random_state=0
    ),
    "learning rate": GradientBoostingSurvivalAnalysis(
        learning_rate=0.1, max_depth=1, random_state=0
    ),
    "dropout": GradientBoostingSurvivalAnalysis(
        learning_rate=1.0, dropout_rate=0.1, max_depth=1, random_state=0
    ),
    "subsample": GradientBoostingSurvivalAnalysis(
        learning_rate=1.0, subsample=0.5, max_depth=1, random_state=0
    ),
}


class GradientBoostingModel:
    def __init__(self, learning_rate=0.1, n_estimators=100,
                 criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_split=None,
                 min_impurity_decrease=0., random_state=None, max_features=None,
                 max_leaf_nodes=None, subsample=1.0, dropout_rate=0.0,
                 verbose=0, ccp_alpha=0.0):
        self.model = GradientBoostingSurvivalAnalysis(self, learning_rate,
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

    def set_params(self, n):
        self.model.set_params(n_estimators=n)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict_survival_function(self):
        self.model.predict_survival_function(self.x_test)

    # ToDo: @use_metrics([...])
    def get_score(self):
        cindex = self.model.score(self.x_test, self.y_test)

        return round(cindex, 3)
