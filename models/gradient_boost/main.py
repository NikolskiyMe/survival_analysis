from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from utils.base import BaseMLClass

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


class GradientBoostingModel(BaseMLClass):
    def __init__(self, x, y):
        test_size = 0.2
        random_state = 1
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size,
            random_state=random_state)

    def get_score(self):
        best_score = 0

        for n in n_estimators:
            for name, est in estimators.items():
                est.set_params(n_estimators=n)
                est.fit(self.x_train, self.y_train)
                cindex = est.score(self.x_test, self.y_test)

                if cindex > best_score:
                    best_score = cindex

        return round(best_score, 3)



