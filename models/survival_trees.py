from sksurv.tree import SurvivalTree

from models.base_model import BaseModel


class SurvivalTreeModel(BaseModel):
    def __init__(self,
                 splitter="best",
                 max_depth=None,
                 min_samples_split=6,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None):
        model = SurvivalTree(splitter,
                             max_depth,
                             min_samples_split,
                             min_samples_leaf,
                             min_weight_fraction_leaf,
                             max_features,
                             random_state,
                             max_leaf_nodes)
        super().__init__(model)

    @property
    def name(self):
        return 'SurvivalTree'
