from sksurv.tree import SurvivalTree

from models.base_model import BaseModel


class SurvivalTreeModel(SurvivalTree):
    @property
    def name(self):
        return 'SurvivalTree'
