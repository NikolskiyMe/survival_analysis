from sksurv.ensemble import (
    ComponentwiseGradientBoostingSurvivalAnalysis,
    ExtraSurvivalTrees,
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest
)

from models.base_model import BaseModel


class ComponentwiseGradientBoostingSurvivalAnalysisModel(ComponentwiseGradientBoostingSurvivalAnalysis):
    @property
    def name(self):
        return 'ComponentwiseGradientBoostingSurvivalAnalysis'


class ExtraSurvivalTreesModel(ExtraSurvivalTrees):
    @property
    def name(self):
        return 'ExtraSurvivalTrees'


class GradientBoostingSurvivalAnalysisModel(GradientBoostingSurvivalAnalysis):
    @property
    def name(self):
        return 'GradientBoostingSurvivalAnalysis'


class RandomSurvivalForestModel(RandomSurvivalForest):
    @property
    def name(self):
        return 'RandomSurvivalForest'
