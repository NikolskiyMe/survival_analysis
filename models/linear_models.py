from sksurv.linear_model import (
    CoxPHSurvivalAnalysis,
    CoxnetSurvivalAnalysis,
    IPCRidge
)

from models.base_model import BaseModel


class CoxPHSurvivalAnalysisModel(CoxPHSurvivalAnalysis):
    @property
    def name(self):
        return 'CoxPHSurvivalAnalysis'


class CoxnetSurvivalAnalysisModel(CoxnetSurvivalAnalysis):
    @property
    def name(self):
        return 'CoxnetSurvivalAnalysis'


class IPCRidgeModel(IPCRidge):
    @property
    def name(self):
        return 'IPCRidge'
