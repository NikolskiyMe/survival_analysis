from sksurv.svm import (
    FastKernelSurvivalSVM,
    FastSurvivalSVM,
    HingeLossSurvivalSVM,
    MinlipSurvivalAnalysis,
    NaiveSurvivalSVM
)

from models.base_model import BaseModel


class FastKernelSurvivalSVMModel(FastKernelSurvivalSVM):
    @property
    def name(self):
        return 'FastKernelSurvivalSVM'


class FastSurvivalSVMModel(FastSurvivalSVM):
    @property
    def name(self):
        return 'FastSurvivalSVM'


class HingeLossSurvivalSVMModel(HingeLossSurvivalSVM):
    @property
    def name(self):
        return 'HingeLossSurvivalSVM'


class MinlipSurvivalAnalysisModel(MinlipSurvivalAnalysis):
    @property
    def name(self):
        return 'MinlipSurvivalAnalysis'


class NaiveSurvivalSVMModel(NaiveSurvivalSVM):
    @property
    def name(self):
        return 'NaiveSurvivalSVM'
