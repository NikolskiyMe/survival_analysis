from utils.data_preparation import prepare_df
from utils.info import Info
from utils.main import Experiment

from models import *

from metrics.main import (
    BrierScore,
    ConcordanceIndexCensored,
    ConcordanceIndexIpcw,
    CumulativeDynamicAuc,
    IntegratedBrierScore
)


if __name__ == '__main__':

    i = Info()
    print(i.metrics)
    print(i.models)

    # Подготовка данных
    x, y = prepare_df('/Users/vladimirnikolskiy/Desktop/practice/Диплом/data.csv')

    # while True:
    experiment = Experiment(x, y)
    models = [CoxPHSurvivalAnalysisModel()]
    metrics = [ConcordanceIndexIpcw]
    experiment.start(models, metrics)
