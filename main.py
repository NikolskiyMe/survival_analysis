from utils.info import Info

from utils.data_preparation import prepare_df
from utils.experiment import Experiment
# from utils.cross_validation import Validation

from models import *

from metrics import *


if __name__ == '__main__':

    i = Info()
    print(i.metrics)
    print(i.models)

    # Подготовка данных
    x, y = prepare_df('/Users/vladimirnikolskiy/Desktop/practice/Диплом/data.csv')

    # Проведение эксперимента
    experiment = Experiment(x, y)
    models = [GradientBoostingSurvivalAnalysisModel,
              ExtraSurvivalTreesModel,
              RandomSurvivalForestModel
              ]
    metrics = [BrierScore]
    experiment.start(models, metrics)  # генерация отчета
