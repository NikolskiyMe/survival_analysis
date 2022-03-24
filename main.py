from utils.data_preparation import prepare_df
from utils.info import Info
from utils.main import Experiment

from models.linear_models.cox_ph_survival_analysis import CoxPHSurvivalAnalysisModel
from models.linear_models.coxnet_survival_analysis import CoxnetSurvivalAnalysisModel
from models.ensemble_models.random_survival_forest import RandomSurvivalForestModel

from models.ensemble_models.gradient_boosting_survival_analysis import GradientBoostingSurvivalAnalysisModel

from metrics.main import ConcordanceIndexIpcw

if __name__ == '__main__':

    i = Info()
    print(i.metrics)
    print(i.models)

    x, y = prepare_df('/Users/vladimirnikolskiy/Desktop/practice/Диплом/data.csv')

    # while True:
    experiment = Experiment(x, y)
    models = [CoxPHSurvivalAnalysisModel()]
    metrics = [ConcordanceIndexIpcw]
    experiment.start(models, metrics)
