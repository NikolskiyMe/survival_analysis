from utils.data_preparation import prepare_df
from utils.experiment import Experiment

from models import *
from metrics import *
from utils.report import make_pdf, print_report

if __name__ == '__main__':

    X, y = prepare_df(
        '/Users/vladimirnikolskiy/Desktop/practice/Диплом/data.csv'
    )

    # Конфигурация моделей
    MODELS = [
        GradientBoostingSurvivalAnalysisModel(
            learning_rate=1.0,
            max_depth=1,
            random_state=0,
            n_estimators=90
        ),
        SurvivalTreeModel(),
        RandomSurvivalForestModel(
            n_estimators=1000,
            min_samples_split=10,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=1
        ),
    ]

    # Конфигурация метрик
    METRICS = [
        MyCIndex(tied_tol=1e-8),
        # MyBrierScore(), не работает,
        MyCIndexIPCW(tau=None, tied_tol=1e-08),
        # MyCumulativeDynamicAuc(times=800, tied_tol=1e-08), не работает
        MyIntegratedBrierScore(times=(300, 450))
    ]

    experiment_1 = Experiment(X, y, test_size=0.2, random_state=1)  # test_size и random_state - опциональные
    experiment_1.hyperparameters_search()  # Если вызываем, то будет hs ToDo
    result = experiment_1.run(MODELS, METRICS)

    print_report(result)
    # make_pdf('test_new', result)

    # experiment_2 = ExperimentCV()
