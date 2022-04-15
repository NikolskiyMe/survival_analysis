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
        # optimize(GradientBoostingSurvivalAnalysisModel, params={'params...'}, n_splits=100, test_size=0.5 ...),
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
        # MyCIndexIPCW(tau=None, tied_tol=1e-08),
        # MyCumulativeDynamicAuc(times=800, tied_tol=1e-08), не работает
        # MyIntegratedBrierScore(times=(300, 450))
    ]

    experiment_1 = Experiment(test_size=0.2, num_of_repeat=5)
    result = experiment_1.run(X=X, y=y, models=MODELS, metrics=METRICS)
    # result -> сериализовать через json.dumps() / json.loads()

    # print_report(result)
    # make_pdf('test_new', result)
