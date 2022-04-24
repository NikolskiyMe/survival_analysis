import numpy as np

from utils.data_preparation import prepare_df

from utils.experiment import Experiment, ExperimentCV

from utils.hyperparameter_search import Optimize

from models import *
from metrics import *


from utils.report import make_pdf, print_report

if __name__ == '__main__':
    X, y = prepare_df(
        '/Users/vladimirnikolskiy/Desktop/practice/Диплом/data.csv'
    )

    METHODS = [
        Optimize(estimator=FastSurvivalSVMModel(max_iter=1000,
                                                tol=1e-5,
                                                random_state=0),
                 param_grid={'alpha': 2. ** np.arange(-12, 13, 2)},
                 n_splits=10,
                 ),
        GradientBoostingSurvivalAnalysisModel(
            learning_rate=1.0,
            max_depth=1,
            random_state=0,
            n_estimators=90
        ),
        FastSurvivalSVMModel(),
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

    METHODS_TEST = [
        SurvivalTreeModel(),
        GradientBoostingSurvivalAnalysisModel(
            learning_rate=1.0,
            max_depth=1,
            random_state=0,
            n_estimators=90
        )
    ]

    # ToDo: не работают закомментирванные метрики
    METRICS = [
        MyCIndex(tied_tol=1e-8),
        # MyBrierScore(),
        MyCIndexIPCW(tau=None, tied_tol=1e-08),
        # MyCumulativeDynamicAuc(times=800, tied_tol=1e-08),
        # MyIntegratedBrierScore(times=(300, 450))
    ]

    # experiment_1 = Experiment(test_size=0.2, num_of_repeat=1)
    # result_1 = experiment_1.run(X=X, y=y, models=METHODS, metrics=METRICS)

    experiment_cv = ExperimentCV()
    result_2 = experiment_cv.run(X=X, y=y, models=METHODS_TEST, metrics=METRICS)

    # print_report(result_1)
    print_report(result_2)
