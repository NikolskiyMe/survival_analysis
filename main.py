import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.svm import FastSurvivalSVM

from utils.data_preparation import prepare_df
from utils.experiment import Experiment

from models import *
from metrics import *
from utils.hyperparameter_search import Optimize
from utils.report import make_pdf, print_report

if __name__ == '__main__':
    X, y = prepare_df(
        '/Users/vladimirnikolskiy/Desktop/practice/Диплом/data.csv'
    )

    # METHODS
    MODELS = [
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

    # Конфигурация метрик
    METRICS = [
        MyCIndex(tied_tol=1e-8),
        # MyBrierScore(),
        # MyCIndexIPCW(tau=None, tied_tol=1e-08),
        # MyCumulativeDynamicAuc(times=800, tied_tol=1e-08),
        # MyIntegratedBrierScore(times=(300, 450))
    ]

    experiment_1 = Experiment(test_size=0.2, num_of_repeat=5)
    result = experiment_1.run(X=X, y=y, models=MODELS, metrics=METRICS)

    # print_report(result)
    # make_pdf('test_new', result)
