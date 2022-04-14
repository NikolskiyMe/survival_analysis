from utils.data_preparation import prepare_df
from utils.experiment import Experiment

from models import *
from metrics import *


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

    # Множество метрик
    METRICS = [
        MyCIndex(n_samples=800, tied_tol=1e-8),
    ]

    # --- Конфигурация эксперимента ---
    experiment_1 = Experiment(X, y)
    #       --- Обязательно ---
    experiment_1.tts()  # train_test_split
    experiment_1.models = MODELS

    #       --- Конфигурация метрик ---
    experiment_1.metrics = METRICS  # default - c_index censored
    #       --- Опционально ---
    experiment_1.cross_validation()  # Если вызываем, то будет cv
    experiment_1.hyperparameters_search()  # Если вызываем, то будет hs
    #       --- Запуск ---
    experiment_1.run(in_report=True)
