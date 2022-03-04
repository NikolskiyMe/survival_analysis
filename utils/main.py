def get_models_list() -> None:
    models_list = {1: 'GradientBoostingSurvivalAnalysis',
                   2: 'RandomSurvivalForest',
                   3: 'FastSurvivalSVM'}
    print(models_list)


def get_metrics_list() -> None:
    metrics_list = {1: 'brier_score',
                    2: 'concordance_index_censored',
                    3: 'integrated_brier_score'}
    print(metrics_list)


class Main:
    """
    Данный класс предоставляет методы для выбора метрик и моделей из
    списка(словаря)
    """

    def select_models(self, *args):
        """
        Функция "select_models" получает на вход номера моделей, после чего
        предлагает задать ее гиперпараметры
        """
        pass

    def select_metrics(self, *args):
        """
        Функция "select_metrics" получает на вход номера метрик, после чего
        производит сравнение по заданным метрикам + время обучения моделей (?),
        выбранных в "select_models", и заносит полученные сравнения в отчет
        """
        pass
