from sklearn.model_selection import train_test_split

from models.ensemble_models.gradient_boosting_survival_analysis import GradientBoostingModel
from models.ensemble_models.random_survival_forest import RandomSF
from utils.data_preparation import prepare_df
from utils.metrics import Score

if __name__ == '__main__':

    # --- ЗАГРУЗКА ДАННЫХ ---
    x, y = prepare_df('datapath')  # подготовка датасета
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

    report = Report()
    report.view()
    score = Score()

    # --- ПОСТРОЕНИЕ МОДЕЛЕЙ ---
    # Python
    # --- --- gradient boosting --- ---
    boosting = GradientBoostingModel('params')
    # boosting.info() -> информация о методах/метриках
    boosting.fit(x_train, y_train)
    score(boosting, 'metric_name')  # печать промежуточного результата
    boosting.add_to_report()  # добавляет в отчет результаты

    # --- --- random survival forest --- ---
    random_forest = RandomSF('params')
    random_forest.fit(x_train, y_train)
    random_forest.add_to_report()

    # --- --- ssvm --- ---

    # R
    # model = RModel('params')

    # --- МЕТРИКИ ---

    # --- ГЕНЕРАЦИЯ ОТЧЕТА ---
    report.save()  # Сохранение в pdf


