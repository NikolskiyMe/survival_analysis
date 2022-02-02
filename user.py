from data_preparation.main import prepare_df  # data_path -> x, y

from models.gradient_boost.main import GradientBoostingModel
from models.ssvm.main import SSVMModel

from models.best_model import model_cmp  # comparison 2+ S-A models

import sys


if __name__ == '__main__':

    # 1. Загрузка данных
    if len(sys.argv) != 2:
        print('Неверное количество аргументов')
    else:
        data_path = sys.argv[1]

        # 2. Подготовка данных
        x, y = prepare_df(data_path)

        # 3. Определяем объекты моделей и получаем оценки

        print('В работе...')

    # ToDo: метод для предложения лучшей модели под конкретный набор данных

