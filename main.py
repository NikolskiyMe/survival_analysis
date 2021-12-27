from data_preparation.main import prepare_df, get_xy
from models.gradient_boost.main import GradientBoostingModel
from models.ssvm.main import SSVMModel


def best_score(x, y):
    ssvm_score = 0
    #ssvm = SSVMModel(x, y)
    #ssvm_score = ssvm.get_score()

    gb_score = 0
    gb = GradientBoostingModel(x, y)
    gb_score = gb.get_score()

    return max(ssvm_score, gb_score)


if __name__ == '__main__':

    # 1. Загрузка данных
    data_path = input('Начало работы. Введите путь до файла(.csv) с данными: ')

    # 2. Подготовка данных
    data_frame = prepare_df(data_path)
    x, y = get_xy(data_frame)

    # 3. Определяем модель(с параметрами), которая дает лучший score
    print('В работе...')
    best_model_score = best_score(x, y)

    print(f'Лучший C-index: {best_model_score}')

