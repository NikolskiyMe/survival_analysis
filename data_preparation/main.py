from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import get_dummies


def prepare_df(path: str) -> tuple[Any, ndarray]:

    df = pd.read_csv(path, delimiter=",")

    """

    df = df.drop('bmi', axis=1)
    df = df.drop('EGFR', axis=1)
    df = df.drop('ALK/ROS1', axis=1)
    df = df.drop('PD-L1', axis=1)

    label_encoder = LabelEncoder()

    label_encoder.fit(df['stage'])
    df['stage'] = label_encoder.transform(df['stage'])

    label_encoder.fit(df['t'])
    df['t'] = label_encoder.transform(df['t'])

    label_encoder.fit(df['n'])
    df['n'] = label_encoder.transform(df['n'])

    label_encoder.fit(df['m'])
    df['m'] = label_encoder.transform(df['m'])
    """

    df['gender'] = df['gender'].map({'мужской': 0, 'женский': 1})

    df = get_dummies(df)
    """
    pandas.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)[source]

    data: Может быть типом массива, типом серии, типом DataFrame

    prefix: Это может быть строка, список строк или словарный тип строк. Значение по умолчанию - Нет. Сопоставьте имя столбца данных со строкой или словарем префикса;

    drop_first:Boolean, по умолчанию False, указывает, следует ли удалить первый столбец.
    """

    array = df.values
    x = array[:, 2:]
    y = array[:, :2]
    y[:, 1], y[:, 0] = y[:, 0].copy(), y[:, 1].copy()
    dt = [('Status', '?'), ('Survival_in_days', '<f8')]
    y = np.array([tuple(i) for i in y], dtype=dt)

    return x, y


