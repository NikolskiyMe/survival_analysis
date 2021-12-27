import numpy as np
import pandas as pd
import pandas.core.frame
from sklearn.preprocessing import LabelEncoder


def prepare_df(path: str) -> pandas.core.frame.DataFrame:

    df = pd.read_csv(path, delimiter=",")
    df['gender'] = df['gender'].map({'мужской': 0, 'женский': 1})

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

    return df


def get_xy(df):
    array = df.values
    x = array[:, 2:]
    y = array[:, :2]
    y[:, 1], y[:, 0] = y[:, 0].copy(), y[:, 1].copy()
    dt = dtype = [('Status', '?'), ('Survival_in_days', '<f8')]
    y = np.array([tuple(i) for i in y], dtype=dt)

    return x, y

