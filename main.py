from sksurv.metrics import concordance_index_censored

import models
from metrics.main import c_index_censored
from utils.info import Info

from utils.data_preparation import prepare_df
from utils.experiment import Experiment

from models import *

from metrics import *
from utils.report_generation import get_report

if __name__ == '__main__':
    # Подготовка данных
    x, y = prepare_df('/Users/vladimirnikolskiy/Desktop/practice/Диплом/data.csv')

    # Проведение эксперимента
    report = {}
# ---------------- Exp1
    experiment = Experiment(GradientBoostingSurvivalAnalysisModel(learning_rate=1.0, max_depth=1, random_state=0, n_estimators=90), x, y)
    report['GBSA'] = []
    chf_funcs, surv_funcs, y_pred, y_train, y_test = experiment.get_res

    pred_risks = [fn(800) for fn in surv_funcs][:1800]

    '''
    metric1 = BrierScore(y_train, y_test, pred_risks)
    print(f'Brier score: {metric1(800)}')
    report['GBSA'].append((metric1.name, metric1(800)))

    metric2 = CIndexIpcw(y_train, y_test, pred_risks)
    print(f'CindexIpcw: {metric2()}')
    report['GBSA'].append((metric2.name, metric2()))
    '''

    true_times = [y[1] for y in y_test]
    true_events = [y[0] for y in y_test]
    pred_risks = [fn(800) for fn in surv_funcs][:1800]

    metric3 = concordance_index_censored(true_events, true_times, y_pred)[0]
    print(f'CindexCensored: {metric3}')
    report['GBSA'].append(('cindex', metric3))

    print('++++++++++++++++++++++++++')

# ------------------- Exp2
    param = {'alpha': 0.000244140625}
    experiment2 = Experiment(FastSurvivalSVMModel(**param), x, y)
    report['FSSVM'] = []
    chf_funcs, _, y_pred, y_train, y_test = experiment2.get_res

    '''
    metric1 = BrierScore(y_train, y_test, y_pred)
    print(f'Brier score: {metric1(800)}')
    report['FSSVM'].append((metric1.name, metric1(800)))

    metric2 = CIndexIpcw(y_train, y_test, y_pred)
    print(f'CindexIpcw: {metric2()}')
    report['FSSVM'].append((metric2.name, metric2()))
    '''

    true_times = [y[1] for y in y_test]
    true_events = [y[0] for y in y_test]
    pred_risks = [fn(800) for fn in surv_funcs][:1800]

    metric3 = concordance_index_censored(true_events, true_times, pred_risks)[0]

    print(f'CindexCensored: {metric3}')
    report['FSSVM'].append(('cindex', metric3))

    print('++++++++++++++++++++++++++')

# ------------------- Exp3
    experiment3 = Experiment(SurvivalTreeModel(), x, y)
    report['Survival Tree'] = []
    _, _, y_pred, y_train, y_test = experiment3.get_res

    '''
    metric1 = BrierScore(y_train, y_test, y_pred)
    print(f'Brier score: {metric1(800)}')
    report['Survival Tree'].append((metric1.name, metric1(800)))

    metric2 = CIndexIpcw(y_train, y_test, y_pred)
    print(f'CindexIpcw: {metric2()}')
    report['Survival Tree'].append((metric2.name, metric2()))
    '''

    true_times = [y[1] for y in y_test]
    true_events = [y[0] for y in y_test]
    # pred_risks = [fn(800) for fn in surv_funcs][:1800]

    metric3 = concordance_index_censored(true_events, true_times, y_pred)[0]

    print(f'CindexCensored: {metric3}')
    report['Survival Tree'].append(('cindex', metric3))

    print('++++++++++++++++++++++++++')

# ------------------- Exp4
    experiment4 = Experiment(RandomSurvivalForestModel(n_estimators=1000,
                                                       min_samples_split=10,
                                                       min_samples_leaf=15,
                                                       max_features="sqrt",
                                                       n_jobs=-1,
                                                       random_state=1), x, y)
    report['RSF'] = []
    _, surv_funcs, y_pred, y_train, y_test = experiment4.get_res

    '''
    metric1 = BrierScore(y_train, y_test, y_pred)
    print(f'Brier score: {metric1(800)}')
    report['RSF'].append((metric1.name, metric1(800)))

    metric2 = CIndexIpcw(y_train, y_test, y_pred)
    print(f'CindexIpcw: {metric2()}')
    report['RSF'].append((metric2.name, metric2()))
    '''

    true_times = [y[1] for y in y_test]
    true_events = [y[0] for y in y_test]
    pred_risks = [fn(800) for fn in surv_funcs][:1800]

    metric3 = c_index_censored(y_pred, true_times, true_events)
    print(f'CindexCensored: {metric3}')
    report['RSF'].append(('cindex', metric3))

    print('++++++++++++++++++++++++++')

    get_report('test', report)
