import numpy
from sksurv.metrics import concordance_index_censored

from utils.errors import NoComparablePairException


def c_index_censored(pred_risks, true_times, true_events):
    cindex, _, _, _, _ = concordance_index_censored(true_events, true_times, pred_risks)
    return cindex


class MyMetric:
    def __init__(self, magic_param):
        self.magic_param = magic_param

    def __call__(self, pred_risks, true_times, true_events):
        return self.magic_param + sum(
            [p - t for p, t in zip(pred_risks, true_times)])