from utils import cp_detector as cpd
from utils import data_generation as gen
from utils import statistic_calculation as stat
from utils import threshold_calculation as th

import matplotlib.pyplot as plt
import numpy as np
import pickle

lin_stat_alpha = {}
lin_stat_alpha_ln = {}
lin_stat_beta = {}
lin_stat_beta_ln = {}

dataset_parameters = {
    "dataset_size": 100,
    "seq_len": 101,
    "d": 10,
    "p": 5,
    "distribution": "t-distribution",
    "k": 2.5,
    "cp": None,
    "nu": 10
}

cp_parameters = {
    "alpha": 0.05,
    "scan": False,
    "data_based": False,
    "ln": False
}

for d in range(10, 1000, 10):
    print(d)
    dataset_parameters["d"] = d
    (fn, fp), (fn_ln, fp_ln) = cpd.cpd_with_ln_compare(
        dataset_parameters, cp_parameters, data_type="t-distribution"
    )
    lin_stat_alpha[d] = fp
    lin_stat_alpha_ln[d] = fp_ln
    lin_stat_beta[d] = fn
    lin_stat_beta_ln[d] = fn_ln

with open('alpha.pickle', 'wb') as f:
    pickle.dump(lin_stat_alpha, f)
with open('alpha_ln.pickle', 'wb') as f:
    pickle.dump(lin_stat_alpha_ln, f)
with open('beta.pickle', 'wb') as f:
    pickle.dump(lin_stat_beta, f)
with open('beta_ln.pickle', 'wb') as f:
    pickle.dump(lin_stat_beta_ln, f)