from omegaconf import DictConfig
from utils import data_generation as gen
from utils import cp_detector as cpd
from pathlib import Path
import torch.nn as nn
import torch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


def compare_with_ln(config: DictConfig):

    dataset_parameters = config.dataset
    cp_parameters = config.model

    stat_alpha = {}
    stat_alpha_ln = {}
    stat_beta = {}
    stat_beta_ln = {}

    for d in range(100, 1000, 100):
    #for d in [100, 250, 500, 1000]:
        dataset_parameters["d"] = d
        (fn, fp), (fn_ln, fp_ln) = cpd.cpd_with_ln_compare(
            dataset_parameters, cp_parameters, data_type="t-distribution"
        )
        stat_alpha[d] = fp
        stat_alpha_ln[d] = fp_ln
        stat_beta[d] = fn
        stat_beta_ln[d] = fn_ln


    with open('alpha.pickle', 'wb') as f:
        pickle.dump(stat_alpha, f)
    with open('alpha_ln.pickle', 'wb') as f:
        pickle.dump(stat_alpha_ln, f)
    with open('beta.pickle', 'wb') as f:
        pickle.dump(stat_beta, f)
    with open('beta_ln.pickle', 'wb') as f:
        pickle.dump(stat_beta_ln, f)

    return (stat_beta, stat_alpha), (stat_beta_ln, stat_alpha_ln)

def evaluate_threshold(config: DictConfig):
    dataset_parameters = config.dataset
    w = 5
    data_with_cp, data_without_cp, t_cp_idxs = gen.generate_data(**dataset_parameters)

    layer_norm = nn.LayerNorm(dataset_parameters["d"], elementwise_affine=False).float()
    data_with_cp_ln = (
        layer_norm(torch.from_numpy(data_with_cp).float()).detach().numpy()
    )
    data_without_cp_ln = (
        layer_norm(torch.from_numpy(data_without_cp).float()).detach().numpy()
    )

    delta_sic_with, likelihood_cp_with = cpd.t_dist_data_inference(data_with_cp, dataset_parameters)
    delta_sic_without, likelihood_cp_without = cpd.t_dist_data_inference(data_without_cp, dataset_parameters)

    delta_sic_with_ln, likelihood_cp_with_ln = cpd.t_dist_data_inference(data_with_cp_ln, dataset_parameters)
    delta_sic_without_ln, likelihood_cp_without_ln = cpd.t_dist_data_inference(data_without_cp_ln, dataset_parameters)
    
    min_delta = int(round(min(delta_sic_with.min(), delta_sic_without.min(), delta_sic_with_ln.min(), delta_sic_without_ln.min()), 0))
    max_delta = int(round(max(delta_sic_with[:, w:-w].max(), delta_sic_without[:, w:-w].max(), delta_sic_with_ln[:, w:-w].max(), delta_sic_without_ln[:, w:-w].max()), 0))

    fn_threshold = {}
    fp_threshold = {}

    fn_threshold_ln = {}
    fp_threshold_ln = {}


    for th in range(min_delta, max_delta, 1):
        detected_cp_idxs = np.where(delta_sic_with.min(1) > th, -1, likelihood_cp_with.argmin(1))
        detected_false_cp_idxs = np.where(delta_sic_without.min(1) > th, -1, likelihood_cp_without.argmin(1))
        fn, fp = cpd.calculate_error(detected_cp_idxs, detected_false_cp_idxs)
        fn_threshold[th] = fn
        fp_threshold[th] = fp

        detected_cp_idxs_ln = np.where(delta_sic_with_ln.min(1) > th, -1, likelihood_cp_with_ln.argmin(1))
        detected_false_cp_idxs_ln = np.where(delta_sic_without_ln.min(1) > th, -1, likelihood_cp_without_ln.argmin(1))
        fn_ln, fp_ln = cpd.calculate_error(detected_cp_idxs_ln, detected_false_cp_idxs_ln)
        fn_threshold_ln[th] = fn_ln
        fp_threshold_ln[th] = fp_ln

    # save metrics
    path_to_pickle = Path()
    path_to_pickle.mkdir(parents=True, exist_ok=True)

    save_errors(fn_threshold, fp_threshold, fn_threshold_ln, fp_threshold_ln, path_to_pickle)

    plt.plot(
        [value for value in fp_threshold.values()],
        [value for value in fn_threshold.values()],
        #"o",
        label="Without LN"
    )

    plt.plot(
        [value for value in fp_threshold_ln.values()],
        [value for value in fn_threshold_ln.values()],
        #"s",
        label="WITH LN"
    )

    plt.xlabel("False alarm rate")
    plt.ylabel("False negative rate");
    plt.legend()
    plt.savefig(os.path.join(Path()/'res.png'), dpi=300, bbox_tight=True)
    plt.close()

    return fn_threshold, fp_threshold, fn_threshold_ln, fp_threshold_ln

def save_errors(fn, fp, fn_ln, fp_ln, path_to_save):
    with open(os.path.join(path_to_save, 'alpha.pickle'), 'wb') as f:
        pickle.dump(fp, f)
    with open(os.path.join(path_to_save, 'alpha_ln.pickle'), 'wb') as f:
        pickle.dump(fp_ln, f)
    with open(os.path.join(path_to_save, 'beta.pickle'), 'wb') as f:
        pickle.dump(fn, f)
    with open(os.path.join(path_to_save, 'beta_ln.pickle'), 'wb') as f:
        pickle.dump(fn_ln, f)


def find_threshold(config: DictConfig):
    dataset_parameters = config.dataset
    w = 5
    data_with_cp, data_without_cp, t_cp_idxs = gen.generate_data(**dataset_parameters)

    layer_norm = nn.LayerNorm(dataset_parameters["d"], elementwise_affine=False).float()
    data_with_cp_ln = (
        layer_norm(torch.from_numpy(data_with_cp).float()).detach().numpy()
    )
    data_without_cp_ln = (
        layer_norm(torch.from_numpy(data_without_cp).float()).detach().numpy()
    )

    delta_sic_with, likelihood_cp_with = cpd.t_dist_data_inference(data_with_cp, dataset_parameters)
    delta_sic_without, likelihood_cp_without = cpd.t_dist_data_inference(data_without_cp, dataset_parameters)

    delta_sic_with_ln, likelihood_cp_with_ln = cpd.t_dist_data_inference(data_with_cp_ln, dataset_parameters)
    delta_sic_without_ln, likelihood_cp_without_ln = cpd.t_dist_data_inference(data_without_cp_ln, dataset_parameters)

    min_delta = int(round(min(delta_sic_with.min(), delta_sic_without.min(), delta_sic_with_ln.min(), delta_sic_without_ln.min()), 0))
    max_delta = int(round(max(delta_sic_with[:, w:-w].max(), delta_sic_without[:, w:-w].max(), delta_sic_with_ln[:, w:-w].max(), delta_sic_without_ln[:, w:-w].max()), 0))

    fn_threshold = {}
    fp_threshold = {}

    fn_threshold_ln = {}
    fp_threshold_ln = {}


    for th in range(min_delta, max_delta, 1):
        detected_cp_idxs = np.where(delta_sic_with.min(1) > th, -1, likelihood_cp_with.argmin(1))
        detected_false_cp_idxs = np.where(delta_sic_without.min(1) > th, -1, likelihood_cp_without.argmin(1))
        fn, fp = cpd.calculate_error(detected_cp_idxs, detected_false_cp_idxs)
        fn_threshold[th] = fn
        fp_threshold[th] = fp

        detected_cp_idxs_ln = np.where(delta_sic_with_ln.min(1) > th, -1, likelihood_cp_with_ln.argmin(1))
        detected_false_cp_idxs_ln = np.where(delta_sic_without_ln.min(1) > th, -1, likelihood_cp_without_ln.argmin(1))
        fn_ln, fp_ln = cpd.calculate_error(detected_cp_idxs_ln, detected_false_cp_idxs_ln)
        fn_threshold_ln[th] = fn_ln
        fp_threshold_ln[th] = fp_ln

    # save metrics
    path_to_pickle = Path()
    path_to_pickle.mkdir(parents=True, exist_ok=True)

    save_errors(fn_threshold, fp_threshold, fn_threshold_ln, fp_threshold_ln, path_to_pickle)

    plt.plot(
        [value for value in fp_threshold.values()],
        [value for value in fn_threshold.values()],
        #"o",
        label="Without LN"
    )

    plt.plot(
        [value for value in fp_threshold_ln.values()],
        [value for value in fn_threshold_ln.values()],
        #"s",
        label="WITH LN"
    )

    plt.xlabel("False alarm rate")
    plt.ylabel("False negative rate");
    plt.legend()
    plt.savefig(os.path.join(Path()/'res.png'), dpi=300, bbox_tight=True)
    plt.close()

    return fn_threshold, fp_threshold, fn_threshold_ln, fp_threshold_ln

def save_errors(fn, fp, fn_ln, fp_ln, path_to_save):
    with open(os.path.join(path_to_save, 'alpha.pickle'), 'wb') as f:
        pickle.dump(fp, f)
    with open(os.path.join(path_to_save, 'alpha_ln.pickle'), 'wb') as f:
        pickle.dump(fp_ln, f)
    with open(os.path.join(path_to_save, 'beta.pickle'), 'wb') as f:
        pickle.dump(fn, f)
    with open(os.path.join(path_to_save, 'beta_ln.pickle'), 'wb') as f:
        pickle.dump(fn_ln, f)

