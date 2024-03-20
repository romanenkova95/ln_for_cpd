"""Utils for CPD."""
import utils.threshold_calculation as th
import utils.statistic_calculation as stat
from utils.data_generation import generate_data
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2, f

def normal_data_inference(data, p: int, ln: bool = False, scan: bool = False):
    """Calculate CPs statistics for normal distribution assumption.

    Args:
        data: np.array with considered sequences
        p: number of dimensions with a change
        ln: if True run procedure considering application of Layer normalizaion to the data, default=False
        scan: if True use "scan"-statistic instead of "linear", default=False
    Returns:
        np.array with corresponded statistic for CPD.

    """
    z = stat.calculate_z(data)
    d = data.shape[-1]
    
    if scan:
        l_statistic = stat.calculate_l_scan(z, p, layer_norm=ln)
    else:
        l_statistic = stat.calculate_l_lin(z, layer_norm=ln)
    return l_statistic


def detect_cps_normal(data: np.array, dataset_parameters: dict, alpha: float = 0.05, ln: bool = False, scan: bool = False, data_based: bool = False):
    """Detect CPs based on statistics in assumption of data normal distribution.

    Args:
        data: np.array with considered sequences
        dataset_parameters: dict with dataset paramters
        alpha: confident level
        ln: if True run procedure considering application of Layer normalizaion to the datWa, default=False
        scan: if True use "scan"-statistic instead of "linear", default=False
        data_based: if True calculate threshold as alpha-quantile of the obtained statistic instead of formulas, default=False
        data_type: control inference for different data distribution. Possible options:
            - "normal" - each vector follow standard multivariate normal distribution
            - "t-distribution" - each vector follow standard multivariate t-distribution distribution
    Returns:
        np.array with detected CP indexes
    """
    p = dataset_parameters["p"]
    d = dataset_parameters["d"]

    l_statistic = normal_data_inference(data=data, p=p, ln=ln, scan=scan)

    if data_based:
        threshold_type = "data_based"
    elif scan:
        threshold_type = "scan"
    else:
        threshold_type = "linear"
    threshold = th.calculate_threshold(l_statistic, alpha, d, p, threshold_type, layer_norm=ln)

    detected_cp_idxs = []

    for seq in l_statistic:
        tmp_cps = np.where(seq > threshold)[0]
        if len(tmp_cps) == 0:
            cp = -1
        else:
            cp = (
                tmp_cps[np.argmax(seq[tmp_cps])] + 1
            )  # because we generate n + 1 points to avoid zero-indexing
            cp += 1
        detected_cp_idxs.append(cp)
    return np.array(detected_cp_idxs), l_statistic

def t_dist_data_inference(data, dataset_parameters, w=5):
    """Calculate CPs statistics for t-distribution assumption.

    Args:
        data: np.array with considered sequences
        dataset_parameters: dict with dataset paramters
        alpha: confidence level for threshold calculation
        w: paramter to avoid edge effects (near zero value of likelihood calculated on 1-2 points), default=3
    Returns:
        np.array with corresponded statistic for CPD.

    """

    theta = stat.em_algorithm(data, dataset_parameters)
    n = dataset_parameters["seq_len"]
    likelihood_no_cp = stat.t_dist_likelihood(data, theta, dataset_parameters)
    delta_sic = []
    likelihood_cp = []

    for k in range(0, n):
        if (k < w) or (n - k <= w):
            delta_sic_k = 999999 * np.ones((dataset_parameters["dataset_size"], ))
            likelihood_cp_k = 999999 * np.ones((dataset_parameters["dataset_size"], ))
        else:
            theta_0 = stat.em_algorithm(data[:, :(k + 1), :], dataset_parameters)
            theta_1 = stat.em_algorithm(data[:, (k + 1):, :], dataset_parameters)
            likelihood_cp_k = stat.t_dist_likelihood(data[:, :(k + 1), :], theta_0, dataset_parameters) + stat.t_dist_likelihood(data[:, (k + 1):, :], theta_1, dataset_parameters)
            delta_sic_k = likelihood_cp_k - likelihood_no_cp #+ dataset_parameters["d"] * np.log(n)

        delta_sic.append(delta_sic_k)
        likelihood_cp.append(likelihood_cp_k)

    delta_sic = np.array(delta_sic).transpose()
    likelihood_cp = np.array(likelihood_cp).transpose()
    return delta_sic, likelihood_cp

def detect_cps_t_dist(data_with_cp, data_without_cp, dataset_parameters, alpha=0.95, w=5):
    """Detect CPs based on SIC criteria in T-distribution assumption.

    Args:
        data_with_cp: np.array with sequences with CP
        data_without_cp: np.array with sequences without CP
        dataset_parameters: dict with dataset paramters
        alpha: confidence level for threshold calculation
        w: paramter to avoid edge effects (near zero value of likelihood calculated on 1-2 points), default=3
    Returns:
        np.array with fn and fp errors

    """
    delta_sic_with, likelihood_cp_with = t_dist_data_inference(data_with_cp, dataset_parameters, w)
    delta_sic_without, likelihood_cp_without = t_dist_data_inference(data_without_cp, dataset_parameters, w)

    delta_sic_with = - delta_sic_with #switch to avoid working with negative value
    delta_sic_without = - delta_sic_without

    threshold = np.quantile(delta_sic_without[:, 5:-5].flatten(), 1 - alpha)

    #predicted_cp = np.where(delta_sic.min(1) > threshold , -1, likelihood_cp.argmin(1))
    detected_cp = np.where(delta_sic_with.max(1) > threshold , likelihood_cp_with.argmin(1), -1)
    detected_false_cp = np.where(delta_sic_without.max(1) > threshold , likelihood_cp_without.argmin(1), -1)
    fn, fp = calculate_error(detected_cp, detected_false_cp)

    return fn, fp  

def calculate_error(detected_cp: np.array, detected_false_cp: np.array):
    """Calculate FN (power) and FP metrics for predictions.

    Args:
        detected_cp: np.array with predicted CP for data WITH CP
        detected_false_cp: np.array with predicted CP for data WITHOUT CP
    Returns:
        FN, FP

    """
    assert detected_cp.shape == detected_false_cp.shape

    false_negative = sum(detected_cp == -1) / len(detected_cp)
    false_positive = sum(detected_false_cp != -1) / len(detected_false_cp)
    return false_negative, false_positive

def cpd_with_ln_compare(
    dataset_parameters: dict, cp_parameters:dict, data_type="normal"
):
    """Compare CP detection with using Layer Norm and without. In t-distribution assumption. 

    Args:
        dataset_parameters: dict with parameters for data generation (dataset size, dimensions, seq_len, etc.)
    Returns:

    """
    data_with_cp, data_without_cp, cp_idxs = generate_data(**dataset_parameters)

    fn, fp = detect_cps_t_dist(data_with_cp, data_without_cp, dataset_parameters, alpha=cp_parameters["alpha"], w=5)

    ### with Layer Norm
    cp_parameters["ln"] = True
    layer_norm = nn.LayerNorm(dataset_parameters["d"], elementwise_affine=False).float()
    data_with_cp_ln = (
        layer_norm(torch.from_numpy(data_with_cp).float()).detach().numpy()
    )
    data_without_cp_ln = (
        layer_norm(torch.from_numpy(data_without_cp).float()).detach().numpy()
    )


    fn_ln, fp_ln = detect_cps_t_dist(data_with_cp_ln, data_without_cp_ln, dataset_parameters, alpha=cp_parameters["alpha"], w=5)
    return (fn, fp), (fn_ln, fp_ln)