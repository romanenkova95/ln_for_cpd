"""Utils for different approaches for CPD threshold calculation."""
import numpy as np
from scipy.stats import chi2, gamma
from math import sqrt, log
from functools import reduce
import operator as op


def calculate_lin_threshold(
    alpha, d, concentration_estimate=True, layer_norm=False
) -> float:
    """Calculate threshold for L-linear statistics.

    Args:
        alpha: confident level
        d: data dimensionality
        concentration_estimate: if True use approximation for threshold calculation, default=True
        layer_norm: if True calculate threshold considering application of Layer normalizaion to the data, default=False

    Returns:
        np.array of statistics
    """
    if concentration_estimate:
        H = sqrt(2 * log(1 / alpha)) + sqrt(1 / (2 * d)) * log(1 / alpha)
    else:
        if layer_norm:
            H = (gamma.ppf(1 - alpha, a=d / 2, scale=2 * (d - 1) / d) - (d - 1)) / sqrt(
                2 * (d - 1) ** 2 / d
            )
        else:
            H = (chi2.ppf(1 - alpha, d) - d) / sqrt(2 * d)
    return H


def calculate_scan_threshold(
    alpha, d, p, concentration_estimate=True, layer_norm=False
) -> float:
    """Calculate threshold for L-scan statistics.

    Args:
        alpha: confident level
        d: data dimensionality
        p: number of dimensions with change points
        concentration_estimate: if True use approximation for threshold calculation, default=True
        layer_norm: if True calculate threshold considering application of Layer normalizaion to the data, default=False

    Returns:
        np.array of statistics
    """

    def ncr(n, r):
        r = min(r, n - r)
        numer = reduce(op.mul, range(n, n - r, -1), 1)
        denom = reduce(op.mul, range(1, r + 1), 1)
        return numer // denom

    ncr_value = ncr(d, p) * (1 / alpha)

    if concentration_estimate:
        T = sqrt(2 / p) * log(ncr_value) + sqrt(2 * log(ncr_value))
    else:
        if layer_norm:
            T = (
                gamma.ppf(1 - ncr(d, p) * alpha, a=p / 2, scale=2 * (d - 1) / d)
                - p * (d - 1) / d
            ) / (sqrt(2 * p) * (d - 1) / d)
        else:
            T = (chi2.ppf(1 - alpha / ncr(d, p), p) - p) / sqrt(2 * p)
    return T


def calculate_threshold(
    l_statistic,
    alpha,
    d,
    p,
    threshold_type="data_based",
    concentration_estimate=True,
    layer_norm=False,
) -> float:
    """Calculate L-statistic and threshold according to the paper's equations or based on the data.

    Args:
        l_statistic: np.array with statistics
        alpha: confident level
        d: data dimensionality
        p: number of dimensions with change points
        threshold_type: type of statistic. Possible variants:
            - "linear" - followed paper's formulas
            - "scan" - followed paper's formulas
            - "data-based" - set threshold as alpha-quantile of the data
        concentration_estimate: if True use approximation for threshold calculation, default=True
        layer_norm: if True calculate threshold considering application of Layer normalizaion to the data, default=False

    Returns:
        threshold value
    """
    if threshold_type == "data_based":
        threshold = np.quantile(l_statistic.flatten(), 1 - alpha)
    elif threshold_type == "linear":
        threshold = calculate_lin_threshold(
            alpha, d, concentration_estimate=True, layer_norm=layer_norm
        )
    elif threshold_type == "scan":
        threshold = calculate_scan_threshold(
            alpha / (2 * d),
            d,
            p,
            concentration_estimate=True,
            layer_norm=layer_norm,
        )
    return threshold

from math import log, exp
from scipy.special import gamma

def c_alpha(alpha, n, p):
    def d_p(x, p):
        return 2 * log(x) + (p / 2) * log(log(x)) - log(gamma(p / 2))
    log_n = log(n)
    d_p_log_n = d_p(log_n, p)
    c_alpha = 1 / (2 * log(log_n)) * ((d_p_log_n - log(log((1 - alpha + exp(-2 * exp(d_p_log_n))) ** -0.5))) ** 2) - p * log_n
    return c_alpha