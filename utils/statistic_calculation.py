"""Calculate statistics for CPD."""
import numpy as np
from numba import jit
from math import sqrt

def calculate_z(sequence: np.array) -> np.array:
    """Calculate Z statistic for sequences.

    Args:
        sequence: data for statistic calculation

    Returns:
        np.array of statistics
    """
    seq_len = sequence.shape[1]
    z = []
    for i in range(1, seq_len):
        z.append(
            sqrt(i * (seq_len - i) / seq_len)
            * (
                np.sum(sequence[:, 1 : i + 1, :], 1) / i
                - np.sum(sequence[:, i + 1 :, :], 1) / (seq_len - i)
            )
        )
    z = np.array(z).transpose(1, 0, 2)
    return z


def calculate_l_lin(z: np.array, layer_norm: bool=False) -> np.array:
    """Calculate L-linear statistics.

    Args:
        z: data with statistic
        layer_norm: if True calculate L-linear statistic considering application of Layer normalizaion to the data, default=False

    Returns:
        np.array of statistics
    """
    d = z.shape[-1]
    if layer_norm:
        l_lin = (np.linalg.norm(z, axis=-1) ** 2 - (d - 1)) / sqrt(2 * (d - 1) ** 2 / d)
    else:
        l_lin = (np.linalg.norm(z, axis=-1) ** 2 - d) / sqrt(2 * d)
    return l_lin


def calculate_l_scan(z: np.array, p: int, layer_norm: bool=False) -> np.array:
    """Calculate L-scan statistics.

    Args:
        z: data with statistic
        p: number of dimensions with change points
        layer_norm: if True calculate L-linear statistic considering application of Layer normalizaion to the data, default=False

    Returns:
        np.array of statistics
    """
    z_squared = np.sort(z**2, axis=-1)[:, :, ::-1]  # descending order

    if layer_norm:
        l_scan = (z_squared[:, :, :p].sum(axis=-1) - p * (d - 1) / d) / (
            sqrt(2 * p) * (d - 1) / d
        )
    else:
        l_scan = (z_squared[:, :, :p].sum(axis=-1) - p) / sqrt(2 * p)
    return l_scan

# def t_dist_likelihood(data, theta, dataset_parameters):
#     """Calculate part of t-distribution likelihood necessary for delta SIC (depend on mean parameter).  

#     Args:
#         data: data for paramteres estimation
#         theta: optimal theta value for 
#         dataset_parameters: dict with dataset paramters.

#     Returns:
#         np.array with necessary part of likelihood
#     """
#     nu = dataset_parameters["nu"]
#     d = dataset_parameters["d"]

#     #delta_i =  (np.linalg.norm(data - theta[:, None, :], axis=2) ** 2)
#     delta_i = ((data - np.expand_dims(theta, 1)) ** 2).sum(2)
#     part_likelihood = (nu + d) * np.log(1 + (1 / nu) * delta_i).sum(1)
#     return part_likelihood    

@jit(nopython=True)
def t_dist_likelihood_numba(data, theta, nu, d):
    dataset_size = theta.shape[0]
    seq_len = data.shape[1]    
    likelihood = np.zeros((dataset_size, ))

    for j in range(dataset_size):
        likelihood_j = 0
        for k in range(seq_len):
            diff_sq = 0
            for l in range(d):
                diff_sq += (data[j, k, l] - theta[j, l])**2
            likelihood_j += (nu + d) * np.log(1 + (1 / nu) * diff_sq)
        likelihood[j] = likelihood_j
    return likelihood

def t_dist_likelihood(data, theta, dataset_parameters):
    """Calculate part of t-distribution likelihood necessary for delta SIC (depend on mean parameter). Optimized with numba package.

    Args:
        data: data for paramteres estimation
        theta: optimal theta value for 
        dataset_parameters: dict with dataset paramters.

    Returns:
        np.array with necessary part of likelihood
    """
    nu = dataset_parameters["nu"]
    d = dataset_parameters["d"]
    likelihood = t_dist_likelihood_numba(data, theta, nu, d)
    return likelihood


# def em_algorithm_old(data, dataset_parameters, em_steps=10):
#     """Run EM algorithm for optimization likelihood of t-distribution. We suppose known sigma and degrees freedom, sigma=identity(d). 

#     Args:
#         data: data for paramteres estimation
#         nu: degrees of freedom for "t-distribution"
#         d: dimension of each vector in sequence
#         theta_prev: init value of theta
#         em_steps: number of EM algorithm steps

#     Returns:
#         np.array with optimal mean value
#     """
#     dataset_size = dataset_parameters["dataset_size"]
#     nu = dataset_parameters["nu"]
#     d = dataset_parameters["d"]
#     np.random.seed(123)
#     theta_prev = np.random.rand(dataset_size, d)

#     for i in range(0, em_steps):
#         w = (nu + d) / (nu + (np.linalg.norm(data - np.expand_dims(theta_prev, 1), axis=2) ** 2))
#         #w = (nu + d) / (nu + ((data - np.expand_dims(theta_prev, 1)) ** 2).sum(2))
#         theta_next = (data * np.expand_dims(w, -1)).sum(1) / np.expand_dims(w, -1).sum(1)
#         theta_prev = theta_next    
#     return theta_prev


@jit(nopython=True)
def em_algorithm_numba(data, nu, d, theta_prev, em_steps=10):
    """Run EM algorithm for optimization likelihood of t-distribution. Optimized version based on numba package. 
    We suppose known sigma and degrees freedom, sigma=identity(d).

    Args:
        data: data for paramteres estimation
        nu: degrees of freedom for "t-distribution"
        d: dimension of each vector in sequence
        theta_prev: init value of theta
        em_steps: number of EM algorithm steps

    Returns:
        np.array with optimal mean value
    """
    dataset_size = theta_prev.shape[0]
    seq_len = data.shape[1]

    for _ in range(0, em_steps):
        theta_next = np.zeros((dataset_size, d))
        for j in range(0, dataset_size):
            w_sum = 1e-18
            for k in range(0, seq_len):
                diff_sq = 0
                for l in range(0, d):
                    diff_sq += (data[j, k, l] - theta_prev[j, l])**2
                w = (nu + d) / (nu + diff_sq)
                w_sum += w
                for l in range(0, d):
                    theta_next[j, l] += data[j, k, l] * w
            theta_next[j] = theta_next[j] / w_sum
        theta_prev = theta_next
    return theta_prev

def em_algorithm(data, dataset_parameters, em_steps=10, **kwargs):
    """Run EM algorithm for optimization likelihood of t-distribution. We suppose known sigma and degrees freedom, sigma=identity(d).

    Args:
        data: data for paramteres estimation
        dataset_parameters: dict with dataset paramters
        em_steps: number of EM algorithm steps
    Returns:
        np.array with optimal mean value
    """
    dataset_size = dataset_parameters["dataset_size"]
    nu = dataset_parameters["nu"]
    d = dataset_parameters["d"]
    np.random.seed(123)
    theta_prev = np.random.rand(dataset_size, d)
    theta_final = em_algorithm_numba(data, nu, d, theta_prev, em_steps)
    return theta_final
