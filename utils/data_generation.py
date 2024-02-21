"""Utils for data generation."""
from typing import Optional, Tuple, List
import random
import numpy as np
from scipy.stats import multivariate_t
from copy import deepcopy

def sample_data(dataset_size, seq_len, d, distribution="normal", nu=10, seed=123):
    """Generate a dataset of sequence, which follows the selected distribution type.

    Args:
        dataset_size: number of generated sequence
        seq_len: length of each generated sequence
        d: dimension of each vector in sequence
        distribution: type of distribution for generated data. Possible options:
            - "normal" - each vector follow standard multivariate normal distribution
            - "laplace" - each component (!) of vector follow standard laplace distribution
            - "t-distribution" - each vector follow standard multivariate t-distribution
        nu: degrees of freedom if "t-distribution", default = 10
        seed: fix seed for reproducibility, default = 123

    Returns:
        np.array of data with sise (dataset_size, seq_len, d)
    """
    np.random.seed(seed)
    random.seed(seed)

    if distribution == "normal":
        data = np.random.multivariate_normal(
            mean=np.zeros(d), cov=np.identity(d), size=(dataset_size, seq_len)
        )
    elif distribution == "laplace":
        data = np.random.laplace(loc=0.0, scale=1.0, size=(dataset_size, seq_len, d))
    elif distribution == "t-distribution":
        data = (
            multivariate_t(np.zeros(d), np.identity(d), df=nu, seed=seed)
            .rvs(dataset_size * seq_len)
            .reshape(dataset_size, seq_len, d)
        )
    else:
        raise ValueError(f"Not implemented type {distribution}.")
    return data


def generate_data(
    dataset_size: int,
    seq_len: int,
    d: int,
    p: int,
    distribution: str = "normal",
    k: int = 0.6,
    cp: Optional[int] = None,
    nu=10,
    seed=123,
) -> Tuple[np.array, List[int]]:
    """Generate a dataset with and without change points.

    Args:
        dataset_size: number of generated sequence
        seq_len: length of each generated sequence
        d: dimension of vector in sequence
        p: number of components with a change (if there is)
        distribution: type of distribution for generated data. Possible options:
            - "normal" - each vector follow standard multivariate normal distribution
            - "laplace" - each component (!) of vector follow standard laplace distribution
            - "t-distribution" - each vector follow standard multivariate t-distribution
        k: coefficient for delta-theta norm controlling, default = 0.6
        cp: fix index of a change point, optional, default = None
        nu: degrees of freedom if "t-distribution", default = 10
        seed: fix seed for reproducibility, default = 123

    Returns:
        Tuple of
            - array with generated data WITH change points
            - array with generated data WITHOUT change points
            - list with change point indexes for first mentioned dataset
    """
    data_cp = sample_data(dataset_size, seq_len, d, distribution, nu, seed)
    data_without_cp = deepcopy(data_cp)


    # define delta_theta - shifts in p dimentsions
    delta_theta = k * np.ones((dataset_size, p))
    if p <= d:
        delta_theta = np.hstack((np.zeros((dataset_size, d - p)), delta_theta))

    # shuffle delta along -1 axis (change occurs not in last dimensions only)
    idx = np.random.rand(*delta_theta.shape).argsort(axis=-1)
    delta_theta = np.take_along_axis(delta_theta, idx, axis=-1)

    # add change points
    cp_idxs = []
    if cp is None:
        cp_idxs = np.random.randint(10, seq_len - 1, dataset_size)
    else:
        cp_idxs = [cp] * dataset_size

    for i in range(len(data_cp)):
        data_cp[i][cp_idxs[i] :] += delta_theta[i]
    return data_cp, data_without_cp, cp_idxs