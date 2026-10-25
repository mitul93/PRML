from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def generate_regression_data(
    f: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    sample_size: int,
    std: float,
    domain: tuple[float, float] = (0, 1),
    uniform: bool = False,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Generates a toy dataset given a function, a domain and a sample size. Then adds
    Gaussian noise to the samples having zero mean and the given standard deviation.

    :param f: a function
    :param sample_size: the size of the sample
    :param std: the standard deviation of the Gaussian noise or the low/upper bound of
        the uniform noise
    :param domain: the domain range
    :param uniform: adds uniform noise instead of Gaussian
    :return: a tuple of the input, target arrays
    """
    x = np.linspace(domain[0], domain[1], sample_size)
    t = f(x) + (
        np.random.uniform(-std, std, size=x.shape)
        if uniform
        else np.random.normal(scale=std, size=x.shape)
    )
    return x, t
