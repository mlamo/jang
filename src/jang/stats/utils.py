from typing import Callable, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d


class PosteriorVariable:

    def __init__(self, name: str, range: tuple, nevals: int = 101, log: bool = False):
        self.name = name
        self.range = range
        self.nevals = nevals
        self.log = log
        self.prior = lambda x: 1

    @property
    def array(self):
        if self.log:
            return np.logspace(*self.range, self.nevals)
        return np.linspace(*self.range, self.nevals)

    def set_prior(self, prior: Callable):
        self.prior = np.vectorize(prior)

    def prior(self, x: Union[float, np.ndarray]):
        return self.prior(x)


def compute_upperlimit_from_x_y(x_arr: np.ndarray, y_arr: np.ndarray, CL: float = 0.9) -> float:
    """Compute the upper limit at a confidence level CL for a given posterior y_arr=P(x_arr)."""
    if np.all(y_arr == 0):
        return np.inf
    int_arr = y_arr[:-1] * (x_arr[1:] - x_arr[:-1])
    cum_arr = np.cumsum(int_arr)
    cum_arr = 1 / cum_arr[-1] * cum_arr
    f = interp1d(cum_arr, x_arr[:-1])
    limit = float(f(CL))
    # limit dangerously closed to lkl upper bound
    if limit > 0.9 * CL * x_arr[-1]:
        return np.inf
    return limit


def normalize(x_arr: np.ndarray, y_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize the posterior y_arr=P(x_arr)."""
    if x_arr.shape != y_arr.shape:
        raise RuntimeError(f"x_arr and y_arr have different dimensions: {x_arr.shape}, {y_arr.shape}")
    if np.all(y_arr == 0):
        return x_arr, y_arr
    integral = np.sum(y_arr[:-1] * (x_arr[1:] - x_arr[:-1]))
    return x_arr, 1 / integral * y_arr
