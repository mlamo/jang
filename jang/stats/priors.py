import numpy as np


def flat(var: np.ndarray):
    return np.ones_like(var)


def invsqrt(var: np.ndarray):
    return 1/np.sqrt(var)
    

def jeffrey_poisson(var: np.ndarray, bkg: np.ndarray, conv: np.ndarray):
    nsamples = len(bkg)
    tmp = [
        conv[i] ** 2 / (conv[i] * var + bkg[i]) if conv[i] > 0 else 0
        for i in range(nsamples)
    ]
    return np.sqrt(np.sum(tmp, axis=0))
    

def signal_parameter(var: np.ndarray, bkg: np.ndarray, conv: np.ndarray, prior_type: str):

    if prior_type == "flat":
        return flat(var)
    elif prior_type == "jeffrey":
        return jeffrey_poisson(var, bkg, conv)
    elif prior_type == "invsqrt":
        return invsqrt(var)
    else:
        raise RuntimeError(f"Unknown prior type {prior_type}")
