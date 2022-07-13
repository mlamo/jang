import abc
from typing import Callable

import numpy as np
from scipy.stats import norm


def angular_distance(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray):
    s = np.cos(ra1 - ra2) * np.cos(dec1) * np.cos(dec2) + np.sin(dec1) * np.sin(dec2)
    return np.arccos(np.clip(s, -1, 1))


class PDF:

    @abc.abstractmethod
    def __call__(self, evt):
        pass


class EnergySignal(PDF):
    """The standard energy signal PDF is a function f(ra,dec,E)."""

    def __init__(self, spectrum: str, func: Callable = None):
        self.func = func
        self.spectrum = spectrum

    def __call__(self, evt):
        return self.func(evt.ra, evt.dec, evt.energy)


class AngularSignal(PDF):
    """The standard angular signal PDF is a function f(ra,dec,ra[src],dec[src],E)."""

    def __init__(self, func: Callable = None):
        self.func = func

    def __call__(self, evt, ra_src: float, dec_src: float):
        dpsi = angular_distance(evt.ra, evt.dec, ra_src, dec_src)
        return self.func(dpsi, evt.energy)


class VonMisesSignal(AngularSignal):
    """A common angular signal PDF is Von Mises distribution f = VM(dpsi, sigma)."""

    def __init__(self):
        pass

    def __call__(self, evt, ra_src: float, dec_src: float):
        dpsi = angular_distance(evt.ra, evt.dec, ra_src, dec_src)
        if evt.sigma > np.radians(7):
            kappa = 1. / evt.sigma**2
            return kappa * np.exp(kappa * np.cos(dpsi)) / (4*np.pi * np.sinh(kappa))
        else:
            return 0.5 / np.pi / evt.sigma**2 * np.exp(-0.5 * (dpsi/evt.sigma)**2)


class EnergyBackground(PDF):
    """The standard energy background PDF is a function f(ra,dec,E)."""

    def __init__(self, func: Callable = None):
        self.func = func

    def __call__(self, evt):
        return self.func(evt.ra, evt.dec, evt.energy)


class AngularBackground(PDF):
    """The standard angular background PDF is a function f(ra,dec,E)."""

    def __init__(self, func: Callable = None):
        self.func = func

    def __call__(self, evt):
        return self.func(evt.ra, evt.dec, evt.energy)


class TimeSignal(PDF):
    """The standard time signal PDF is a function f(deltaT)."""

    def __init__(self, func: Callable = None):
        self.func = func

    def __call__(self, evt):
        return self.func(evt.dt)


class TimeBoxSignal(PDF):
    """A common time signal PDF is 1/dt for t0 <= t < t0+dt and 0 otherwise."""

    def __init__(self, t0: float, dt: float):
        self.t0 = t0
        self.dt = dt

    def __call__(self, evt, t0: float = None, dt: float = None):
        t0 = self.t0 if t0 is None else t0
        dt = self.dt if dt is None else dt
        return 1/dt if t0 <= evt.dt < t0+dt else 0


class TimeGausSignal(PDF):
    """A commin time signal PDF is a normal distribution centered on t0."""

    def __init__(self, t0: float, sigma: float):
        self.t0 = t0
        self.sigma = sigma

    def __call__(self, evt, t0: float = None, sigma: float = None):
        t0 = self.t0 if t0 is None else t0
        sigma = self.sigma if sigma is None else sigma
        return norm.pdf(evt.dt, loc=t0, scale=sigma)


class TimeBackground(PDF):
    """The standard time background PDF is an uniform function f(deltaT) = 1/timewindow."""

    def __init__(self, timewindow: float):
        self.timewindow = timewindow

    def __call__(self, evt):
        return 1/self.timewindow
