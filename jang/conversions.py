"""Utility functions to perform conversions."""

import abc
import astropy.time
import datetime
import numpy as np
import scipy.integrate
from typing import List

Mpc_to_cm = 3.0856776e24
erg_to_GeV = 624.15
solarmass_to_erg = 1.787e54
second_to_day = 1/86400


class JetModelBase(metaclass=abc.ABCMeta):
    """Abstract model for neutrino emission jetting."""

    def __init__(self, jet_opening: float):
        self.jet_opening = jet_opening

    @abc.abstractmethod
    def etot_to_eiso(self, viewing_angle: float):
        """Conversion to total energy to the equivalent isotropic energy."""
        pass

    @property
    def str_filename(self):
        return self.__repr__().lower().replace(",", "_").replace(" ", "")


class JetIsotropic(JetModelBase):
    """Isotropic emission of neutrinos."""

    def __init__(self):
        super().__init__(np.inf)

    def etot_to_eiso(self, viewing_angle: float) -> float:
        return 1

    def __repr__(self):
        return "Isotropic"


class JetVonMises(JetModelBase):
    """Emission in a Gaussian jet."""

    def __init__(self, jet_opening: float, with_counter: bool = False):
        super().__init__(jet_opening)
        self.with_counter = with_counter
        self.kappa = np.float128(1 / (self.jet_opening ** 2))

    def etot_to_eiso(self, viewing_angle: float) -> float:
        if np.isinf(self.jet_opening):
            return 1
        if self.with_counter:
            return (
                self.kappa
                * np.cosh(self.kappa * np.cos(viewing_angle))
                / np.sinh(self.kappa)
            )
        return (
            self.kappa
            * np.exp(self.kappa * np.cos(viewing_angle))
            / np.sinh(self.kappa)
        )

    def __repr__(self):
        return "VonMises,%.1f deg%s" % (
            self.jet_opening,
            ",w/counter" if self.with_counter else "",
        )


class JetRectangular(JetModelBase):
    """Emission in a rectangular jet (constant inside a cone, zero elsewhere)."""

    def __init__(self, jet_opening: float, with_counter: bool = False):
        super().__init__(jet_opening)
        self.with_counter = with_counter

    def etot_to_eiso(self, viewing_angle: float) -> float:
        if not 0 < self.jet_opening < np.pi:
            return 1
        if self.with_counter:
            if (
                viewing_angle <= self.jet_opening
                or viewing_angle >= np.pi - self.jet_opening
            ):
                return 1 / (1 - np.cos(self.jet_opening))
            return 0
        if viewing_angle <= self.jet_opening:
            return 2 / (1 - np.cos(self.jet_opening))
        return 0

    def __repr__(self):
        return "Constant,%.1f deg%s)" % (
            self.jet_opening,
            ",w/counter" if self.with_counter else "",
        )


def list_jet_models() -> List[JetModelBase]:
    """List all available jet models, with a scanning in opening angles."""
    full_list = []
    full_list.append(JetIsotropic())
    for opening in range(5, 60 + 1, 5):
        for with_counter in (False, True):
            full_list.append(JetVonMises(opening, with_counter=with_counter))
            full_list.append(JetRectangular(opening, with_counter=with_counter))
    return full_list


def phi_to_eiso(
    energy_range: tuple, spectrum: str = "x**-2", distance: float = 1
) -> float:
    """Convert from flux normalization to total isotropic energy for a given spectrum and distance."""
    distance_cm = distance * Mpc_to_cm
    f = eval("lambda y: %s * (np.exp(y))**2" % spectrum.replace("x", "np.exp(y)"))
    integration = scipy.integrate.quad(f, *np.log(energy_range), limit=100)[0]
    return integration * (4 * np.pi * distance_cm ** 2) / erg_to_GeV


def eiso_to_phi(
    energy_range: tuple, spectrum: str = "x**-2", distance: float = 1
) -> float:
    """Convert from total isotropic energy to flux normalization for a given spectrum and distance."""
    return 1 / phi_to_eiso(energy_range, spectrum, distance)


def etot_to_eiso(viewing_angle: float, model: JetModelBase) -> float:
    """Convert from total energy to the equivalent isotropic energy for a given jet model and viewing angle."""
    return model.etot_to_eiso(viewing_angle)


def fnu_to_etot(radiated_energy_gw: float) -> float:
    """Convert from fraction of radiated energy to total energy."""
    return radiated_energy_gw * solarmass_to_erg


def utc_to_jd(dtime: datetime.datetime) -> float:
    """Convert from UTC time (datetime format) to julian date."""
    t = astropy.time.Time(dtime, format="datetime")
    return t.jd


def jd_to_mjd(jd: float) -> float:
    """Convert Julian Date to Modified Julian Date."""
    return jd - 2400000.5
