"""Defining the neutrino detectors and all related objects (samples, acceptances...)."""

import abc
import itertools
import logging
import os
import warnings
from collections.abc import Iterable
from typing import List, Optional, Tuple, Union

import astropy
import astropy.coordinates
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import yaml
from astropy.units import Quantity, Unit, rad
from scipy.linalg import block_diag
from scipy.stats import gamma, multivariate_normal, truncnorm

import jang.stats.pdfs as pdf

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=scipy.integrate.IntegrationWarning)


def infer_uncertainties(input_array: Union[float, np.ndarray], nsamples: int, correlation: Optional[float] = None) -> np.ndarray:
    """Infer uncertainties based on an input array that could be:
        - 0-D (same error for each sample)
        - 1-D (one error per sample)
        - 2-D (correlation matrix)
    """
    if input_array is None:
        return None
    input_array = np.array(input_array)
    correlation_matrix = (correlation if correlation is not None else 0) * np.ones((nsamples, nsamples))
    np.fill_diagonal(correlation_matrix, 1)
    # if uncertainty is a scalar (error for all samples)
    if input_array.ndim == 0:
        return input_array * correlation_matrix * input_array
    # if uncertainty is a vector (error for each sample)
    if input_array.shape == (nsamples,):
        return np.array([[input_array[i] * correlation_matrix[i, j] * input_array[j] for i in range(nsamples)] for j in range(nsamples)])
    # if uncertainty is a covariance matrix
    if input_array.shape == (nsamples, nsamples):
        return input_array
    raise RuntimeError("The size of uncertainty_acceptance does not match with the number of samples")


class Acceptance:
    """General class to handle detector acceptance for a given sample, spectrum and neutrino flavour.
    The acceptance is stored as a numpy array representing a HealPix map."""

    def __init__(self, rinput: Union[np.ndarray, float, int, str]):
        if isinstance(rinput, np.ndarray):  # pass the numpy array directly
            self.map = rinput
            self.nside = hp.npix2nside(len(self.map))
        elif isinstance(rinput, float) or isinstance(rinput, int):  # constant value for all sky
            self.map = rinput
            self.nside = 0
        elif isinstance(rinput, str) and rinput.endswith(".npy"):  # input npy file
            self.from_npy(rinput)
        else:  # default value = 0
            logging.getLogger("jang").warning("Acceptance is set to 0!")
            self.map = 0
            self.nside = 0

    def __call__(self, ra: float, dec: float):
        ipix = hp.ang2pix(self.nside, np.pi / 2 - dec, ra)
        return self.evaluate(ipix)

    def is_zero(self):
        if self.nside == 0:
            return True
        return np.all(self.map == 0)

    def from_npy(self, file: str):  # pragma: no cover
        assert os.path.isfile(file)
        self.map = np.load(file, allow_pickle=True)
        self.nside = hp.npix2nside(len(self.map))

    def change_resolution(self, nside):
        if self.nside != nside:
            if self.is_zero():
                self.map = np.zeros(hp.nside2npix(nside))
            else:
                self.map = hp.pixelfunc.ud_grade(self.map, nside)
                self.nside = nside

    def evaluate(self, ipix: int, nside: Optional[int] = None):
        if self.is_zero():
            return 0
        ipix_acc = ipix
        if nside is not None and nside != self.nside:
            ipix_acc = hp.ang2pix(self.nside, *hp.pix2ang(nside, ipix))
        return self.map[ipix_acc]

    def draw(self, outfile: str, log: bool = False):  # pragma: no cover
        if self.is_zero():
            return
        plt.close("all")
        hp.mollview(self.map, min=None if log else 0, rot=180, cmap="Blues", title="", unit=r"Acceptance [cm$^{2}$/GeV]", norm="log" if log else None)
        hp.graticule()
        plt.savefig(outfile, dpi=300)


class Background(abc.ABC):
    @abc.abstractmethod
    def prepare_toys(self, ntoys: int):
        pass

    @property
    @abc.abstractmethod
    def nominal(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass


class BackgroundFixed(Background):
    def __init__(self, b0: float):
        self.b0 = b0

    def prepare_toys(self, ntoys: int):
        return self.b0 * np.ones(ntoys)

    @property
    def nominal(self):
        return self.b0

    def __repr__(self):
        return f"{self.b0:.2e}"


class BackgroundGaussian(Background):
    def __init__(self, b0: float, error_b: float):
        self.b0, self.error_b = b0, error_b

    def prepare_toys(self, ntoys: int):
        return truncnorm.rvs(-self.b0 / self.error_b, np.inf, loc=self.b0, scale=self.error_b, size=ntoys)

    @property
    def nominal(self):
        return self.b0

    def __repr__(self):
        return f"{self.b0:.2e} +/- {self.error_b:.2e}"


class BackgroundPoisson(Background):
    def __init__(self, Noff: int, alpha_offon: int):
        self.Noff, self.alpha_offon = Noff, alpha_offon

    def prepare_toys(self, ntoys: int):
        return gamma.rvs(self.Noff + 1, scale=1 / self.alpha_offon, size=ntoys)

    @property
    def nominal(self):
        return self.Noff / self.alpha_offon

    def __repr__(self):
        return f"{self.nominal:.2e} = {self.Noff:d}/{self.alpha_offon:d}"


class NuEvent:
    """Class to handle a single neutrino candidate."""
    
    def __init__(
        self,
        dt: float = np.nan,
        ra: float = np.nan,
        dec: float = np.nan,
        energy: float = np.nan,
        sigma: float = np.nan,
        altitude: float = np.nan,
        azimuth: float = np.nan,
    ):
        """Event is defined by:
        - dt = t(neutrino)-t(GW) [in seconds]
        - ra/dec = reconstructed equatorial directions [in radians]
        - energy = reconstructed energy [in GeV]
        - sigma = uncertainty on reconstructed direction [in radians]
        - altitude/azimuth = reconstructed local directions [in radians]
        """
        self.dt = dt
        self.ra = ra
        self.dec = dec
        self.energy = energy
        self.sigma = sigma
        self.altitude = altitude
        self.azimuth = azimuth

    def __repr__(self):
        r = f"Event(deltaT={self.dt:.0f} s, ra/dec={np.rad2deg(self.ra):.1f}/{np.rad2deg(self.dec):.1f} deg, energy={self.energy:.2g} GeV, "
        r += f"sigma={np.rad2deg(self.sigma):.2f} deg, alt/azi={np.rad2deg(self.altitude):.1f}/{np.rad2deg(self.azimuth):.1f} deg)"
        return r

    @property
    def sindec(self):
        return np.sin(self.dec)

    @property
    def sinalt(self):
        return np.sin(self.altitude)

    @property
    def log10energy(self):
        return np.log10(self.energy)


class NuSample:
    """Class to handle a given neutrino sample characterised by its name, observed events, expected background and PDFs."""

    def __init__(self, name: str = None, shortname: str = None):
        self.acceptances = {}
        self.name = name
        self.shortname = shortname if shortname is not None else name
        self.energy_range = (None, None)
        self.nobserved = np.nan
        self.background = None
        self.events = None
        self.pdfs = {
            "signal": {"ang": None, "ene": None, "time": None},
            "background": {"ang": None, "ene": None, "time": None},
        }

    def set_energy_range(self, emin: float, emax: float):
        self.energy_range = (emin, emax)

    def set_acceptance(self, acceptance: Union[np.ndarray, float], spectrum: str, nside: Optional[int] = None):
        acc = Acceptance(acceptance)
        if nside is not None:
            acc.change_resolution(nside)
        assert isinstance(spectrum, str)
        self.acceptances[spectrum] = acc

    def set_observations(self, nobserved: int, bkg: Background):
        self.nobserved = nobserved
        self.background = bkg

    def set_events(self, events: List[NuEvent]):
        self.events = events

    def set_pdfs(
        self,
        sig_ang: pdf.AngularSignal = None,
        sig_ene: pdf.EnergySignal = None,
        sig_time: pdf.TimeSignal = None,
        bkg_ang: pdf.AngularBackground = None,
        bkg_ene: pdf.EnergyBackground = None,
        bkg_time: pdf.TimeBackground = None,
    ):
        if (sig_ang is None) ^ (bkg_ang is None):
            raise RuntimeError("One of the angular PDFs is missing!")
        if (sig_ene is None) ^ (bkg_ene is None):
            raise RuntimeError("One of the energy PDFs is missing!")
        if (sig_time is None) ^ (bkg_time is None):
            raise RuntimeError("One of the time PDFs is missing!")
        self.pdfs["signal"]["ang"] = sig_ang
        self.pdfs["signal"]["ene"] = sig_ene
        self.pdfs["signal"]["time"] = sig_time
        self.pdfs["background"]["ang"] = bkg_ang
        self.pdfs["background"]["ene"] = bkg_ene
        self.pdfs["background"]["time"] = bkg_time

    @property
    def log_energy_range(self) -> Tuple[float, float]:
        return np.log(self.energy_range[0]), np.log(self.energy_range[1])

    @property
    def log10_energy_range(self) -> Tuple[float, float]:
        return np.log10(self.energy_range[0]), np.log10(self.energy_range[1])


class ToyNuDet:
    """Class to handle toys related to detector systematics."""

    def __init__(
        self,
        nobserved: np.ndarray,
        nbackground: np.ndarray,
        var_acceptance: np.ndarray,
        events: Optional[List[List[NuEvent]]] = None,
    ):
        self.nobserved = np.array(nobserved)
        self.nbackground = np.array(nbackground)
        self.var_acceptance = np.array(var_acceptance)
        self.events = events

    def __str__(self):
        return "ToyNuDet: n(observed)=%s, n(background)=%s, var(acceptance)=%s, events=%s" % (
            self.nobserved,
            self.nbackground,
            self.var_acceptance,
            self.events,
        )


class NuDetectorBase(abc.ABC):
    """Class to handle the neutrino detector information."""

    def __init__(self):
        self.name = None
        self._samples = []
        self.error_acceptance = None

    @property
    def samples(self):
        return self._samples

    @property
    def nsamples(self):
        return len(self._samples)

    def get_acceptances(self, spectrum: str):
        accs = []
        for sample in self.samples:
            if spectrum not in sample.acceptances:
                raise RuntimeError("Acceptance for spectrum %s is not available in sample %s" % (spectrum, sample.name))
            accs.append(sample.acceptances[spectrum])
        nsides = np.array([acc.nside for acc in accs])
        if not np.all(np.isin(nsides, [0, max(nsides)])):
            raise RuntimeError("All acceptance maps are not in the same resolution. Exiting!")
        return accs, max(nsides)

    def get_nonempty_acceptance_pixels(self, spectrum: str, nside: int):
        accs, _ = self.get_acceptances(spectrum)
        npix = hp.nside2npix(nside)
        acctot = np.zeros(npix)
        for acc in accs:
            for ipix in range(npix):
                acctot[ipix] += acc.evaluate(ipix, nside)
        return np.nonzero(acctot)[0]

    def prepare_toys(self, ntoys: int = 0) -> List[ToyNuDet]:

        toys = []
        nobserved = np.array([s.nobserved for s in self.samples])
        background = np.array([s.background for s in self.samples])
        events = [s.events for s in self.samples]

        if np.any(np.isnan(nobserved)):
            raise RuntimeError("[NuDetector] The number of observed events is not correctly filled.")

        # if no toys
        if ntoys == 0:
            nbackground = [bkg.nominal for bkg in background]
            toy = ToyNuDet(nobserved, nbackground, np.ones(self.nsamples), events)
            return [toy]

        # acceptance toys
        toys_acceptance = multivariate_normal.rvs(mean=np.ones(self.nsamples), cov=self.error_acceptance, size=ntoys)
        for i in range(ntoys):
            while np.any(toys_acceptance[i] < 0):
                toys_acceptance[i] = multivariate_normal.rvs(mean=np.ones(self.nsamples), cov=self.error_acceptance)

        # background toys
        toys_background = np.array([bkg.prepare_toys(ntoys) for bkg in background]).T

        for i in range(ntoys):
            toys.append(ToyNuDet(nobserved, toys_background[i], toys_acceptance[i], events))
        return toys


class NuDetector(NuDetectorBase):
    """Class to handle the neutrino detector information."""

    def __init__(self, infile: Optional[Union[dict, str]] = None):
        super().__init__()
        self.earth_location = None
        self.error_acceptance_corr = None
        if infile is not None:
            self.load(infile)

    # setter functions

    def load(self, rinput: Union[dict, str]):
        """Load the detector configuration from either
        - JSON file (format defined in the examples folder
        - dictionary object (with same format as JSON).
        """
        log = logging.getLogger("jang")

        if isinstance(rinput, str):  # pragma: no cover
            assert os.path.isfile(rinput)
            with open(rinput) as f:
                data = yaml.safe_load(f)
            log.info("[NuDetector] Object is loaded from the file %s.", rinput)
        elif isinstance(rinput, dict):
            data = rinput
            log.info("[NuDetector] Object is loaded from a dictionary object.")
        else:
            raise TypeError("Unknown input format for Detector constructor")
        self.name = data["name"]
        if "earth_location" in data:
            unit = Unit(data["earth_location"]["units"])
            self.earth_location = astropy.coordinates.EarthLocation(
                lat=data["earth_location"]["latitude"] * unit,
                lon=data["earth_location"]["longitude"] * unit,
            )
        for i in range(data["nsamples"]):
            smp = NuSample(
                name=data["samples"]["names"][i],
                shortname=data["samples"]["shortnames"][i] if "shortnames" in data["samples"] else None,
            )
            data["samples"]["energyrange"] = np.array(data["samples"]["energyrange"], dtype=float)
            if data["samples"]["energyrange"].shape == (data["nsamples"], 2):
                smp.set_energy_range(*data["samples"]["energyrange"][i])
            elif data["samples"]["energyrange"].shape == (2,):
                smp.set_energy_range(*data["samples"]["energyrange"])
            else:
                raise RuntimeError("[NuDetector] Unknown format for energy range.")
            self._samples.append(smp)
        if "errors" in data:
            self.error_acceptance = data["errors"]["acceptance"]
            self.error_acceptance_corr = data["errors"]["acceptance_corr"]
        else:
            self.error_acceptance = 0
            self.error_acceptance_corr = 0
        self.check_errors_validity()

    def set_observations(self, nobserved: list, background: list):
        if len(nobserved) != self.nsamples:
            raise RuntimeError("[NuDetector] Incorrect size for nobserved as compared to the number of samples.")
        if len(background) != self.nsamples:
            raise RuntimeError("[NuDetector] Incorrect size for nbackground as compared to the number of samples.")
        for i, smp in enumerate(self.samples):
            smp.set_observations(nobserved[i], background[i])

    def set_acceptances(self, acceptances: list, spectrum: str, nside: Optional[int] = None):
        if len(acceptances) != self.nsamples:
            raise RuntimeError("[NuDetector] Incorrect number of acceptance maps as compared to the number of samples.")
        for acceptance, sample in zip(acceptances, self.samples):
            sample.set_acceptance(acceptance, spectrum, nside)

    # converters

    def jd_to_lst(self, jd: float) -> float:  # pragma: no cover
        t = astropy.time.Time(jd, format="jd")
        return t.sidereal_time("apparent", longitude=self.earth_location)

    def radec_to_altaz(self, ra: Quantity, dec: Quantity, jd: float) -> Tuple[Quantity, Quantity]:
        c_eq = astropy.coordinates.SkyCoord(ra=ra, dec=dec, frame="icrs")
        c_loc = c_eq.transform_to(
            astropy.coordinates.AltAz(obstime=astropy.time.Time(jd, format="jd"), location=self.earth_location)
        )
        return c_loc.alt, c_loc.az

    def altaz_to_radec(self, alt: Quantity, az: Quantity, jd: float) -> Tuple[Quantity, Quantity]:
        altaz = astropy.coordinates.AltAz(
            alt=alt,
            az=az,
            obstime=astropy.time.Time(jd, format="jd"),
            location=self.earth_location,
        )
        radec = altaz.transform_to(astropy.coordinates.ICRS())
        return radec.ra, radec.dec

    # other

    def check_errors_validity(self):
        self.error_acceptance = infer_uncertainties(
            self.error_acceptance, self.nsamples, correlation=self.error_acceptance_corr
        )


class SuperNuDetector(NuDetectorBase):
    """Class to handle several detectors simultaneously."""

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.detectors = []

    @property
    def samples(self):
        return itertools.chain.from_iterable([det.samples for det in self.detectors])

    @property
    def nsamples(self):
        return sum([det.nsamples for det in self.detectors])

    def add_detector(self, det: NuDetector):
        log = logging.getLogger("jang")
        if det.name in [d.name for d in self.detectors]:
            log.error(
                "[SuperDetector] Detector with same name %s is already loaded in the SuperDetector. Skipped.",
                det.name,
            )
            return
        log.info("[SuperDetector] Detector %s is added to the SuperDetector.", det.name)
        self.detectors.append(det)
        self.error_acceptance = block_diag(*[d.error_acceptance for d in self.detectors])


class EffectiveAreaBase(abc.ABC):
    """Class to handle detector effective area for a given sample and neutrino flavour."""

    def __init__(self, sample: NuSample):
        self.sample = sample
        self.args_evaluate = []

    @abc.abstractmethod  # pragma: no cover
    def evaluate(self, energy: Union[float, Iterable]):
        pass

    def to_acceptance(self, detector: NuDetector, nside: int, jd: float, spectrum: str):
        if nside is None or nside <= 0:
            raise RuntimeError("A positive nside should be provided!")
        npix = hp.nside2npix(nside)
        acc_map = np.zeros(npix)
        dec, ra = hp.pix2ang(nside, range(npix))
        dec, ra = (np.pi / 2 - dec), ra

        f_spectrum = eval("lambda x: %s" % spectrum)

        def func(x: float, *args):
            return self.evaluate(np.exp(x), *args) * f_spectrum(np.exp(x)) * np.exp(x)

        for ipix in range(npix):

            if "altitude" in self.args_evaluate and "azimuth" in self.args_evaluate:
                alt, az = detector.radec_to_altaz(ra * rad, dec * rad, jd)
                arg = (alt[ipix].rad, az[ipix].rad)
            elif "altitude" in self.args_evaluate:
                alt, az = detector.radec_to_altaz(ra * rad, dec * rad, jd)
                arg = (alt[ipix].rad,)
            elif "ra" in self.args_evaluate and "dec" in self.args_evaluate:
                arg = (ra[ipix], dec[ipix])
            elif "dec" in self.args_evaluate:
                arg = (dec[ipix],)
            else:
                arg = ()

            acc_map[ipix], _ = scipy.integrate.quad(
                func,
                *(self.sample.log_energy_range),
                args=arg,
                limit=500,
            )
        return np.array(acc_map)
