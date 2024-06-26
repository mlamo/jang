from cached_property import cached_property
import itertools
import numpy as np

import jang.utils.conversions
from jang.io import GW, NuDetectorBase, Parameters
from jang.io.utils import get_search_region


class Analysis:
    def __init__(self, gw: GW, detector: NuDetectorBase, parameters: Parameters):
        self._gw = gw
        self._detector = detector
        self._parameters = parameters
        self._gwvars = ["ra", "dec"]
        self.toys_gw = self.toys_det = None

    def add_gw_variables(self, *args):
        for v in args:
            if v not in self._gwvars:
                self._gwvars.append(v)

    @cached_property
    def acceptances(self):
        acceptances, nside = self._detector.get_acceptances(self._parameters.spectrum)
        if self._parameters.nside is None:
            self._parameters.nside = nside  # pragma: no cover
        elif self._parameters.nside != nside:
            raise RuntimeError("Something went wrong with map resolutions!")  # pragma: no cover
        return acceptances

    def prepare_toys(self, add_gw_vars: list = None, fixed_gwpixel: int = None):
        if add_gw_vars is not None:
            self.add_gw_variables(*add_gw_vars)
        _ = self.acceptances
        # GW toys
        region_restricted = get_search_region(self._detector, self._gw, self._parameters)
        if fixed_gwpixel is not None:
            self.toys_gw = self._gw.fits.prepare_toy(nside=self._parameters.nside, fixed_pixel=fixed_gwpixel)
        else:
            self.toys_gw = self._gw.prepare_toys(*self._gwvars, nside=self._parameters.nside, region_restriction=region_restricted)
            
        # Detector toys
        if self._parameters.apply_det_systematics:
            self.toys_det = self._detector.prepare_toys(self._parameters.ntoys_det_systematics)
        else:
            self.toys_det = self._detector.prepare_toys(0)

    @property
    def toys(self):
        if self.toys_det is None or self.toys_gw is None:
            self.prepare_toys()
        return itertools.product(self.toys_gw, self.toys_det)

    def phi_to_nsig(self, toy: tuple):
        phi_to_nsig = [
            acc.evaluate(toy[0].ipix, nside=self._parameters.nside) for i, acc in enumerate(self.acceptances)
        ]
        phi_to_nsig = np.array(phi_to_nsig)
        phi_to_nsig *= toy[1].var_acceptance
        phi_to_nsig /= 6
        return phi_to_nsig

    def eiso_to_phi(self, toy: tuple):
        return jang.utils.conversions.eiso_to_phi(
            self._parameters.range_energy_integration,
            self._parameters.spectrum,
            toy[0].luminosity_distance,
        )

    def etot_to_eiso(self, toy: tuple):
        return jang.utils.conversions.etot_to_eiso(toy[0].theta_jn, self._parameters.jet)

    def fnu_to_etot(self, toy: tuple):
        return jang.utils.conversions.fnu_to_etot(toy[0].radiated_energy)

    def eiso_to_nsig(self, toy: tuple):
        return self.eiso_to_phi(toy) * self.phi_to_nsig(toy)

    def etot_to_nsig(self, toy: tuple):
        return self.etot_to_eiso(toy) * self.eiso_to_nsig(toy)

    def fnu_to_nsig(self, toy: tuple):
        return self.fnu_to_etot(toy) * self.etot_to_nsig(toy)
