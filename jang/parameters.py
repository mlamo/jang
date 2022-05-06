"""Class to hold general analysis parameters."""

import numpy as np
import os
import yaml
from typing import Optional

from jang.conversions import JetModelBase


class Parameters:
    def __init__(self, file: Optional[str] = None):
        self.file = None
        self.spectrum = None
        self.jet = None
        if file is not None:
            assert os.path.isfile(file)
            self.file = file
            with open(self.file, "r") as f:
                params = yaml.safe_load(f)
            self.nside = params["analysis"]["nside"]
            if self.nside < 0:
                self.nside = None
            self.apply_det_systematics = params["analysis"]["apply_det_systematics"]
            self.ntoys_det_systematics = params["analysis"]["ntoys_det_systematics"]
            self.search_region = params["analysis"]["search_region"]
            self.range_flux = np.array(params["range"]["log10_flux"], dtype=int)
            self.range_etot = np.array(params["range"]["log10_etot"], dtype=int)
            self.range_fnu = np.array(params["range"]["log10_fnu"], dtype=int)
            self.range_energy_integration = np.array(
                params["range"]["neutrino_energy_GeV"], dtype=float
            )
            self.prior_signal = params["analysis"]["prior_signal"]

    def set_models(
        self, spectrum: Optional[str] = None, jet: Optional[JetModelBase] = None
    ):
        """Set the neutrino spectrum model (format 'x**-2') and jet model."""
        if spectrum is not None:
            self.spectrum = spectrum
        if jet is not None:
            self.jet = jet

    @property
    def str_filename(self):
        """Get the representation of the parameters in string format for suffixing filenames."""
        str_model = self.spectrum.replace("x", "E").replace("**", "")
        if self.jet is not None:
            str_model += "_" + self.jet.str_filename
        return str_model

    def get_searchregion_gwfraction(self) -> float:
        spl = self.search_region.split("_")
        if len(spl) >= 2 and spl[0] == "region":
            return float(spl[1]) / 100
        if len(spl) == 1 and spl[0] == "bestfit":
            return 0
        return None

    def get_searchregion_iszeroincluded(self) -> bool:
        """Returns True if the pixels with zero acceptance should be included."""
        spl = self.search_region.split("_")
        if spl[-1] == "excludezero":
            return False
        return True
