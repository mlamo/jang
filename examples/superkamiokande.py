"""Example to handle Super-Kamiokande specific format.

In this case, the input from Super-K are simply the number of events (observed and expected),
as well as the detector effective area in the format of 2D histograms with x=log10(energy [in GeV]) and y=zenith angle [in rad].
The provided values in this example are all dummy, except for the effective area that is the one published in https://doi.org/10.5281/zenodo.4724822.
"""

from collections.abc import Iterable
import numpy as np
import os
import ROOT
from typing import Union

import jang.conversions
import jang.gw
import jang.limits
import jang.results
from jang.neutrinos import EffectiveAreaBase, BackgroundFixed, Detector, Sample
from jang.parameters import Parameters


class EffectiveAreaSK(EffectiveAreaBase):
    def __init__(self, filename: str, sample: Sample):
        super().__init__(sample)
        self.filename = filename
        self.rootfile = self.rootgraphs = None
        self.args_evaluate = ["altitude"]
        self.read()

    def read(self):
        assert os.path.isfile(self.filename)
        self.rootfile = ROOT.TFile(self.filename, "r")
        self.rootgraphs = []
        for flav in ("nue", "nueb", "numu", "numub"):
            gname = f"Aeff_2D_{self.sample.shortname}_{flav}"
            if self.rootfile.GetListOfKeys().Contains(gname):
                self.rootgraphs.append(self.rootfile.Get(gname))

    def evaluate(self, energy: Union[float, Iterable], altitude: float) -> Union[float, np.ndarray]:
        if isinstance(energy, Iterable):
            aeff = np.zeros_like(energy)
            for i, x in enumerate(np.log10(energy)):
                aeff[i] = np.sum([r.Interpolate(x, np.pi / 2 - altitude) for r in self.rootgraphs])
            return aeff
        return np.sum([r.Interpolate(np.log10(energy), np.pi / 2 - altitude) for r in self.rootgraphs])


def single_event(gwname: str, gwdbfile: str, det_results: dict, pars: Parameters, dbfile: str = None):
    """Compute the limits for a given GW event and using the detector results stored in dictionary.
    If dbfile is provided, the obtained results are stored in a database at this path.

    The `det_results` dictionary should contain the following keys:
        - nobs: list of observed number of events (length = 4 [number of samples])
        - nbkg: list of expected number of events (length = 4 [number of samples])
        - effarea: path to the effective area file
    """

    database_gw = jang.gw.Database(gwdbfile)
    database_res = jang.results.Database(dbfile)

    sk = Detector("examples/input_files/detector_superk.yaml")
    effarea_sk = [
        EffectiveAreaSK(filename=det_results["effarea"], sample=s) for s in sk.samples
    ]
    gw = database_gw.find_gw(gwname, pars)

    accs = [
        effarea.to_acceptance(sk, pars.nside, gw.jd, pars.spectrum)
        for effarea in effarea_sk
    ]
    sk.set_acceptances(accs, pars.spectrum)
    bkg = [BackgroundFixed(b) for b in det_results["nbkg"]]
    sk.set_observations(det_results["nobs"], bkg)

    def pathpost(x):
        return f"{os.path.dirname(dbfile)}/lkls/{x}_{gw.name}_{sk.name}_{pars.str_filename}" if dbfile is not None else None
    limit_flux = jang.limits.get_limit_flux(sk, gw, pars, pathpost("flux"))
    limit_etot = jang.limits.get_limit_etot(sk, gw, pars, pathpost("eiso"))
    limit_fnu = jang.limits.get_limit_fnu(sk, gw, pars, pathpost("fnu"))

    jang.significance.compute_prob_null_hypothesis(sk, gw, pars)

    database_res.add_entry(sk, gw, pars, limit_flux, limit_etot, limit_fnu, pathpost("flux"), pathpost("eiso"), pathpost("fnu"))
    if dbfile is not None:
        database_res.save()


if __name__ == "__main__":

    parameters = Parameters("examples/input_files/config.yaml")
    parameters.set_models("x**-2", jang.conversions.JetIsotropic())
    parameters.nside = 8

    gwdb = "examples/input_files/gw_catalogs/database_example.csv"
    detresults = {
        "nobs": [0, 0, 0],
        "nbkg": [0, 0, 0],
        "effarea": "examples/input_files/effarea_superk.root",
    }
    single_event("GW190412", gwdb, detresults, parameters)
