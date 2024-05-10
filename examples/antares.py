"""Example to handle ANTARES specific format.

In this case, the input from ANTARES are simply the number of events (observed and expected),
as well as the acceptances in the format of skymap in equatorial coordinates
(assuming a spectrum E^-2 if we want to probe such spectrum).
The provided values in this example are all dummy.
"""

import healpy as hp
import numpy as np
import os

import jang.conversions
import jang.gw
import jang.limits
import jang.results
import jang.significance
from jang.neutrinos import Detector, BackgroundGaussian
from jang.parameters import Parameters


def single_event(
    gwname: str, gwdbfile: str, det_results: dict, pars: Parameters, dbfile: str = None
):
    """Compute the limits for a given GW event and using the detector results stored in dictionary.
    If dbfile is provided, the obtained results are stored in a database at this path.

    The `det_results` dictionary should contain the following keys:
        - nobs: list of observed number of events (length = 4 [number of samples])
        - nbkg: list of expected number of events (length = 4 [number of samples])
        - acceptances: list of files containing acceptance skymaps (one per sample)
    """

    database_gw = jang.gw.Database(gwdbfile)
    database_res = jang.results.Database(dbfile)

    antares = Detector("examples/input_files/detector_antares.yaml")
    gw = database_gw.find_gw(gwname, pars)

    antares.set_acceptances(det_results["acceptances"], pars.spectrum, pars.nside)
    bkg = [BackgroundGaussian(b, 0.20 * b) for b in det_results["nbkg"]]
    antares.set_observations(det_results["nobs"], bkg)
    pars.nside = antares.get_acceptances(pars.spectrum)[1]

    jang.limits.get_limitmap_flux(antares, gw, pars, "tests/test")

    def pathpost(x):
        return f"{os.path.dirname(dbfile)}/lkls/{x}_{gw.name}_{antares.name}_{pars.str_filename}" if dbfile is not None else None
    limit_flux = jang.limits.get_limit_flux(antares, gw, pars, pathpost("flux"))
    limit_etot = jang.limits.get_limit_etot(antares, gw, pars, pathpost("eiso"))
    limit_fnu = jang.limits.get_limit_fnu(antares, gw, pars, pathpost("fnu"))

    jang.significance.compute_prob_null_hypothesis(antares, gw, pars)

    database_res.add_entry(antares, gw, pars, limit_flux, limit_etot, limit_fnu, pathpost("flux"), pathpost("eiso"), pathpost("fnu"))
    if dbfile is not None:
        database_res.save()


if __name__ == "__main__":

    parameters = Parameters("examples/input_files/config.yaml")
    parameters.set_models("x**-2", jang.conversions.JetIsotropic())

    gwdb = "examples/input_files/gw_catalogs/database_example.csv"
    npix = hp.nside2npix(8)
    detresults = {
        "nobs": [1, 0, 0, 0],
        "nbkg": [0.5, 0.1, 0.3, 0.1],
        "acceptances": [2*np.ones(npix), np.ones(npix), 0.5*np.ones(npix), np.ones(npix)]
    }
    single_event("GW190412", gwdb, detresults, parameters)
