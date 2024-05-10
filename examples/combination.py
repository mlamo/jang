"""Example to perform the combination between ANTARES and Super-Kamiokande.

This is taking benefit of the the specific formats described in ANTARES and Super-Kamiokande examples.
All provided values are dummy.
"""

import healpy as hp
import numpy as np

import jang.gw
import jang.limits
import jang.results
from jang.parameters import Parameters
from jang.neutrinos import BackgroundFixed, Detector, SuperDetector

from examples.superkamiokande import EffectiveAreaSK


def single_event(gwname: str, gwdbfile: str, ant_results: dict, sk_results: dict, pars: Parameters, dbfile: str = None):

    database_gw = jang.gw.Database(gwdbfile)
    database_res = jang.results.Database(dbfile)

    antares = Detector("examples/input_files/detector_antares.yaml")
    sk = Detector("examples/input_files/detector_superk.yaml")
    effarea_sk = [EffectiveAreaSK(filename=sk_results["effarea"], sample=s) for s in sk.samples]

    # Combination
    ant_sk = SuperDetector("ANTARES + Super-Kamiokande")
    ant_sk.add_detector(antares)
    ant_sk.add_detector(sk)

    gw = database_gw.find_gw(gwname, pars)

    antares.set_acceptances(ant_results["acceptances"], pars.spectrum, pars.nside)
    bkg_ant = [BackgroundFixed(b) for b in ant_results["nbkg"]]
    antares.set_observations(ant_results["nobs"], bkg_ant)
    pars.nside = antares.get_acceptances(pars["spectrum"])[0].nside

    accs = [
        effarea.to_acceptance(sk, pars.nside, gw.jd, pars.spectrum)
        for effarea in effarea_sk
    ]
    sk.set_acceptances(accs, pars.spectrum, pars.nside)
    bkg_sk = [BackgroundFixed(b) for b in sk_results["nbkg"]]
    sk.set_observations(sk_results["nobs"], bkg_sk)

    limit_flux = jang.limits.get_limit_flux(ant_sk, gw, pars)
    limit_etot = jang.limits.get_limit_etot(ant_sk, gw, pars)
    limit_fnu = jang.limits.get_limit_fnu(ant_sk, gw, pars)
    database_res.add_entry(ant_sk, gw, pars, limit_flux, limit_etot, limit_fnu)

    if dbfile is not None:
        database_res.save()


if __name__ == "__main__":

    pars = Parameters("examples/input_files/config.yaml")
    pars.set_models("x**-2", jang.conversions.JetIsotropic())

    gw_db_file = "examples/input_files/gw_catalogs/database_example.csv"
    npix = hp.nside2npix(8)
    ant_results = {
        "nobs": [0, 0, 0, 0],
        "nbkg": [0, 0, 0, 0],
        "acceptances": [np.ones(npix), np.zeros(npix), np.ones(npix), np.zeros(npix)],
    }
    sk_results = {
        "nobs": [0, 0, 0],
        "nbkg": [0, 0, 0],
        "effarea": "examples/input_files/effarea_superk.root",
    }
    single_event("GW190412", gw_db_file, ant_results, sk_results, pars)
