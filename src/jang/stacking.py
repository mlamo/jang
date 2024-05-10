import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
from typing import Optional

import jang.parameters
import jang.results

matplotlib.use("Agg")
np.seterr(divide="ignore")


def stack_events_listgw(db: jang.results.Database, list_gw: list, pars: jang.parameters.Parameters):

    log = logging.getLogger("jang")
    triggers = []

    etot_arr = np.logspace(*pars.range_etot)
    fnu_arr = np.logspace(*pars.range_fnu)
    loglkl_etot = np.zeros_like(etot_arr)
    loglkl_fnu = np.zeros_like(fnu_arr)

    for _, ev in db.db.iterrows():
        if ev["GW.name"] not in list_gw:
            continue
        if not ev["Results.posterior_etot"] or not ev["Results.posterior_fnu"]:
            log.warning("Missing posterior files for %s, skipping...", ev["GW.name"])
            continue
        if ev["GW.name"] in triggers:
            log.error("%s is present several times in the database, skipping this one!", ev["GW.name"])
            continue
        triggers.append(ev["GW.name"])
        #
        x, y = np.load(ev["Results.posterior_etot"] + ".npy")
        f = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
        loglkl_etot += np.log(f(etot_arr))
        #
        x, y = np.load(ev["Results.posterior_fnu"] + ".npy")
        f = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
        loglkl_fnu += np.log(f(fnu_arr))
    if len(triggers) != len(list_gw):
        log.error("Some wanted GWs have not been found.")
    lkl_etot = np.exp(loglkl_etot - loglkl_etot[0])
    limit_etot = jang.stats.compute_upperlimit_from_x_y(etot_arr, lkl_etot)

    lkl_fnu = np.exp(loglkl_fnu - loglkl_fnu[0])
    limit_fnu = jang.stats.compute_upperlimit_from_x_y(fnu_arr, lkl_fnu)

    return len(triggers), limit_etot, limit_fnu


def stack_events_weightedevents(
    db: jang.results.Database,
    gw_withweights: dict,
    pars: jang.parameters.Parameters,
    npe: int = 10000,
    outfile: Optional[str] = None,
):

    log = logging.getLogger("jang")

    etot_arr = np.logspace(*pars.range_etot)
    fnu_arr = np.logspace(*pars.range_fnu)
    loglkl_etot = {}
    loglkl_fnu = {}

    for _, ev in db.db.iterrows():
        if not ev["Results.posterior_etot"] or not ev["Results.posterior_fnu"]:
            log.warning("Missing posterior files for %s, skipping...", ev["GW.name"])
            continue
        if ev["GW.name"] in loglkl_etot.keys():
            log.error("%s is present several times in the database, skipping this one!", ev["GW.name"])
            continue
        #
        x, y = np.load(ev["Results.posterior_etot"] + ".npy")
        f = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
        loglkl_etot[ev["GW.name"]] = np.log(f(etot_arr))
        #
        x, y = np.load(ev["Results.posterior_fnu"] + ".npy")
        f = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
        loglkl_fnu[ev["GW.name"]] = np.log(f(fnu_arr))
    if len(loglkl_fnu.items()) != len(gw_withweights):
        log.error("Some wanted GWs have not been found.")
        return np.inf, np.inf
    results = []
    for _ in range(npe):
        sumloglkl_etot = np.zeros_like(etot_arr)
        sumloglkl_fnu = np.zeros_like(fnu_arr)
        ngws = 0
        for gw, prob in gw_withweights.items():
            if np.random.random() < prob:
                sumloglkl_etot += loglkl_etot[gw]
                sumloglkl_fnu += loglkl_fnu[gw]
            ngws += 1
        lkl_etot = np.exp(sumloglkl_etot - sumloglkl_etot[0])
        limit_etot = jang.stats.compute_upperlimit_from_x_y(etot_arr, lkl_etot)
        lkl_fnu = np.exp(sumloglkl_fnu - sumloglkl_fnu[0])
        limit_fnu = jang.stats.compute_upperlimit_from_x_y(fnu_arr, lkl_fnu)
        results.append([ngws, limit_etot, limit_fnu])
    results = np.array(results)

    plt.close("all")
    min_etot = np.min(np.ma.masked_invalid(np.log10(results[:, 1]))) - 0.1
    max_etot = np.max(np.ma.masked_invalid(np.log10(results[:, 1]))) + 0.1
    plt.hist(results[:, 1], bins=np.logspace(min_etot, max_etot, 31))
    plt.xscale("log")
    plt.xlabel(r"$E^{90\%}_{tot,\nu}$ [erg]")
    plt.ylabel("# of pseudo-experiments")
    plt.axvline(np.median(results[:, 1]), color="red")
    if outfile is not None:
        plt.savefig(outfile, dpi=300)

    plt.close("all")
    min_fnu = np.min(np.ma.masked_invalid(np.log10(results[:, 2]))) - 0.1
    max_fnu = np.max(np.ma.masked_invalid(np.log10(results[:, 2]))) + 0.1
    plt.hist(results[:, 2], bins=np.logspace(min_fnu, max_fnu, 31))
    plt.xscale("log")
    plt.xlabel(r"$E^{90\%}_{tot,\nu}/E_{radiated}$")
    plt.ylabel("# of pseudo-experiments")
    plt.axvline(np.median(results[:, 2]), color="red")
    if outfile is not None:
        plt.savefig(f"{os.path.splitext(outfile)[0]}_fnu{os.path.splitext(outfile)[1]}", dpi=300)

    return np.median(results[:, 1]), np.median(results[:, 2])


def stack_events(db: jang.results.Database, pars: jang.parameters.Parameters):

    list_gw = list(db.db["GW.name"])
    return stack_events_listgw(db, list_gw, pars)
