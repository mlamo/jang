import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

from jang import DisableLogger
from jang.gw import GW
from jang.limits import get_limit_flux
from jang.neutrinos import DetectorBase, EffectiveAreaBase
from jang.parameters import Parameters

plt.close("all")


def prepare_acceptances(
    det: DetectorBase, aeffs: List[EffectiveAreaBase], energy_bins: List[Tuple], pars: Parameters, jd: float
):

    accs = []
    for aeff, sample in zip(aeffs, det.samples):
        for energy_bin in energy_bins:
            acc = {"sample": sample.name, "energy_bin": energy_bin}
            new_sample = copy.deepcopy(sample)
            new_sample.set_energy_range(*energy_bin)
            new_aeff = copy.deepcopy(aeff)
            new_aeff.sample = new_sample
            acc["acceptance"] = new_aeff.to_acceptance(det, pars.nside, jd, pars.spectrum)
            accs.append(acc)

    return accs


def get_flux_limits(
    detector: DetectorBase, aeffs: List[EffectiveAreaBase], gw: GW, parameters: Parameters, energy_bins: List[Tuple]
):

    results = prepare_acceptances(detector, aeffs, energy_bins, parameters, gw.jd)

    newdet = copy.deepcopy(detector)
    for r in results:
        if np.all(r["acceptance"] == 0):
            r["limit"] = np.inf
            continue
        accs = [r["acceptance"] if s.name == r["sample"] else 0 for s in newdet.samples]
        newdet.set_acceptances(accs, parameters.spectrum, parameters.nside)
        with DisableLogger():
            r["limit"] = get_limit_flux(newdet, gw, parameters)

    return results


def plot_flux_limits(outfile: str, results: List[dict], sample_styles: dict):

    plt.close("all")
    for r in results:
        plt.plot(
            r["energy_bin"],
            [r["limit"]] * 2,
            color=sample_styles[r["sample"]]["color"],
            linestyle=sample_styles[r["sample"]]["style"] if "style" in sample_styles[r["sample"]] else "-",
        )

    plt.xlabel("True neutrino energy [GeV]", fontsize=15)
    plt.ylabel(r"Upper limit on $E^2 dn/dE$ [GeV/cm$^2$]", fontsize=15)

    handles = []
    for s, v in sample_styles.items():
        handles.append(
            matplotlib.lines.Line2D([], [], label=s, color=v["color"], linestyle=v["style"] if "style" in v else "-")
        )
    plt.legend(handles=handles, fontsize=13)
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
