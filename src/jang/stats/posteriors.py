"""Computation of posteriors."""

from jang.stats.utils import PosteriorVariable
import numpy as np
from typing import List, Tuple

from jang.gw import GW
from jang.neutrinos import Detector
from jang.parameters import Parameters
from jang.analysis import Analysis
import jang.stats.likelihoods as lkl
import jang.stats.priors as prior


def compute_flux_posterior(
    variables: List[PosteriorVariable], detector: Detector, gw: GW, parameters: Parameters, fixed_gwpixel: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior as a function of all-flavour neutrino flux at Earth.
    The list of 'variables' **SHOULD** contain a PosteriorVariable with name 'flux'.

    Args:
        variables (List[PosteriorVariable]): list of variables to be used in the likelihood and kept in the marginalised posterior.
        detector (Detector): holds the nominal results
        gw (GW): holds the gravitational wave information
        parameters (Parameters): holds the needed parameters (skymap resolution to be used, neutrino spectrum and integration range...)
        fixed_gwpixel (int): probe only the direction given by this pixel id (for resolution=parameters.nside)

    Returns:
        np.ndarray: array of the variable flux
        np.ndarray: array of computed posterior
    """

    ana = Analysis(gw=gw, detector=detector, parameters=parameters)

    # build parameter space (needs at least 'flux' variable)
    arr_vars = np.meshgrid(*[var.array for var in variables], indexing="ij")
    arr_vars = {var.name: arr_vars[i] for i, var in enumerate(variables)}
    arr_vars[0] = arr_vars.pop("flux")
    arr_post = np.zeros_like(arr_vars[0])

    ana.prepare_toys(fixed_gwpixel=fixed_gwpixel)

    for toy in ana.toys:
        phi_to_nsig = ana.phi_to_nsig(toy)
        if parameters.likelihood_method == "poisson":
            arr_post += lkl.poisson_several_samples(
                toy[1].nobserved, toy[1].nbackground, phi_to_nsig, arr_vars
            ) * prior.signal_parameter(arr_vars[0], toy[1].nbackground, phi_to_nsig, parameters.prior_signal)
        elif parameters.likelihood_method == "pointsource":
            arr_post += lkl.pointsource_several_samples(
                toy[1].nobserved,
                toy[1].nbackground,
                toy[1].events,
                phi_to_nsig,
                detector.samples,
                toy[0].ra,
                toy[0].dec,
                arr_vars,
            ) * prior.signal_parameter(arr_vars[0], toy[1].nbackground, phi_to_nsig, parameters.prior_signal)

    return arr_vars, arr_post


def compute_etot_posterior(
    variables: List[PosteriorVariable], detector: Detector, gw: GW, parameters: Parameters
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior as a function of total energy. The list of 'variables' **SHOULD** contain a PosteriorVariable with name 'etot'.

    Args:
        variables (List[PosteriorVariable]): list of variables to be used in the likelihood and kept in the marginalised posterior.
        detector (Detector): holds the nominal results
        gw (GW): holds the gravitational wave information
        parameters (Parameters): holds the needed parameters (skymap resolution to be used, neutrino spectrum and integration range...)

    Returns:
        np.ndarray: array of the variable Etot
        np.ndarray: array of computed posterior
    """

    ana = Analysis(gw=gw, detector=detector, parameters=parameters)
    ana.add_gw_variables("luminosity_distance", "theta_jn")

    # build parameter space (needs at least 'flux' variable)
    arr_vars = np.meshgrid(*[var.array for var in variables], indexing="ij")
    arr_vars = {var.name: arr_vars[i] for i, var in enumerate(variables)}
    arr_vars[0] = arr_vars.pop("etot")
    arr_post = np.zeros_like(arr_vars[0])

    for toy in ana.toys:
        etot_to_nsig = ana.etot_to_nsig(toy)
        if parameters.likelihood_method == "poisson":
            arr_post += lkl.poisson_several_samples(
                toy[1].nobserved, toy[1].nbackground, etot_to_nsig, arr_vars
            ) * prior.signal_parameter(arr_vars[0], toy[1].nbackground, etot_to_nsig, parameters.prior_signal)
        elif parameters.likelihood_method == "pointsource":
            arr_post += lkl.pointsource_several_samples(
                toy[1].nobserved,
                toy[1].nbackground,
                toy[1].events,
                etot_to_nsig,
                detector.samples,
                toy[0].ra,
                toy[0].dec,
                arr_vars,
            ) * prior.signal_parameter(arr_vars[0], toy[1].nbackground, etot_to_nsig, parameters.prior_signal)
    return arr_vars, arr_post


def compute_fnu_posterior(
    variables: List[PosteriorVariable], detector: Detector, gw: GW, parameters: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior as a function of fnu=E(tot)/E(radiated). The list of 'variables' **SHOULD** contain a PosteriorVariable with name 'fnu'.

    Args:
        variables (List[PosteriorVariable]): list of variables to be used in the likelihood and kept in the marginalised posterior.
        detector (Detector): holds the nominal results
        gw (GW): holds the gravitational wave information
        parameters (Parameters): holds the needed parameters (skymap resolution to be used, neutrino spectrum and integration range...)

    Returns:
        np.ndarray: array of the variable fnu
        np.ndarray: array of computed posterior
    """

    ana = Analysis(gw=gw, detector=detector, parameters=parameters)
    ana.add_gw_variables("luminosity_distance", "theta_jn", "radiated_energy")

    # build parameter space (needs at least 'flux' variable)
    arr_vars = np.meshgrid(*[var.array for var in variables], indexing="ij")
    arr_vars = {var.name: arr_vars[i] for i, var in enumerate(variables)}
    arr_vars[0] = arr_vars.pop("fnu")
    arr_post = np.zeros_like(arr_vars[0])

    for toy in ana.toys:
        fnu_to_nsig = ana.fnu_to_nsig(toy)
        if parameters.likelihood_method == "poisson":
            arr_post += lkl.poisson_several_samples(
                toy[1].nobserved, toy[1].nbackground, fnu_to_nsig, arr_vars
            ) * prior.signal_parameter(arr_vars[0], toy[1].nbackground, fnu_to_nsig, parameters.prior_signal)
        elif parameters.likelihood_method == "pointsource":
            arr_post += lkl.pointsource_several_samples(
                toy[1].nobserved,
                toy[1].nbackground,
                toy[1].events,
                fnu_to_nsig,
                detector.samples,
                toy[0].ra,
                toy[0].dec,
                arr_vars,
            ) * prior.signal_parameter(arr_vars[0], toy[1].nbackground, fnu_to_nsig, parameters.prior_signal)
    return arr_vars, arr_post
