"""Computation of significance."""

import logging
import numpy as np
from scipy.stats import poisson

from jang.analysis import Analysis
from jang.gw import GW
from jang.neutrinos import Detector, ToyResult
from jang.parameters import Parameters
import jang.stats
import jang.stats.likelihoods as lkl
import jang.stats.priors as prior


def compute_prob_null_hypothesis(detector: Detector, gw: GW, parameters: Parameters, bkg_events: list = None):

    if parameters.likelihood_method == "poisson":
        B01 = compute_bayes_factor_poisson(detector, gw, parameters)
    elif parameters.likelihood_method == "pointsource":
        B01 = compute_bayes_factor_pointsource(detector, gw, parameters, bkg_events)

    p0 = B01 / (1 + B01)
    logging.getLogger("jang").info(
        "[Significance] %s, %s, %s, P(H0 | data) = %.3g %%",
        gw.name,
        detector.name,
        parameters.spectrum,
        100 * p0,
    )
    return p0


def compute_bayes_factor_poisson(detector: Detector, gw: GW, parameters: Parameters):

    ana = Analysis(gw=gw, detector=detector, parameters=parameters)
    ana.prepare_toys()

    flux = jang.stats.PosteriorVariable("flux", parameters.range_flux[0:2], parameters.range_flux[2], log=True).array
    var = {0: flux[:-1]}

    B0, B1 = 0, 0
    B0_N, B1_N = 0, 0

    for toy in ana.toys:
        phi_to_nsig = ana.phi_to_nsig(toy)
        B1 += np.sum(
            lkl.poisson_several_samples(toy[1].nobserved, toy[1].nbackground, phi_to_nsig, var)
            * prior.signal_parameter(var[0], toy[1].nbackground, phi_to_nsig, parameters.prior_signal)
            * np.diff(flux)
        )
        B0 += lkl.poisson_several_samples(toy[1].nobserved, toy[1].nbackground, phi_to_nsig, {0: 0.0})

        toy_observedH0 = [poisson.rvs(b) for b in toy[1].nbackground]
        B1_N += np.sum(
            lkl.poisson_several_samples(toy_observedH0, toy[1].nbackground, phi_to_nsig, var)
            * prior.signal_parameter(var[0], toy[1].nbackground, phi_to_nsig, parameters.prior_signal)
            * np.diff(flux)
        )
        B0_N += lkl.poisson_several_samples(toy_observedH0, toy[1].nbackground, phi_to_nsig, {0: 0.0})

    return B0 / B1 * B1_N / B0_N


def compute_bayes_factor_pointsource(detector: Detector, gw: GW, parameters: Parameters, bkg_events: list):

    ana = Analysis(gw=gw, detector=detector, parameters=parameters)
    ana.prepare_toys()

    flux = jang.stats.PosteriorVariable("flux", parameters.range_flux[0:2], parameters.range_flux[2], log=True).array
    var = {0: flux[:-1]}

    B0, B1 = 0, 0
    B0_N, B1_N = 0, 0

    for toy in ana.toys:
        phi_to_nsig = ana.phi_to_nsig(toy)
        B1 += np.sum(
            lkl.pointsource_several_samples(
                toy[1].nobserved,
                toy[1].nbackground,
                toy[1].events,
                phi_to_nsig,
                detector.samples,
                toy[0].ra,
                toy[0].dec,
                var,
            )
            * prior.signal_parameter(var[0], toy[1].nbackground, phi_to_nsig, parameters.prior_signal)
            * np.diff(flux)
        )
        B0 += lkl.poisson_several_samples(toy[1].nobserved, toy[1].nbackground, phi_to_nsig, {0: 0.0})

        toy_H0 = ToyResult([poisson.rvs(b) for b in toy[1].nbackground], toy[1].nbackground, toy[1].var_acceptance)
        events = []
        for i, nobs in enumerate(toy_H0.nobserved):
            events.append(np.random.choice(bkg_events[i], size=nobs, replace=False))
        toy_H0.events = events

        B1_N += np.sum(
            lkl.pointsource_several_samples(
                toy_H0.nobserved,
                toy_H0.nbackground,
                toy_H0.events,
                phi_to_nsig,
                detector.samples,
                toy[0].ra,
                toy[0].dec,
                var,
            )
            * prior.signal_parameter(var[0], toy_H0.nbackground, phi_to_nsig, parameters.prior_signal)
            * np.diff(flux)
        )
        B0_N += lkl.poisson_several_samples(toy_H0.nobserved, toy_H0.nbackground, phi_to_nsig, {0: 0.0})

    return B0 / B1 * B1_N / B0_N
