#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import healpy as hp
import numpy as np
import tempfile
import unittest

import jang.conversions
import jang.gw
import jang.parameters
import jang.stats
import jang.stats.priors


class TestJetModels(unittest.TestCase):
    def test_isotropic(self):
        jet = jang.conversions.JetIsotropic()
        jet.etot_to_eiso(0)
        print(jet, jet.str_filename)

    def test_vonmises(self):
        jet = jang.conversions.JetVonMises(np.inf)
        jet.etot_to_eiso(0)
        jet.etot_to_eiso(0.5)
        print(jet, jet.str_filename)
        jet = jang.conversions.JetVonMises(0.1)
        jet.etot_to_eiso(0)
        jet.etot_to_eiso(0.5)
        print(jet, jet.str_filename)
        jet = jang.conversions.JetVonMises(0.1, with_counter=True)
        jet.etot_to_eiso(0)
        jet.etot_to_eiso(0.5)
        print(jet, jet.str_filename)

    def test_rectangular(self):
        jet = jang.conversions.JetRectangular(np.inf)
        jet.etot_to_eiso(0)
        jet.etot_to_eiso(0.5)
        print(jet, jet.str_filename)
        jet = jang.conversions.JetRectangular(0.1)
        jet.etot_to_eiso(0)
        jet.etot_to_eiso(0.5)
        print(jet, jet.str_filename)
        jet = jang.conversions.JetRectangular(0.1, with_counter=True)
        jet.etot_to_eiso(0)
        jet.etot_to_eiso(0.5)
        print(jet, jet.str_filename)

    def test_list(self):
        jang.conversions.list_jet_models()


class TestConversions(unittest.TestCase):
    def test_fluxenergy(self):
        jang.conversions.phi_to_eiso((1, 100), "x**-2", 1)
        jang.conversions.eiso_to_phi((1, 100), "x**-2", 1)
        jang.conversions.etot_to_eiso(0.3, jang.conversions.JetIsotropic())
        jang.conversions.fnu_to_etot(1)


class TestGW(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        config_str = """
            analysis:
              nside: 8
              apply_det_systematics: 0
              ntoys_det_systematics: 0
              search_region: region_90_excludezero
              likelihood: poisson
              prior_signal: flat

            range:
              log10_flux: [-5, 5, 1000]
              log10_etot: [48, 62, 1400]
              log10_fnu: [-5, 10, 1500]
              neutrino_energy_GeV: [0.1, 1e8]
        """
        self.config_file = f"{self.tmpdir}/config.yaml"
        with open(self.config_file, "w") as f:
            f.write(config_str)
        self.pars = jang.parameters.Parameters(self.config_file)
        self.dbgw = jang.gw.Database(
            "examples/input_files/gw_catalogs/database_example.csv"
        )
        self.gw = self.dbgw.find_gw("GW190412", self.pars)
        self.tmpdir = tempfile.mkdtemp()

    def test_skymap(self):
        self.assertTrue(np.all(self.gw.fits.get_signal_region(8, None) == np.arange(hp.nside2npix(8))))
        self.assertTrue(np.all(self.gw.fits.get_signal_region(8, 0.90) == [163, 131, 164]))
        self.assertTrue(np.all(np.isclose(self.gw.fits.get_ra_dec_bestfit(8), (3.730641276137879, 0.6228265854120028))))

    def test_database(self):
        emptydb = jang.gw.Database()
        emptydb.add_entry("ev", "", "")
        with self.assertRaises(RuntimeError):
            emptydb.save()
        emptydb = jang.gw.Database(f"{self.tmpdir}/db.csv")
        emptydb.add_entry("ev", "", "")
        emptydb.save(f"{self.tmpdir}/db.csv")
        emptydb.save()
        #
        with self.assertRaises(RuntimeError):
            self.dbgw.find_gw("missing_ev", self.pars)
        self.dbgw.list_all()
        self.dbgw.list(self.pars, "BBH", 0, 1000)
        self.dbgw.list(self.pars, "BNS", 1000, 0)
        self.dbgw.add_entry("ev", "", "")
        self.dbgw.save(f"{self.tmpdir}/db.csv")


class TestParameters(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        config_str = """
            analysis:
              nside: -1
              apply_det_systematics: 0
              ntoys_det_systematics: 0
              search_region: bestfit
              likelihood: poisson
              prior_signal: flat

            range:
              log10_flux: [-5, 5, 1000]
              log10_etot: [48, 62, 1400]
              log10_fnu: [-5, 10, 1500]
              neutrino_energy_GeV: [0.1, 1e8]
        """
        self.config_file = f"{self.tmpdir}/config.yaml"
        with open(self.config_file, "w") as f:
            f.write(config_str)

    def test_pars(self):
        pars = jang.parameters.Parameters(self.config_file)
        pars.set_models("x**-2", jang.conversions.JetIsotropic())
        pars.str_filename
        pars.search_region = "bestfit"
        self.assertEqual(pars.get_searchregion_gwfraction(), 0)
        self.assertTrue(pars.get_searchregion_iszeroincluded())
        pars.search_region = "region_90"
        self.assertEqual(pars.get_searchregion_gwfraction(), 0.90)
        self.assertTrue(pars.get_searchregion_iszeroincluded())
        pars.search_region = "region_90_excludezero"
        self.assertEqual(pars.get_searchregion_gwfraction(), 0.90)
        self.assertFalse(pars.get_searchregion_iszeroincluded())
        pars.search_region = ""
        self.assertEqual(pars.get_searchregion_gwfraction(), None)


class TestStatTools(unittest.TestCase):
    def test_lkl(self):
        jang.stats.compute_upperlimit_from_x_y(np.zeros(10), np.zeros(10))
        self.assertEqual(
            jang.stats.compute_upperlimit_from_x_y(np.arange(10), np.arange(10)),
            np.inf,
        )
        jang.stats.normalize(np.zeros(10), np.zeros(10))

    def test_priors(self):
        var, bkg, conv = np.arange(100), np.ones(10), np.ones(10)
        jang.stats.priors.signal_parameter(var, bkg, conv, "flat")
        jang.stats.priors.signal_parameter(var, bkg, conv, "jeffrey")
        with self.assertRaises(RuntimeError):
            jang.stats.priors.signal_parameter(var, bkg, conv, "missing")
