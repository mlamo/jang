from astropy.units import deg
import healpy as hp
import numpy as np
from typing import Iterable, Union
import unittest

from jang.neutrinos import (
    Acceptance,
    BackgroundFixed,
    BackgroundGaussian,
    BackgroundPoisson,
    Detector,
    EffectiveAreaBase,
    Sample,
    SuperDetector,
    ToyResult,
)
from jang.neutrinos import infer_uncertainties


class TestSample(unittest.TestCase):
    def setUp(self):
        self.s1 = Sample("sample")
        self.s1.set_observations(0, BackgroundFixed(0.5))
        self.s1.set_energy_range(1, 1000)

    def test_members(self):
        self.assertEqual(self.s1.shortname, "sample")
        self.assertEqual(self.s1.log10_energy_range, (0, 3))
        self.assertEqual(self.s1.log_energy_range, (np.log(1), np.log(1000)))
        self.assertEqual(self.s1.nobserved, 0)
        self.assertEqual(self.s1.background.nominal, 0.5)

    def test_background(self):
        print(BackgroundFixed(1))
        print(BackgroundGaussian(1, 0.1))
        print(BackgroundPoisson(10, 10))


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.dict_det1 = {
            "name": "Test",
            "nsamples": 1,
            "samples": {"names": ["smp"], "energyrange": [0, 1]},
            "earth_location": {"latitude": 0, "longitude": 0, "units": "deg"},
            "errors": {"acceptance": 0, "acceptance_corr": 0, "background": 0},
        }
        self.dict_det2 = {
            "name": "Test2",
            "nsamples": 4,
            "samples": {
                "names": ["sample1", "sample2", "sample3", "sample4"],
                "shortnames": ["s1", "s2", "s3", "s4"],
                "energyrange": [[0, 1], [0, 1], [0, 1], [0, 1]],
            },
            "earth_location": {"latitude": 10.0, "longitude": 5.0, "units": "deg"},
            "errors": {"acceptance": 0.40, "acceptance_corr": 1, "background": 0.40},
        }
        self.d1 = Detector(self.dict_det1)
        self.d2 = Detector(self.dict_det2)
        self.d2.set_observations(
            [0, 0, 0, 0],
            [
                BackgroundFixed(0.3),
                BackgroundGaussian(0.1, 0.01),
                BackgroundGaussian(0.1, 0.02),
                BackgroundPoisson(3, 10),
            ],
        )

    def test_acceptance(self):
        self.d1.set_acceptances([np.zeros(hp.nside2npix(4))], "x**-2", nside=8)
        self.assertEqual(self.d1.get_acceptances("x**-2")[0][0].evaluate(0), 0)
        self.assertEqual(self.d1.get_acceptances("x**-2")[0][0].evaluate(0, nside=2), 0)
        with self.assertRaises(RuntimeError):
            self.d2.set_acceptances([0], "x**-2")
        self.d2.set_acceptances([0, 0, 0, 0], "x**-2")
        self.assertTrue(self.d2.get_acceptances("x**-2")[0][0].is_zero())
        self.assertEqual(self.d2.get_acceptances("x**-2")[0][0].evaluate(0), 0)
        with self.assertRaises(RuntimeError):
            self.d2.get_acceptances("x**-3")
        self.d2.set_acceptances([np.zeros(hp.nside2npix(8)), 0, 0, np.zeros(hp.nside2npix(4))], "x**-2.5")
        with self.assertRaises(RuntimeError):
            self.d2.get_acceptances("x**-2.5")
        #
        with self.assertRaises(ValueError):
            Acceptance(np.ones(13))

    def test_members(self):
        self.assertEqual(self.d1.nsamples, 1)
        self.assertEqual(self.d2.nsamples, 4)
        self.assertEqual(self.d1.name, "Test")
        self.assertEqual(len(self.d2.samples), 4)
        self.assertEqual(self.d1.samples[0].shortname, "smp")

    def test_conv(self):
        ra = np.random.uniform(0, 360, size=10) * deg
        dec = np.random.uniform(-90, 90, size=10) * deg
        alt, az = self.d2.radec_to_altaz(ra, dec, 2450000)
        nra, ndec = self.d2.altaz_to_radec(alt, az, 2450000)
        for i in range(10):
            self.assertAlmostEqual(ra[i].to(deg).value, nra[i].to(deg).value)
            self.assertAlmostEqual(dec[i].to(deg).value, ndec[i].to(deg).value)
        alt = np.random.uniform(-90, 90, size=10) * deg
        az = np.random.uniform(0, 360, size=10) * deg
        ra, dec = self.d2.altaz_to_radec(alt, az, 2455000)
        nalt, naz = self.d2.radec_to_altaz(ra, dec, 2455000)
        for i in range(10):
            self.assertAlmostEqual(alt[i].to(deg).value, nalt[i].to(deg).value)
            self.assertAlmostEqual(az[i].to(deg).value, naz[i].to(deg).value)

    def test_exceptions(self):
        self.dict_det1["samples"]["energyrange"] = 0
        with self.assertRaises(RuntimeError):
            Detector(self.dict_det1)
        with self.assertRaises(TypeError):
            Detector(0)
        with self.assertRaises(RuntimeError):
            self.d1.set_observations([0, 0], [BackgroundFixed(0)])
        with self.assertRaises(RuntimeError):
            self.d1.set_observations([0], [BackgroundFixed(0), BackgroundFixed(0)])

    def test_toys(self):
        with self.assertRaises(RuntimeError):
            self.d1.prepare_toys(0)
        t = self.d2.prepare_toys(0)
        self.assertEqual(len(t), 1)
        t = self.d2.prepare_toys(500)
        self.assertEqual(len(t), 500)

    def test_superdetector(self):
        sd = SuperDetector("SD")
        sd.add_detector(self.d1)
        sd.add_detector(self.d2)
        with self.assertLogs(level="ERROR"):
            sd.add_detector(self.d2)
        with self.assertRaises(RuntimeError):
            sd.prepare_toys(500)
        self.d1.set_observations([0], [BackgroundFixed(0)])
        self.assertEqual(sd.nsamples, 5)
        self.assertEqual(len(list(sd.samples)), 5)
        sd.prepare_toys(0)
        sd.prepare_toys(500)
        #
        self.d1.set_acceptances([np.zeros(hp.nside2npix(4))], "x**-2")
        self.d2.set_acceptances(
            [
                np.zeros(hp.nside2npix(4)),
                np.zeros(hp.nside2npix(4)),
                np.zeros(hp.nside2npix(4)),
                np.zeros(hp.nside2npix(4)),
            ],
            "x**-2",
        )
        sd.get_acceptances("x**-2")
        with self.assertRaises(RuntimeError):
            sd.get_acceptances("x**-3")


class MyEffectiveArea(EffectiveAreaBase):
    def evaluate(self, energy: Union[float, Iterable]):
        return np.ones_like(energy)


class TestEffectiveArea(unittest.TestCase):
    def test_convert(self):
        self.dict_det = {
            "name": "Test",
            "nsamples": 1,
            "samples": {"names": ["smp"], "energyrange": [1, 100]},
            "earth_location": {"latitude": 0, "longitude": 0, "units": "deg"},
            "errors": {"acceptance": 0, "acceptance_corr": 0, "background": 0},
        }
        det = Detector(self.dict_det)
        aeff = MyEffectiveArea(det.samples[0])
        aeff.to_acceptance(det, 4, 2450000, "x**-2")
        with self.assertRaises(RuntimeError):
            aeff.to_acceptance(det, None, 2450000, "x**-2")


class TestOther(unittest.TestCase):
    def test_infer_uncertainties(self):
        self.assertTrue(
            np.array_equal(infer_uncertainties(0, 2), np.array([[0, 0], [0, 0]]))
        )
        self.assertTrue(
            np.array_equal(infer_uncertainties([1, 2], 2), np.array([[1, 0], [0, 4]]))
        )
        self.assertTrue(
            np.array_equal(
                infer_uncertainties([1, 2], 2, 1), np.array([[1, 2], [2, 4]])
            )
        )
        self.assertTrue(
            np.array_equal(
                infer_uncertainties([[1, 0], [0, 4]], 2), np.array([[1, 0], [0, 4]])
            )
        )
        self.assertIsNone(infer_uncertainties(None, 1))
        with self.assertRaises(RuntimeError):
            infer_uncertainties([0, 0, 0], 2)

    def test_toyresult(self):
        t = ToyResult([0, 1], [0.5, 1.5], [1, 1])
        self.assertEqual(
            t.__str__(),
            "ToyResult: n(observed)=[0 1], n(background)=[0.5 1.5], var(acceptance)=[1 1], events=None",
        )
