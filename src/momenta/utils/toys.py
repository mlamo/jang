"""
Copyright (C) 2026  Christoph Raab

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy.units import Quantity
from scipy.stats import vonmises, vonmises_fisher

from momenta.utils.conversions import to_radians

def pointsource_spatial(ra: float|Angle|Quantity=None, dec: float|Angle|Quantity=None,
                        coords: SkyCoord=None,
                        err: float|Angle|Quantity=0.,
                        size: int=10000) -> tuple[np.ndarray, np.ndarray]:
    """Sample RA, Dec in degrees of point source coordinates with uncertainty.

    Args:
        ra (float, Angle of Quantity, optional): Source right ascension in degree. Optional if coords are given.
        dec (float, Angle of Quantity, optional): Source declination in degree. Optional if coords are given.
        coords (SkyCoord, optional): Source coordinates. Optional if ra and dec are given.
        err (float, Angle of Quantity, optional): 1-sigma equivalent width of the distribution. Defaults to 0..
        size (int, optional): Number of toy coordinates to return. Defaults to 10000.

    Raises:
        TypeError: if neither coords nor ra and dec are given.
        ValueError: if err < 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: Toy sample right ascensions, declinations in degrees.
    """
    # Convert all into floats in rad for SciPy
    ra, dec, err = to_radians(ra), to_radians(dec), to_radians(err)
    if coords is None:
        if (ra is None) or (dec is None):
            raise TypeError("Supply at least (coords) or (ra, dec) arguments")
        coords = SkyCoord(ra=ra, dec=dec, unit="rad", frame="icrs", representation_type="spherical")
    if err < 0:
        raise ValueError(f"{err=}<0")
    # zero width, all are identical to centroid
    if err == 0:
        return np.full(size, coords.ra.deg), np.full(size, coords.dec.deg)
    # else sample from vMF distribution
    kappa = 1./err**2 # this is an approximation/convention, does not preserve containment for large err
    vmf = vonmises_fisher(mu=coords.cartesian.xyz, kappa=kappa)
    xyz = vmf.rvs(size=size)
    # convert back to coordinates, ignore any floating point error changing the normalization
    lonlat = SkyCoord(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], representation_type="cartesian", frame="icrs").spherical
    return lonlat.lon.deg, lonlat.lat.deg