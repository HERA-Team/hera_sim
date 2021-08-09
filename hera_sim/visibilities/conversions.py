"""A number of mappings which may be useful for visibility simulators."""
import astropy_healpix as aph
from astropy_healpix import healpy
import numpy as np


def eq2top_m(ha, dec):
    """Calculate the equatorial to topocentric conversion matrix.

    Conversion at a given hour angle (ha) and declination (dec). Ripped
    straight from aipy.

    Parameters
    ----------
    ha : float
        Hour angle [rad].
    dec : float
        Declination [rad].

    Returns
    -------
    ndarray
        Coordinate transform matrix converting equatorial coordinates to
        topocentric coordinates. Shape=(3, 3).
    """
    sin_H, cos_H = np.sin(ha), np.cos(ha)
    sin_d, cos_d = np.sin(dec), np.cos(dec)
    zero = np.zeros_like(ha)

    rot_matrix = np.array(
        [
            [sin_H, cos_H, zero],
            [-sin_d * cos_H, sin_d * sin_H, cos_d],
            [cos_d * cos_H, -cos_d * sin_H, sin_d],
        ]
    )

    if len(rot_matrix.shape) == 3:
        rot_matrix = rot_matrix.transpose([2, 0, 1])

    return rot_matrix


def healpix_to_crd_eq(h, nest=False):
    """
    Determine equatorial co-ordinates of a healpix map's pixels.

    Parameters
    ----------
    h : array_like
        The HEALPix array. Shape=(12*N^2,) for integer N.
    nest : bool, optional
        Whether to use the NEST configuration for the HEALPix array.

    Returns
    -------
    ndarray
       The equatorial coordinates of each HEALPix pixel.
       Shape=(12*N^2, 3) for integer N.
    """
    assert h.ndim == 1, "h must be a 1D array."

    px = np.arange(len(h))
    return np.array(
        healpy.pix2vec(aph.npix_to_nside(len(h)), px, nest=nest), dtype=np.float32
    )


def lm_to_az_za(ell, m):
    """
    Convert l and m (on intervals -1, +1) to azimuth and zenith angle.

    Parameters
    ----------
    l, m : array_like
        Normalized angular coordinates on the interval (-1, +1).

    Returns
    -------
    az, za : array_like
        Corresponding azimuth and zenith angles (in radians).
    """
    lsqr = ell ** 2.0 + m ** 2.0
    n = np.where(lsqr < 1.0, np.sqrt(1.0 - lsqr), 0.0)

    az = -np.arctan2(m, ell)
    za = np.pi / 2.0 - np.arcsin(n)
    return az, za
