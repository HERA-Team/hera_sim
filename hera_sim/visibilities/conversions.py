"""
A number of mappings which may be useful for visibility simulators.
"""
import healpy
import numpy as np


def uvbeam_to_lm(uvbeam, freqs, n_pix_lm=63, **kwargs):
    """
    Convert a UVbeam to a uniform (l,m) grid

    Parameters
    ----------
    uvbeam : UVBeam object
        Beam to convert to an (l, m) grid.
    freqs : array_like
        Frequencies to interpolate to in [Hz]. Shape=(NFREQS,).
    n_npix_lm : int, optional
        Number of pixels for each side of the beam grid. Default is 63.

    Returns
    -------
    ndarray
        The beam map cube. Shape=(NFREQS, BEAM_PIX, BEAM_PIX).
    """

    L = np.linspace(-1, 1, n_pix_lm, dtype=np.float32)
    L, m = np.meshgrid(L, L)
    L = L.flatten()
    m = m.flatten()

    lsqr = L ** 2 + m ** 2
    n = np.where(lsqr < 1, np.sqrt(1 - lsqr), 0)

    az = -np.arctan2(m, L)
    za = np.pi/2 - np.arcsin(n)

    efield_beam = uvbeam.interp(az, za, freqs, **kwargs)[0]
    efieldXX = efield_beam[0, 0, 1]

    # Get the relevant indices of res
    bm = np.zeros((len(freqs), len(L)))

    bm = efieldXX

    if np.max(bm) > 0:
        bm /= np.max(bm)

    return bm.reshape((len(freqs), n_pix_lm, n_pix_lm))


def eq2top_m(ha, dec):
    """
    Calculates the equatorial to topocentric conversion matrix.
    
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

    map = np.array([[sin_H, cos_H, zero],
                    [-sin_d * cos_H, sin_d * sin_H, cos_d],
                    [cos_d * cos_H, -cos_d * sin_H, sin_d]])

    if len(map.shape) == 3:
        map = map.transpose([2, 0, 1])

    return map


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
    crd_eq = np.array(healpy.pix2vec(healpy.get_nside(h), px, nest=nest),
                      dtype=np.float32)
    return crd_eq


def lm_to_az_za(l, m):
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
    lsqr = l**2. + m**2.
    n = np.where(lsqr < 1., np.sqrt(1. - lsqr), 0.)
    
    az = -np.arctan2(m, l)
    za = np.pi/2. - np.arcsin(n)
    return az, za
    
