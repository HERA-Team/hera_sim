"""
A module for generating realistic HERA noise.
"""

import os

import aipy
import numpy as np
from scipy.interpolate import RectBivariateSpline

this_dir, this_filename = os.path.split(__file__)
HERA_TSKY_VS_LST_NPZ = os.path.join(this_dir, "data", "HERA_Tsky_vs_LST.npz")

npz = np.load(
    HERA_TSKY_VS_LST_NPZ
)  # Tsky vs fq/lst from Beardsley, beam v XXX, GSM v XXX
fqs = npz["freqs"] / 1e3
lsts = npz["lsts"] / 12.0 * np.pi
lsts = np.concatenate([lsts[-10:] - 2 * np.pi, lsts, lsts[:10] + 2 * np.pi])
HERA_Tsky_xx = npz["HERA_Tsky"][0].T
HERA_Tsky_yy = npz["HERA_Tsky"][1].T
HERA_Tsky_xx = np.concatenate([HERA_Tsky_xx[-10:], HERA_Tsky_xx, HERA_Tsky_xx[:10]])
HERA_Tsky_yy = np.concatenate([HERA_Tsky_yy[-10:], HERA_Tsky_yy, HERA_Tsky_yy[:10]])
HERA_Tsky_mdl = {}
HERA_Tsky_mdl["xx"] = RectBivariateSpline(lsts, fqs, HERA_Tsky_xx)
HERA_Tsky_mdl["yy"] = RectBivariateSpline(lsts, fqs, HERA_Tsky_yy)

# XXX replicated from hera_pspec.pspec
HERA_BEAM_POLY = np.array([8.07774113e+08, -1.02194430e+09,
                           5.59397878e+08, -1.72970713e+08, 3.30317669e+07, -3.98798031e+06,
                           2.97189690e+05, -1.24980700e+04, 2.27220000e+02])  # See HERA Memo #27


# XXX replicated from hera_pspec.pspec
def jy2T(f, bm_poly=HERA_BEAM_POLY):
    """
    Conversion factor from Jy to mK.

    Args:
        f (float or ndarray): frequencies of observation [GHz]
        bm_poly (ndarray): array defining a numpy polynomial, which defines the beam width as a function of
            frequency. Defau;t is a poly fit to the PAPER primary beam.

    Returns:
        float or array: conversion factor [mK]/[Jy] for given frequencies.
    """
    lam = aipy.const.c / (f * 1e9)
    bm = np.polyval(bm_poly, f)
    return 1e-23 * lam ** 2 / (2 * aipy.const.k * bm) * 1e3


# XXX replicated from hera_pspec.oqe
def white_noise(size=1):
    """
    Produce complex white noise with a variance of unity.

    Args:
        size (int or tuple, optional): shape of output samples.

    Returns:
        complex ndarray: random white noise samples

    """
    sig = 1.0 / np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j * np.random.normal(
        scale=sig, size=size
    )


def resample_Tsky(fqs, lsts, Tsky_mdl=None, Tsky=180.0, mfreq=0.18, index=-2.5):
    """
    Evaluate sky temperature as a function of LST and frequency.

    Args:
        fqs (ndarray): frequencies [GHz]
        lsts (ndarray): LSTs (radians)
        Tsky_mdl (callable, optional): a model for Tsky as a function of LST and frequency.
            If a callable is given, it must have the signature ``Tsky_mdl(lsts, fqs)`` and return a 2D
            array of shape ``(len(lsts), len(fqs))``. If not given, a spatially-uniform temperature will be used.
        Tsky (float, optional): temperature of the sky [K] at `mfreq`, only used if `Tsky_mdl` not given.
        mfreq (float, optional): reference frequency [GHz], only used if `Tsky_mdl` not given.
        index (float, optional): spectral index of the temperature, only used if `Tsky_mdl` not given.

    Returns:
        2D ndarray: temperature of the sky at each (LST, fq) pair [K].
    """
    if Tsky_mdl is not None:
        tsky = Tsky_mdl(lsts, fqs)  # support an interpolation object
    else:
        tsky = Tsky * (fqs / mfreq) ** index  # default to a scalar
        tsky = np.resize(tsky, (lsts.size, fqs.size))
    return tsky


def sky_noise_jy(Tsky, fqs, lsts, bm_poly=HERA_BEAM_POLY, inttime=10.7):
    """
    Produce sky noise for a range of frequencies and LSTs.

    Args:
        Tsky (2D ndarray): the temperature of the sky. This is expected to be an array of shape ``(n_lst, n_freq)``,
            however the routine will run without exception if it is a float or an nD array with last dimension of
            size ``n_freq``.
        fqs (ndarray): frequencies of observation [GHz].
        lsts: unneeded.
        bm_poly (ndarray): array defining a numpy polynomial, which defines the beam width as a function of
            frequency. Defau;t is a poly fit to the PAPER primary beam.
        inttime (float): integration time [sec].

def thermal_noise(fqs, lsts, Tsky_mdl=None, Trx=0, bm_poly=HERA_BEAM_POLY, inttime=10.7, **kwargs):
    """
    Create thermal noise visibilities.

    Args:
        fqs (1d array): frequencies, in GHz.
        lsts (1d array): times, in rad.
        Tsky_mdl (callable, optional): a callable model, with signature ``Tsky_mdl(lsts, fqs)``, which returns a 2D
            array of global beam-averaged sky temperatures (in K) as a function of LST and frequency.
        Trx (float, optional): receiver temperature, in K.
        bm_poly (np.poly1d, optional): a polynomial defining the frequency-dependence of the beam size.
        inttime (float, optional): the integration time, in sec.
        **kwargs: passed to :func:`resample_Tsky`.

    Returns:
        2d array size(lsts, fqs): the thermal visibilities [Jy].

    """
    Tsky = resample_Tsky(fqs, lsts, Tsky_mdl=Tsky_mdl, **kwargs)
    Tsky += Trx
    return sky_noise_jy(Tsky, fqs, lsts, bm_poly=bm_poly, inttime=inttime)
