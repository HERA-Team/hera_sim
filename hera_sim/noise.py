"""
A module for generating realistic HERA noise.
"""
from __future__ import absolute_import
from . import h1c, h2c
VERSIONS = {'h1c':h1c, 'h2c':h2c}
import numpy as np
from scipy.interpolate import RectBivariateSpline
import aipy
import os
from .data import DATA_PATH

DEFAULT = 'h1c'

def set_default(season):
    assert season in VERSIONS.keys()
    global DEFAULT
    DEFAULT = season

def get_version(version=None):
    if version is None:
        version = DEFAULT
    return VERSIONS[version]

def jy2T(fqs, version=None):
    """
    Return [mK] / [Jy] conversion for given frequencies
    and HERA version.
    """
    ver = get_version(version)
    
    omega_p = ver.noise.get_omega_p(fqs)
    return _jy2T(fqs, omega_p)

def _jy2T(fqs, omega_p):
    """
    Return [mK] / [Jy] for a beam size vs. frequency.

    Arg:
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the observation to be generated.
        omega_p (array-like): shape=(NFREQS,) steradians
            Sky-integral of beam power.

    Returns:
        jy_to_mK (array-like): shape=(NFREQS,)
            a frequency-dependent scalar converting Jy to mK for the provided
            beam size.'''
    """
    lam = aipy.const.c / (fqs * 1e9)
    return 1e-23 * lam ** 2 / (2 * aipy.const.k * omega_p) # Kelvin


def white_noise(size=1):
    """
    Produce complex Gaussian white noise with a variance of unity.

    Args:
        size (int or tuple, optional):
            shape of output samples.

    Returns:
        noise (ndarray): shape=size
            random white noise realization
    """
    sig = 1.0 / np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j * np.random.normal(
        scale=sig, size=size
    )

def sky_noise_jy(lsts, fqs, Tsky=None, version=None, B=None, inttime=10.7):
    """
    Generate Gaussian noise (in Jy units) corresponding to a sky temperature
    model integrated for the specified integration time and bandwidth.

    Args:
        Tsky (array-like): shape=(NTIMES,NFREQS), K
            the sky temperature at each time/frequency observation
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the observation
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the observation
        omega_p (array-like): shape=(NFREQS,) steradians
            Sky-integral of beam power.
        B (float): default=None, GHz
            the channel width used to integrate noise.  If not provided,
            defaults to the delta between fqs,
        inttime (float): default=10.7, seconds
            the time used to integrate noise.  If not provided, defaults
            to delta between lsts.

    Returns:
        noise (array-like): shape=(NTIMES,NFREQS)
            complex Gaussian noise vs. time and frequency
    """
    ver = get_version(version)
    if Tsky is None:
        Tsky = ver.noise.resample_Tsky(lsts, fqs)

    if B is None:
        B = np.average(fqs[1:] - fqs[:-1])
    B_Hz = B * 1e9 # bandwidth in Hz

    if inttime is None:
        inttime = (lsts[1] - lsts[0]) / (2 * np.pi) * aipy.const.sidereal_day
    
    T2jy = 1.0 / jy2T(fqs, version)  # K to Jy conversion
    T2jy.shape = (1, -1)
    Vnoise_jy = T2jy * Tsky / np.sqrt(inttime * B_Hz) # see noise_study.py for discussion of why no factor of 2 here
    
    return white_noise(Vnoise_jy.shape) * Vnoise_jy


def thermal_noise(fqs, lsts, Tsky_mdl=None, Trx=0, omega_p=None, inttime=10.7, **kwargs):
    """
    Create thermal noise visibilities.

    Args:
        fqs (1d array): frequencies, in GHz.
        lsts (1d array): times, in rad.
        Tsky_mdl (callable, optional): a callable model, with signature ``Tsky_mdl(lsts, fqs)``, which returns a 2D
            array of global beam-averaged sky temperatures (in K) as a function of LST and frequency.
        Trx (float, optional): receiver temperature, in K.
        omega_p (array-like): shape=(NFREQS,) steradians
            Sky-integral of beam power. Default is to use noise.HERA_BEAM_POLY to create omega_p.
        inttime (float, optional): the integration time, in sec.
        **kwargs: passed to :func:`resample_Tsky`.

    Returns:
        2d array size(lsts, fqs): the thermal visibilities [Jy].
    """
    if omega_p is None:
        omega_p = bm_poly_to_omega_p(fqs)
    Tsky = resample_Tsky(fqs, lsts, Tsky_mdl=Tsky_mdl, **kwargs)
    Tsky += Trx
    return sky_noise_jy(Tsky, fqs, lsts, omega_p, inttime=inttime)
