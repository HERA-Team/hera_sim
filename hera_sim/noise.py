"""
A module for generating realistic HERA noise.
"""
from __future__ import absolute_import
import numpy as np
import aipy
import glob
from .hera_season import get_season, SEASONS

def bm_poly_to_omega_p(fqs, bm_poly=None):
    """
    Convert a set of beam polynomial coefficients to a beam size.
    This method is maintained for backwards compatability.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            frequency array
        bm_poly (polynomial): defaults to H1C beam polynomial
            polynomial fit to sky-integral of peak-normalized beam-power 
            pattern as a function of frequency

    Returns:
        omega_p (array-like): shape=(NFREQS,), steradian
            sky-integral of peak-normalized beam power
    """
    if bm_poly is None:
        # default to HERA H1C beam polynomial
        seas = get_season('h1c')
        DATA_PATH = seas.data.DATA_PATH
        beamfile = glob.glob('/{}*BEAM_POLY.npy'.format(DATA_PATH))[0]
        HERA_BEAM_POLY = np.load(beamfile)
    else:
        HERA_BEAM_POLY = bm_poly
    return np.polyval(HERA_BEAM_POLY)

def jy2T(fqs, omega_p=None, season=None):
    """
    Return [mK] / [Jy] for a beam size vs. frequency.

    Arg:
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the observation to be generated.
        omega_p (array-like): shape=(NFREQS,) steradians
            Sky-integral of beam power. If nothing is passed to omega_p, then
            the value appropriate for the H1C observing season is used.

    Returns:
        jy_to_mK (array-like): shape=(NFREQS,)
            a frequency-dependent scalar converting Jy to mK for the provided
            beam area
    """
    if omega_p is None:
        seas = get_season(season)
        omega_p = seas.noise.get_omega_p(fqs)
    assert len(omega_p)==len(fqs)
    lam = aipy.const.c / (fqs * 1e9)
    return 1e-23 * lam ** 2 / (2 * aipy.const.k * omega_p) * 1e3 # XXX make Kelvin in future


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

# XXX should inttime's default value be left at 10.7? this seems h1c specific.
def sky_noise_jy(lsts, fqs, Tsky=None, omega_p=None, season=None, B=None, inttime=10.7):
    """
    Generate Gaussian noise (in Jy units) corresponding to a sky temperature
    model integrated for the specified integration time and bandwidth.

    Args:
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the observation
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the observation
        Tsky (array-like): shape=(NTIMES,NFREQS), K
            the sky temperature at each time/frequency observation
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
    seas = get_season(season)
    if Tsky is None:
        Tsky = seas.noise.resample_Tsky(lsts, fqs)

    if B is None:
        B = np.average(fqs[1:] - fqs[:-1])
    B_Hz = B * 1e9 # bandwidth in Hz

    if inttime is None:
        inttime = (lsts[1] - lsts[0]) / (2 * np.pi) * aipy.const.sidereal_day
    
    T2jy = 1.0 / jy2T(fqs, omega_p, season)  # K to Jy conversion
    T2jy.shape = (1, -1)
    Vnoise_jy = T2jy * Tsky / np.sqrt(inttime * B_Hz) # see noise_study.py for discussion of why no factor of 2 here
    
    return white_noise(Vnoise_jy.shape) * Vnoise_jy

# XXX should inttime's default value be left at 10.7 s? this seems h1c specific.
def thermal_noise(lsts, fqs, Tsky_mdl=None, omega_p=None, season=None, Trx=0, inttime=10.7, **kwargs):
    """
    Create thermal noise visibilities.

    Args:
        lsts (1d array): times, in rad.
        fqs (1d array): frequencies, in GHz.
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
    seas = get_season(season)

    Tsky = seas.noise.resample_Tsky(lsts, fqs, Tsky_mdl=Tsky_mdl, **kwargs)
    Tsky += Trx
    return sky_noise_jy(lsts, fqs, Tsky, omega_p, season, inttime=inttime)
