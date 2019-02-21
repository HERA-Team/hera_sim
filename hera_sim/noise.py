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

HERA_BEAM_POLY = np.array([8.07774113e+08, -1.02194430e+09,
                           5.59397878e+08, -1.72970713e+08, 3.30317669e+07, -3.98798031e+06,
                           2.97189690e+05, -1.24980700e+04, 2.27220000e+02])  # See HERA Memo #27


def jy2T(fq, bm_poly=HERA_BEAM_POLY):
    """
    Return [mK] / [Jy] for a beam size vs. frequency.

    Arg:
        fqs: array-like, shape=(NFREQS,), GHz
            the spectral frequencies of the observation to be generated.
        bm_poly: polynomial, default=HERA_BEAM_POLY
            a polynomial fit to the solid-angle beam size of the observation
            as a function of frequency.  Used to convert temperatures to Jy.
    Returns:
        jy_to_mK: array-like, shape=(NFREQS,)
            a frequency-dependent scalar converting Jy to mK for the provided
            beam size.'''
    """
    lam = aipy.const.c / (fq * 1e9)
    bm = np.polyval(bm_poly, fq)
    return 1e-23 * lam ** 2 / (2 * aipy.const.k * bm) * 1e3 # XXX make Kelvin in future


def white_noise(size=1):
    """
    Produce complex Gaussian white noise with a variance of unity.

    Args:
        size (int or tuple, optional):
            shape of output samples.

    Returns:
        noise: ndarray, shape=size
            random white noise realization

    """
    sig = 1.0 / np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j * np.random.normal(
        scale=sig, size=size
    )


# XXX reverse fqs and lsts in this function?
def resample_Tsky(fqs, lsts, Tsky_mdl=None, Tsky=180.0, mfreq=0.18, index=-2.5):
    """
    Re-sample a model of the sky temperature at particular freqs and lsts.

    Args:
        fqs: array-like, shape=(NFREQS,), GHz
            the spectral frequencies of the observation to be generated.
        lsts: array-like, shape=(NTIMES,), radians
            local sidereal times of the observation to be generated.
        Tsky_mdl: interpolation object, default=None
            if provided, an interpolation object that returns the sky temperature as a
            function of (lst, freqs).  Called as Tsky(lsts,fqs).
        Tsky: float, Kelvin
            if Tsky_mdl not provided, an isotropic sky temperature
            corresponding to the provided mfreq.
        mfreq: float, GHz
            the spectral frequency, in GHz, at which Tsky is specified
        index: float
            the spectral index used to extrapolate Tsky to other frequencies
    Returns:
        tsky: array-like, shape=(NTIMES,NFREQS)
            sky temperature vs. time and frequency
    """
    if Tsky_mdl is not None:
        tsky = Tsky_mdl(lsts, fqs)  # support an interpolation object
    else:
        tsky = Tsky * (fqs / mfreq) ** index  # default to a scalar
        tsky = np.resize(tsky, (lsts.size, fqs.size))
    return tsky


# XXX make inttime default=None
# XXX reorder fqs/lsts
def sky_noise_jy(Tsky, fqs, lsts, bm_poly=HERA_BEAM_POLY, B=None, inttime=10.7):
    """
    Generate Gaussian noise (in Jy units) corresponding to a sky temperature
    model integrated for the specified integration time and bandwidth.

    Args:
        Tsky: array-like, shape=(NTIMES,NFREQS), K
            the sky temperature at each time/frequency observation
        fqs: array-like, shape=(NFREQS,), GHz
            the spectral frequencies of the observation
        lsts: array-like, shape=(NTIMES,), radians
            local sidereal times of the observation
        bm_poly: polynomial, default=HERA_BEAM_POLY
            a polynomial fit to the solid-angle beam size of the observation
            as a function of frequency.  Used to convert temperatures to Jy.
        B: float, default=None, GHz
            the bandwidth used to integrate noise.  If not provided,
            defaults to the delta between fqs,
        inttime: float, default=10.7, seconds
            the time used to integrate noise.  If not provided, defaults
            to delta between lsts.
    Returns:
        noise: array-like, shape=(NTIMES,NFREQS)
            complex Gaussian noise vs. time and frequency
    """
    if B is None:
        B = np.average(fqs[1:] - fqs[:-1])
    B_Hz = B * 1e9 # bandwidth in Hz
    if inttime is None:
        inttime = (lsts[1] - lsts[0]) / (2 * np.pi) * aipy.const.sidereal_day
    # XXX fix below when jy2T changed to Jy/K
    T2jy = 1e3 / jy2T(fqs, bm_poly=bm_poly)  # K to Jy conversion
    T2jy.shape = (1, -1)
    Vnoise_jy = T2jy * Tsky / np.sqrt(inttime * B) # see noise_study.py for discussion of why no factor of 2 here
    return white_noise(Vnoise_jy.shape) * Vnoise_jy


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
