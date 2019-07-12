"""
A module for generating realistic HERA noise.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
import os
from .data import DATA_PATH

# XXX find a way to make this neater?
# XXX either way, needs updating for H2C
HERA_TSKY_VS_LST_NPZ = os.path.join(DATA_PATH, 'HERA_Tsky_vs_LST.npz')
npz = np.load(HERA_TSKY_VS_LST_NPZ) # Tsky vs fq/lst from Beardsley, beam v XXX, GSM v XXX
fqs = npz['freqs'] / 1e3
lsts = npz['lsts'] / 12. * np.pi
lsts = np.concatenate([lsts[-10:]-2*np.pi, lsts, lsts[:10]+2*np.pi])
HERA_Tsky_xx = npz['HERA_Tsky'][0].T
HERA_Tsky_yy = npz['HERA_Tsky'][1].T
HERA_Tsky_xx = np.concatenate([HERA_Tsky_xx[-10:], HERA_Tsky_xx, HERA_Tsky_xx[:10]])
HERA_Tsky_yy = np.concatenate([HERA_Tsky_yy[-10:], HERA_Tsky_yy, HERA_Tsky_yy[:10]])
HERA_Tsky_mdl = {}
HERA_Tsky_mdl['xx'] = RectBivariateSpline(lsts, fqs, HERA_Tsky_xx, kx=4, ky=4)
HERA_Tsky_mdl['yy'] = RectBivariateSpline(lsts, fqs, HERA_Tsky_yy, kx=4, ky=4)


HERA_BEAM_POLY_NPY = os.path.join(DATA_PATH, 'HERA_H2C_BEAM_POLY.npy')
HERA_BEAM_POLY = np.load(HERA_BEAM_POLY_NPY)

def get_omega_p(fqs, bm_poly=HERA_BEAM_POLY):
    """
    Convert polynomial coefficients to beam area.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            frequency array
        bm_poly (polynomial): default=HERA_BEAM_POLY
            a polynomial fit to sky-integral, solid-angle beam size of
            observation as a function of frequency.

    Returns:
        omega_p : (array-like): shape=(NFREQS,), steradian
            sky-integral of peak-normalized beam power
    """
    return np.polyval(bm_poly, fqs)


def resample_Tsky(lsts, fqs, Tsky_mdl=None, Tsky=180.0, mfreq=0.18, index=-2.5):
    """
    Re-sample a model of the sky temperature at particular freqs and lsts.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the observation to be generated.
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the observation to be generated.
        Tsky_mdl (callable): interpolation object, default=None
            if provided, an interpolation object that returns the sky temperature as a
            function of (lst, freqs).  Called as Tsky(lsts,fqs).
        Tsky (float): Kelvin
            if Tsky_mdl not provided, an isotropic sky temperature
            corresponding to the provided mfreq.
        mfreq (float): GHz
            the spectral frequency, in GHz, at which Tsky is specified
        index (float): default=-2.5
            the spectral index used to extrapolate Tsky to other frequencies

    Returns:
        tsky (array-like): shape=(NTIMES,NFREQS)
            sky temperature vs. time and frequency
    """
    if Tsky_mdl is not None:
        tsky = Tsky_mdl(lsts, fqs)  # support an interpolation object
    else:
        tsky = Tsky * (fqs / mfreq) ** index  # default to a scalar
        tsky = np.resize(tsky, (lsts.size, fqs.size))
    return tsky

