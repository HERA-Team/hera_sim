'''A module for generating realistic HERA noise.'''

import numpy as np
from scipy.interpolate import RectBivariateSpline
import aipy
import os
this_dir, this_filename = os.path.split(__file__)
HERA_TSKY_VS_LST_NPZ = os.path.join(this_dir, 'data','HERA_Tsky_vs_LST.npz')

npz = np.load(HERA_TSKY_VS_LST_NPZ) # Tsky vs fq/lst from Beardsley, beam v XXX, GSM v XXX
fqs = npz['freqs'] / 1e3
lsts = npz['lsts'] / 12. * np.pi
lsts = np.concatenate([lsts[-10:]-2*np.pi, lsts, lsts[:10]+2*np.pi])
HERA_Tsky_xx = npz['HERA_Tsky'][0].T
HERA_Tsky_yy = npz['HERA_Tsky'][1].T
HERA_Tsky_xx = np.concatenate([HERA_Tsky_xx[-10:], HERA_Tsky_xx, HERA_Tsky_xx[:10]])
HERA_Tsky_yy = np.concatenate([HERA_Tsky_yy[-10:], HERA_Tsky_yy, HERA_Tsky_yy[:10]])
HERA_Tsky_mdl = {}
HERA_Tsky_mdl['xx'] = RectBivariateSpline(lsts, fqs, HERA_Tsky_xx)
HERA_Tsky_mdl['yy'] = RectBivariateSpline(lsts, fqs, HERA_Tsky_yy)

# XXX replicated from hera_pspec.pspec
HERA_BEAM_POLY = np.array([  8.07774113e+08,  -1.02194430e+09,
    5.59397878e+08,  -1.72970713e+08, 3.30317669e+07,  -3.98798031e+06,
    2.97189690e+05,  -1.24980700e+04, 2.27220000e+02]) # See HERA Memo #27

# XXX replicated from hera_pspec.pspec
def jy2T(f, bm_poly=HERA_BEAM_POLY): 
    '''Return [mK] / [Jy] for a beam size vs. frequency (in GHz) defined by the
    polynomial bm_poly.  Default is a poly fit to the PAPER primary beam.'''
    lam = aipy.const.c / (f * 1e9)
    bm = np.polyval(bm_poly, f)
    return 1e-23 * lam**2 / (2 * aipy.const.k * bm) * 1e3

# XXX replicated from hera_pspec.oqe
def white_noise(size):
    sig = 1./np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j*np.random.normal(scale=sig, size=size)

def resample_Tsky(fqs, lsts, Tsky_mdl=None, Tsky=180., mfreq=0.18, index=-2.5):
    if Tsky_mdl is not None: tsky = Tsky_mdl(lsts,fqs) # support an interpolation object
    else:
        tsky = Tsky * (fqs / mfreq)**index# default to a scalar
        tsky = np.resize(tsky, (lsts.size, fqs.size))
    return tsky

def sky_noise_jy(Tsky, fqs, lsts, bm_poly=HERA_BEAM_POLY, inttime=10.7):
    B = np.average(fqs[1:] - fqs[:-1]) * 1e9 # bandwidth in Hz
    T2jy = 1e3 / jy2T(fqs, bm_poly=bm_poly) # K to Jy conversion
    T2jy.shape = (1,-1)
    Vnoise_jy = T2jy * Tsky / np.sqrt(inttime * B) # see noise_study.py for discussion of why no facItor of 2 here
    return white_noise(Vnoise_jy.shape) * Vnoise_jy # generate white noise with amplitude set by Vnoise_jy

