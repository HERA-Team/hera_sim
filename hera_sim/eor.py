'''A module for generating a rough eor-like signal.'''

import numpy as np
from scipy.interpolate import RectBivariateSpline
import aipy
from . import noise
from . import utils


def noiselike_eor(lsts, fqs, bl_len_ns, eor_amp=1e-5, spec_tilt=0.0,
                  fr_width=None, min_dly=0, max_dly=3000):
    """
    Generate a noise-like EoR signal that tracks the sky.
    Modeled after foregrounds.diffuse_foreground().

    Args:
        lsts : ndarray with LSTs [radians]
        fqs : ndarray with frequencies [GHz]
        bl_len_ns : float, East-West baseline length [nanosec]
        eor_amp : float, amplitude of EoR signal [arbitrary]
        spec_tilt : float, spectral slope of EoR spectral amplitude
            as a function of delay in microseconds
        fr_width : float, width of FR filter in 2pi lambda / sec
        min_dly : float, minimum |delay| in nanosec of EoR signal
        max_dly : float, maximum |delay| in nanosec of EoR signal

    Returns: 
        vis : 2D ndarray holding simulated complex visibility
    """
    # get fringe rate and generate an LST grid
    fr_max = np.max(utils.calc_max_fringe_rate(fqs, bl_len_ns))
    dt = 0.5/fr_max # over-resolve by factor of 2
    ntimes = int(np.around(aipy.const.sidereal_day / dt))
    lst_grid = np.linspace(0, 2*np.pi, ntimes, endpoint=False)

    # generate white noise
    vis = noise.white_noise((ntimes, len(fqs))) * eor_amp

    # Fringe-Rate Filter given baseline
    vis = utils.rough_fringe_filter(vis, lst_grid, fqs, bl_len_ns, fr_width=fr_width)

    # interpolate at fed LSTs
    mdl_real = RectBivariateSpline(lst_grid, fqs, vis.real)
    mdl_imag = RectBivariateSpline(lst_grid, fqs, vis.imag)
    vis = mdl_real(lsts, fqs) + 1j * mdl_imag(lsts, fqs)

    # introduce a spectral tilt and filter out certain modes
    visFFT = np.fft.fft(vis, axis=1)
    dlys = np.abs(np.fft.fftfreq(len(fqs), d=np.median(np.diff(fqs))) / 1e3).clip(1e-3, np.inf)
    visFFT *= dlys ** spec_tilt
    visFFT[:, dlys < np.abs(min_dly) / 1e3] = 0.0
    visFFT[:, dlys > np.abs(max_dly) / 1e3] = 0.0
    vis = np.fft.ifft(visFFT, axis=1)

    return vis



