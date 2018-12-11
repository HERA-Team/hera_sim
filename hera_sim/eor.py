'''A module for generating a rough eor-like signal.'''

import numpy as np
from scipy import interpolate
import aipy
from . import noise
from . import utils


def noiselike_eor(lsts, fqs, bl_len_ns, eor_amp=1e-5, spec_tilt=0.0,
                  fr_width=None, min_delay=0, max_delay=3000, fr_max_mult=2.0,
                  interp_mode='nearest'):
    """
    Generate a noise-like EoR signal that is fringe-rate filtered
    according to its projected East-West baseline length.

    Args:
        lsts : ndarray with LSTs [radians]
        fqs : ndarray with frequencies [GHz]
        bl_len_ns : float, East-West baseline length [nanosec]
        eor_amp : float, amplitude of EoR signal [arbitrary]
        spec_tilt : float, spectral slope of EoR spectral amplitude
            as a function of delay in microseconds
        fr_width : float, width of Gaussian FR filter in 1 / sec
        min_delay : float, minimum |delay| in nanosec of EoR signal
        max_delay : float, maximum |delay| in nanosec of EoR signal
        fr_max_mult : float, multiplier of fr_max to get lst_grid resolution
        interp_mode : str, method of interpolating visibility from oversampled
            LST grid to the desired LSTs. options=['nearest', 'linear', 'cubic']

    Returns: 
        vis : 2D ndarray holding simulated complex visibility
    """
    # get fringe rate and generate an LST grid
    fr_max = np.max(utils.calc_max_fringe_rate(fqs, bl_len_ns))
    dt = 1.0/(fr_max_mult * fr_max)  # over-resolve by fr_mult factor
    ntimes = int(np.around(aipy.const.sidereal_day / dt))
    lst_grid = np.linspace(0, 2*np.pi, ntimes, endpoint=False)

    # generate white noise
    vis = noise.white_noise((ntimes, len(fqs))) * eor_amp

    # Fringe-Rate Filter given baseline
    vis, ff, frs = utils.rough_fringe_filter(vis, lst_grid, fqs, bl_len_ns, fr_width=fr_width)

    # interpolate visibility
    mdl_real = interpolate.interp1d(lst_grid, vis.real.T, kind=interp_mode)
    mdl_imag = interpolate.interp1d(lst_grid, vis.imag.T, kind=interp_mode)
    vis = mdl_real(lsts).T + 1j * mdl_imag(lsts).T

    # introduce a spectral tilt and filter out certain modes
    visFFT = np.fft.fft(vis, axis=1)
    delays = np.abs(np.fft.fftfreq(len(fqs), d=np.median(np.diff(fqs))) / 1e3).clip(1e-3, np.inf)
    visFFT *= delays ** spec_tilt
    visFFT[:, delays < np.abs(min_delay) / 1e3] = 0.0
    visFFT[:, delays > np.abs(max_delay) / 1e3] = 0.0
    vis = np.fft.ifft(visFFT, axis=1)

    return vis

