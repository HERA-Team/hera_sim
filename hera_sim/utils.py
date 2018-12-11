""" Utility module """

import numpy as np
from scipy import interpolate
import aipy
from . import noise

def rough_delay_filter(noise, fqs, bl_len_ns):
    """
    A rough high-pass filtering of noise array
    across frequency.

    Args:
        noise : 1D or 2D ndarray, filtered along last axis
        fqs : 1D frequency array, [GHz]
        bl_len_ns : baseline length, [nanosec]

    Returns:
        filt_noise : delay-filtered noise
    """
    delays = np.fft.fftfreq(fqs.size, fqs[1]-fqs[0])
    _noise = np.fft.fft(noise)
    delay_filter = np.exp(-delays**2 / (2*bl_len_ns**2))
    delay_filter.shape = (1,) * (_noise.ndim-1) + (-1,)
    filt_noise = np.fft.ifft(_noise * delay_filter)
    return filt_noise

def calc_max_fringe_rate(fqs, bl_len_ns):
    """
    Calculate the fringe-rate max fringe-rate
    seen by an East-West baseline.

    Args:
        fqs : frequency array [GHz]
        bl_len_ns : East-West baseline length [ns]
    Returns:
        fr_max : fringe rate [Hz]
    """
    bl_wavelen = fqs * bl_len_ns
    fr_max = 2*np.pi/aipy.const.sidereal_day * bl_wavelen
    return fr_max

def rough_fringe_filter(noise, lsts, fqs, bl_len_ns, fr_width=None):
    """
    Perform a rough fringe rate filter on noise array
    along the zeroth axis.

    Args:
        noise : 1D or 2D ndarray, filtered along zeroth axis
        lsts : 1D lst array [radians]
        fqs : 1D frequency array [GHz]
        bl_len_ns : baseline length, [nanosec]
        fr_width : half-width of a Gaussian FR filter in [1/sec]
            to apply. If None, filter is a flat-top FR filter.
            Can be a float or an array of size fqs.

    Returns:
        filt_noise : fringe-rate-filtered noise
    """
    times = lsts / (2*np.pi) * aipy.const.sidereal_day
    fringe_rates = np.fft.fftfreq(times.size, times[1]-times[0])
    fringe_rates.shape = (-1,) + (1,) * (noise.ndim-1)
    _noise = np.fft.fft(noise, axis=0)
    fr_max = calc_max_fringe_rate(fqs, bl_len_ns)
    fr_max.shape = (1,) * (noise.ndim-1) + (-1,)

    if fr_width is None:
        # use a top-hat filter with width set by maximum fr
        fng_filter = np.where(np.abs(fringe_rates) < fr_max, 1., 0)
    else:
        # use a gaussian centered at max fr
        fng_filter = np.exp(-0.5 * ((fringe_rates-fr_max)/fr_width)**2)

    filt_noise = np.fft.ifft(_noise * fng_filter, axis=0)

    return filt_noise, fng_filter, fringe_rates


def custom_fringe_filter(noise, lsts, fqs, FR_filter, filt_lsts, filt_fqs, interp_mode='linear'):
    """
    Fringe-rate filter a noise array with a custom fringe-rate
    filter along the zeroth axis.

    Args:
        noise : 2D ndarray, filtered along zeroth axis
        lsts : 1D lst array for noise [radians]
        fqs : 1D frequency array for noise [GHz]
        FR_filter : 2D ndarray, FR filter to apply to noise
        filt_lsts : 1D lst array for FR_filter [radians]
        filt_fqs : 1D frequency array for FR_filter [GHz]
        interp_mode : str, interpolation mode.
            options=['nearest', 'linear', 'cubic']

    Returns:
        filt_noise : fringe-rate-filtered noise
    """
    # interpolate FR_filter at lsts and fqs
    mdl = interpolate.interp2d(filt_lsts, filt_fqs, FR_filter, kind=interp_mode)
    FR_filter = mdl(lsts, fqs)

    # set things close to zero to zero
    FR_filter[np.isclose(FR_filter, 0.0)] = 0.0

    # peak normalize filter along time
    FR_filter /= np.max(FR_filter, axis=0, keepdims=True)

    # FR noise, apply filter and FT back
    filt_noise = np.fft.fft(noise, axis=0)
    filt_noise = np.fft.ifft(filt_noise * FR_filter, axis=0)

    return filt_noise, FR_filter


def compute_ha(lsts, ra):
    ha = lsts - ra
    ha = np.where(ha > np.pi, ha-2*np.pi, ha)
    ha = np.where(ha < -np.pi, ha+2*np.pi, ha)
    return ha

