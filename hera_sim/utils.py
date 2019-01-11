""" Utility module """

import aipy
import numpy as np


def _get_bl_len_vec(bl_len_ns):
    """
    Converts a baseline length in a variety of formats to a standard
    length-3 vector.

    Args:
        bl_len_ns (scalar or array_like): the baseline length in nanosec (i.e.
            1e9 * metres / c). If scalar, interpreted as E-W length, if len(2),
            interpreted as EW and NS length, otherwise the full [EW, NS, Z]
            length. Unspecified dimensions are assumed to be zero.

    Returns:
        length-3 array: The full [EW, NS, Z] baseline vector.
    """
    if np.isscalar(bl_len_ns):
        bl_len_ns = np.array([bl_len_ns, 0, 0])
    elif len(bl_len_ns) < 3:
        # make a length-3 array
        bl_len_ns = np.pad(
            bl_len_ns, pad_width=3 - len(bl_len_ns),
            mode='constant'
        )[-3:]

    return bl_len_ns


def get_bl_len_magnitude(bl_len_ns):
    """
    Get the magnitude of the length of the given baseline.

    Args:
        bl_len_ns (scalar or array_like): the baseline length in nanosec (i.e.
            1e9 * metres / c). If scalar, interpreted as E-W length, if len(2),
            interpreted as EW and NS length, otherwise the full [EW, NS, Z]
            length. Unspecified dimensions are assumed to be zero.

    Returns:
        float: The magnitude of the baseline length.
    """
    bl_len_ns = _get_bl_len_vec(bl_len_ns)
    return np.sqrt(np.sum(bl_len_ns ** 2))


def rough_delay_filter(noise, fqs, bl_len_ns, normalise=None):
    """
    A rough high-pass filtering of noise array
    across frequency.

    Args:
        noise : 1D or 2D ndarray, filtered along last axis
        fqs : 1D frequency array, [GHz]
        bl_len_ns (scalar or array_like): the baseline length in nanosec (i.e.
            1e9 * metres / c). If scalar, interpreted as E-W length, if len(2),
            interpreted as EW and NS length, otherwise the full [EW, NS, Z]
            length. Unspecified dimensions are assumed to be zero.
        normalise: float, optional
            If set, will normalise the filter such that the power of the output
            matches the power of the input times the normalisation factor.
            If not set, the filter merely has a maximum of unity.

    Returns:
        filt_noise : delay-filtered noise
    """
    bl_len_ns = get_bl_len_magnitude(bl_len_ns)

    delays = np.fft.fftfreq(fqs.size, fqs[1] - fqs[0])
    _noise = np.fft.fft(noise)
    delay_filter = np.exp(-delays ** 2 / (2 * bl_len_ns ** 2))

    if normalise is not None:
        norm = normalise /np.sqrt(np.sum(delay_filter**2))
        delay_filter *= norm * np.sqrt(noise.shape[-1])

    delay_filter.shape = (1,) * (_noise.ndim - 1) + (-1,)
    filt_noise = np.fft.ifft(_noise * delay_filter)
    return filt_noise


def calc_max_fringe_rate(fqs, bl_len_ns):
    """
    Calculate the fringe-rate at zenith of
    a E-W baseline length

    Args:
        fqs : frequency array [GHz]
        bl_len_ns (scalar or array_like): the baseline length in nanosec (i.e.
            1e9 * metres / c). If scalar, interpreted as E-W length, if len(2),
            interpreted as EW and NS length, otherwise the full [EW, NS, Z]
            length. Unspecified dimensions are assumed to be zero.

    Returns:
        fr_max : fringe rate [lambda / sec]
    """
    # Convert to 3-vector
    bl_len_ns = _get_bl_len_vec(bl_len_ns)
    earth_rotation = np.array([0, 2 * np.pi / aipy.const.sidereal_day, 0])

    return fqs * np.cross(bl_len_ns, earth_rotation)[-1]
    # bl_wavelen = fqs * bl_len_ns
    # fr_max = 2*np.pi/aipy.const.sidereal_day * bl_wavelen
    # return fr_max


def rough_fringe_filter(noise, lsts, fqs, bl_len_ns, fr_width=None, normalise=None):
    """
    Perform a rough fringe rate filter on noise array
    along second-to-last axis.

    Args:
        noise : 1D or 2D ndarray, filtered along last axis
        fqs : 1D frequency array, [GHz]
        bl_len_ns (scalar or array_like): the baseline length in nanosec (i.e.
            1e9 * metres / c). If scalar, interpreted as E-W length, if len(2),
            interpreted as EW and NS length, otherwise the full [EW, NS, Z]
            length. Unspecified dimensions are assumed to be zero.
        fr_width : half-width of a Gaussian FR filter in [1/sec]
            to apply. If None filter is a flat-top FR filter.
            Can be a float or an array of size fqs.
        normalise: float, optional
            If set, will normalise the filter such that the power of the output
            matches the power of the input times the normalisation factor.
            If not set, the filter merely has a maximum of unity.

    Returns:
        filt_noise : fringe-rate-filtered noise
    """
    times = lsts / (2 * np.pi) * aipy.const.sidereal_day
    fringe_rates = np.fft.fftfreq(times.size, times[1] - times[0])
    fringe_rates.shape = (-1,) + (1,) * (noise.ndim - 1)
    _noise = np.fft.fft(noise, axis=-2)
    fr_max = calc_max_fringe_rate(fqs, bl_len_ns)
    fr_max.shape = (1,) * (noise.ndim - 1) + (-1,)

    if fr_width is None:
        # use a top-hat filter with width set by maximum fr
        fng_filter = np.where(np.abs(fringe_rates) < fr_max, 1., 0)
    else:
        # use a gaussian centered at max fr
        fng_filter = np.exp(-0.5 * ((fringe_rates - fr_max) / fr_width) ** 2)

    if normalise is not None:
        norm = normalise /np.sqrt(np.sum(fng_filter**2))
        fng_filter *= norm * np.sqrt(noise.shape[-2])

    filt_noise = np.fft.ifft(_noise * fng_filter, axis=-2)

    return filt_noise, fng_filter, fringe_rates


def compute_ha(lsts, ra):
    ha = lsts - ra
    ha = np.where(ha > np.pi, ha - 2 * np.pi, ha)
    ha = np.where(ha < -np.pi, ha + 2 * np.pi, ha)
    return ha
