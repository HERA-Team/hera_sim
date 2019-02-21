""" Utility module """

import numpy as np
from scipy import interpolate
import aipy
from . import noise


def gen_delay_filter(fqs, bl_len_ns, standoff=0.0, filter_type='gauss'):
    """
    Generate a delay filter in delay space.

    Args:
        fqs : 1D frequency array, [GHz]
        bl_len_ns : total baseline length, [nanosec]
        standoff : supra-horizon buffer, [nanosec]
        filter_type : str, options=['gauss', 'trunc_gauss', 'tophat', 'none']
            This sets the filter profile. Gauss has a 1-sigma
            as horizon (+ standoff) divided by four, trunc_gauss
            is same but truncated above 4-sigma. 'none' means
            filter is identically one.

    Returns:
        delay_filter : 1D array of delay filter in delay space
    """
    # setup
    delays = np.fft.fftfreq(fqs.size, fqs[1]-fqs[0])

    # add standoff: four sigma is horizon
    one_sigma = (bl_len_ns + standoff) / 4.0

    # create filter
    if filter_type in [None, 'none', 'None']:
        delay_filter = np.ones_like(delays)
    elif filter_type in ['gauss', 'trunc_gauss']:
        delay_filter = np.exp(-0.5 * (delays / one_sigma)**2)
        if filter_type == 'trunc_gauss':
            delay_filter[np.abs(delays) > (one_sigma * 4)] = 0.0
    elif filter_type == 'tophat':
        delay_filter = np.ones_like(delays)
        delay_filter[np.abs(delays) > (one_sigma * 4)] = 0.0
    else:
        raise ValueError("Didn't recognize filter_type {}".format(filter_type))

    # normalize filter by area
    delay_filter /= np.sum(delay_filter)

    return delay_filter


def rough_delay_filter(data, fqs, bl_len_ns, standoff=0.0, filter_type='gauss'):
    """
    A rough low-pass delay filter of data array along last axis.

    Args:
        data : 1D or 2D ndarray, filtered along last axis
        fqs : 1D frequency array, [GHz]
        bl_len_ns : total baseline length, [nanosec]
        standoff : supra-horizon buffer, [nanosec]
        filter_type : str, options=['gauss', 'trunc_gauss', 'tophat', 'none']
            This sets the filter profile. Gauss has a 1-sigma
            as horizon (+ standoff) divided by two, trunc_gauss
            is same but truncated above 2-sigma. 'none' means
            filter is identically one.

    Returns:
        filt_data : filtered data array
        delay_filter : delay filter applied to data
    """
    # fft data across last axis
    dfft = np.fft.ifft(data, axis=-1)

    # get delay filter
    delay_filter = gen_delay_filter(fqs, bl_len_ns, standoff=standoff, filter_type=filter_type)

    # apply filtering and fft back
    filt_data = np.fft.fft(dfft * delay_filter, axis=-1)

    return filt_data, delay_filter


def gen_fringe_filter(lsts, fqs, ew_bl_len_ns, filter_type='tophat', **filter_kwargs):
    """
    Generate a fringe rate filter in fringe-rate & freq space.

    Args:
        lsts : 1D lst array [radians]
        fqs : 1D frequency array [GHz]
        ew_bl_len_ns : projected East-West baseline length, [nanosec]
        filter_type : str, options=['tophat', 'gauss', 'custom', 'none']
        filter_kwargs : kwargs for different filter types
            fr_width : ('gauss') float or 1D array of len(fqs), sets gaussian width
            FR_filter : ('custom') 2D array (Nfrates, Nfreqs) with custom filter (must be fftshifted, see below)
            FR_frates : ('custom') 1D array of FR_filter fringe rates [Hz] (must be monotonically increasing)
            FR_freqs : ('custom') 1D array of FR_filter freqs [GHz]

    Returns:
        fringe_filter : 2D array of fringe filter in fringe-rate & freq space

    Notes:
        If filter_type == 'tophat'
            filter is a tophat out to max fringe-rate set by ew_bl_len_ns
        If filter_type == 'gauss':
            filter is a Gaussian centered on max fringe-rate
            with width set by kwarg fr_width in Hz
        If filter_type == 'custom':
            filter is a custom 2D (Nfrates, Nfreqs) filter fed as 'FR_filter'
            its frate array is fed as "FR_frates" in Hz, its freq array is
            fed as "FR_freqs" in GHz
            Note that input FR_filter must be fftshifted along axis 0,
            but output filter is ifftshifted back along axis 0.
        If filter_type == 'none':
            fringe filter is identically one.
    """
    # setup
    times = lsts / (2*np.pi) * aipy.const.sidereal_day
    frates = np.fft.fftfreq(times.size, times[1]-times[0])

    if filter_type in [None, 'none', 'None']:
        fringe_filter = np.ones((len(times), len(fqs)), dtype=np.float)
    elif filter_type == 'tophat':
        fr_max = np.repeat(calc_max_fringe_rate(fqs, ew_bl_len_ns)[None, :], len(lsts), axis=0)
        frates = np.repeat(frates[:, None], len(fqs), axis=1)
        fringe_filter = np.where(np.abs(frates) < fr_max, 1., 0)
    elif filter_type == 'gauss':
        assert 'fr_width' in filter_kwargs, "If filter_type=='gauss' must feed fr_width kwarg"
        fr_max = np.repeat(calc_max_fringe_rate(fqs, ew_bl_len_ns)[None, :], len(lsts), axis=0)
        frates = np.repeat(frates[:, None], len(fqs), axis=1)
        fringe_filter = np.exp(-0.5 * ((frates - fr_max) / filter_kwargs['fr_width'])**2)
    elif filter_type == 'custom':
        assert 'FR_filter' in filter_kwargs, "If filter_type=='custom', must feed 2D FR_filter array"
        assert 'FR_frates' in filter_kwargs, "If filter_type=='custom', must feed 1D FR_frates array"
        assert 'FR_freqs' in filter_kwargs, "If filter_type=='custom', must feed 1D FR_freqs array"
        # interpolate FR_filter at frates and fqs
        mdl = interpolate.RectBivariateSpline(filter_kwargs['FR_frates'], filter_kwargs['FR_freqs'],
                                              filter_kwargs['FR_filter'], kx=3, ky=3)
        fringe_filter = np.fft.ifftshift(mdl(np.fft.fftshift(frates), fqs), axes=0)
        # set things close to zero to zero
        fringe_filter[np.isclose(fringe_filter, 0.0)] = 0.0
    else:
        raise ValueError("filter_type {} not recognized".format(filter_type))

    return fringe_filter


def rough_fringe_filter(data, lsts, fqs, ew_bl_len_ns, filter_type='tophat', **filter_kwargs):
    """
    A rough fringe rate filter of data along zeroth axis.

    Args:
        data : 1D or 2D array to filter along zeroth axis
        lsts : 1D lst array [radians]
        fqs : 1D frequency array [GHz]
        ew_bl_len_ns : projected East-West baseline length, [nanosec]
        filter_type : str, options=['tophat', 'gauss', 'custom', 'none']
        filter_kwargs : kwargs for different filter types
            fr_width : ('gauss') float or 1D array of len(fqs), sets gaussian width
            FR_filter : ('custom') 2D array (Nfrates, Nfreqs) with custom filter
            FR_frates : ('custom') 1D array of FR_filter fringe rates [Hz]
            FR_freqs : ('custom') 1D array of FR_filter freqs [GHz]

    Returns:
        filt_data : filtered data
        fringe_filter : 2D array of fringe filter in fringe-rate & freq space

    Notes:
        If filter_type == 'tophat'
            filter is a tophat out to max fringe-rate set by ew_bl_len_ns
        If filter_type == 'gauss':
            filter is a Gaussian centered on max fringe-rate
            with width set by kwarg fr_width in Hz
        If filter_type == 'custom':
            filter is a custom 2D (Nfrates, Nfreqs) filter fed as 'FR_filter'
            its frate array is fed as "FR_frates" in Hz, its freq array is
            fed as "FR_freqs" in GHz
        If filter_type == 'none':
            fringe filter is identically one.
    """
    # fft data along zeroth axis
    dfft = np.fft.ifft(data, axis=0)

    # get filter
    fringe_filter = gen_fringe_filter(lsts, fqs, ew_bl_len_ns, filter_type=filter_type, **filter_kwargs)

    # apply filter
    filt_data = np.fft.fft(dfft * fringe_filter, axis=0)

    return filt_data, fringe_filter


def calc_max_fringe_rate(fqs, ew_bl_len_ns):
    """
    Calculate the max fringe-rate
    seen by an East-West baseline.

    Args:
        fqs : frequency array [GHz]
        ew_bl_len_ns : projected East-West baseline length [ns]
    Returns:
        fr_max : fringe rate [Hz]
    """
    bl_wavelen = fqs * ew_bl_len_ns
    fr_max = 2*np.pi/aipy.const.sidereal_day * bl_wavelen
    return fr_max


def compute_ha(lsts, ra):
    ha = lsts - ra
    ha = np.where(ha > np.pi, ha-2*np.pi, ha)
    ha = np.where(ha < -np.pi, ha+2*np.pi, ha)
    return ha

