""" Utility module """
import numpy as np
import astropy.constants as const
import astropy.units as u
from scipy.interpolate import RectBivariateSpline


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
        return np.array([bl_len_ns, 0, 0])
    elif len(bl_len_ns) < 3:
        # make a length-3 array
        return np.pad(bl_len_ns, pad_width=3 - len(bl_len_ns), mode="constant")[-3:]

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


def gen_delay_filter(
    fqs,
    bl_len_ns,
    standoff=0.0,
    filter_type="gauss",
    min_delay=None,
    max_delay=None,
    normalize=None,
):
    """
    Generate a delay filter in delay space.

    Args:
        fqs (ndarray): frequency array [GHz]
        bl_len_ns (float or array): total baseline length or baseline vector in [ns]
        standoff (float): supra-horizon buffer [nanosec]
        filter_type (str): options=['gauss', 'trunc_gauss', 'tophat', 'none']
            This sets the filter profile. Gauss has a 1-sigma
            as horizon (+ standoff) divided by four, trunc_gauss
            is same but truncated above 1-sigma. 'none' means
            filter is identically one.
        min_delay (float): minimum absolute delay of filter
        max_delay (float): maximum absolute delay of filter
        normalize: float, optional
            If set, will normalize the filter such that the power of the output
            matches the power of the input times the normalization factor.
            If not set, the filter merely has a maximum of unity.

    Returns:
        delay_filter (ndarray): delay filter in delay space
    """
    # setup
    delays = np.fft.fftfreq(fqs.size, fqs[1] - fqs[0])
    if isinstance(bl_len_ns, np.ndarray):
        bl_len_ns = np.linalg.norm(bl_len_ns)

    # add standoff: four sigma is horizon
    one_sigma = (bl_len_ns + standoff) / 4.0

    # create filter
    if filter_type in [None, "none", "None"]:
        delay_filter = np.ones_like(delays)
    elif filter_type in ["gauss", "trunc_gauss"]:
        delay_filter = np.exp(-0.5 * (delays / one_sigma) ** 2)
        if filter_type == "trunc_gauss":
            delay_filter[np.abs(delays) > (one_sigma * 4)] = 0.0
    elif filter_type == "tophat":
        delay_filter = np.ones_like(delays)
        delay_filter[np.abs(delays) > (one_sigma * 4)] = 0.0
    else:
        raise ValueError("Didn't recognize filter_type {}".format(filter_type))

    # set bounds
    if min_delay is not None:
        delay_filter[np.abs(delays) < min_delay] = 0.0
    if max_delay is not None:
        delay_filter[np.abs(delays) > max_delay] = 0.0

    # normalize
    if normalize is not None:
        norm = normalize / np.sqrt(np.sum(delay_filter ** 2))
        delay_filter *= norm * np.sqrt(len(delay_filter))

    return delay_filter


def rough_delay_filter(
    data,
    fqs,
    bl_len_ns,
    standoff=0.0,
    delay_filter_type="gauss",
    min_delay=None,
    max_delay=None,
    normalize=None,
):
    """
    A rough low-pass delay filter of data array along last axis.

    Args:
        data (ndarray): data to be filtered along last axis
        fqs (ndarray) : frequency array [GHz]
        bl_len_ns (float or array): total baseline length or baseline vector [nanosec]
        standoff (float): supra-horizon buffer [nanosec]
        filter_type (str): options=['gauss', 'trunc_gauss', 'tophat', 'none']
            This sets the filter profile. Gauss has a 1-sigma
            as horizon (+ standoff) divided by four, trunc_gauss
            is same but truncated above 1-sigma. 'none' means
            filter is identically one.
        min_delay (float): minimum absolute delay of filter
        max_delay (float): maximum absolute delay of filter
        normalize: float, optional
            If set, will normalize the filter such that the power of the output
            matches the power of the input times the normalization factor.
            If not set, the filter merely has a maximum of unity.

    Returns:
        filt_data (ndarray): filtered data array
    """
    # fft data across last axis
    dfft = np.fft.fft(data, axis=-1)

    # get delay filter
    delay_filter = gen_delay_filter(
        fqs,
        bl_len_ns,
        standoff=standoff,
        filter_type=delay_filter_type,
        min_delay=min_delay,
        max_delay=max_delay,
        normalize=normalize,
    )

    # apply filtering and fft back
    filt_data = np.fft.ifft(dfft * delay_filter, axis=-1)

    return filt_data


def gen_fringe_filter(lsts, fqs, ew_bl_len_ns, filter_type="tophat", **filter_kwargs):
    """
    Generate a fringe rate filter in fringe-rate & freq space.

    Args:
        lsts (ndarray): lst array [radians]
        fqs (ndarray): frequency array [GHz]
        ew_bl_len_ns (float): projected East-West baseline length [nanosec]
        filter_type (str): options=['tophat', 'gauss', 'custom', 'none']
        filter_kwargs: kwargs for different filter types
            filter_type == 'gauss'
                fr_width (float or array): Sets gaussian width in fringe-rate [Hz]
            filter_type == 'custom'
                FR_filter (ndarray): shape (Nfrates, Nfreqs) with custom filter (must be fftshifted, see below)
                FR_frates (ndarray): array of FR_filter fringe rates [Hz] (must be monotonically increasing)
                FR_freqs (ndarray): array of FR_filter freqs [GHz]

    Returns:
        fringe_filter (ndarray): 2D ndarray in fringe-rate & freq space

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
    times = lsts / (2 * np.pi) * u.sday.to("s")
    frates = np.fft.fftfreq(times.size, times[1] - times[0])

    if filter_type in [None, "none", "None"]:
        fringe_filter = np.ones((len(times), len(fqs)), dtype=np.float)
    elif filter_type == "tophat":
        fr_max = np.repeat(
            calc_max_fringe_rate(fqs, ew_bl_len_ns)[None, :], len(lsts), axis=0
        )
        frates = np.repeat(frates[:, None], len(fqs), axis=1)
        fringe_filter = np.where(np.abs(frates) <= np.abs(fr_max), 1.0, 0)
    elif filter_type == "gauss":
        assert (
            "fr_width" in filter_kwargs
        ), "If filter_type=='gauss' must feed fr_width kwarg"
        fr_max = np.repeat(
            calc_max_fringe_rate(fqs, ew_bl_len_ns)[None, :], len(lsts), axis=0
        )
        frates = np.repeat(frates[:, None], len(fqs), axis=1)
        fringe_filter = np.exp(
            -0.5 * ((frates - fr_max) / filter_kwargs["fr_width"]) ** 2
        )
    elif filter_type == "custom":
        assert (
            "FR_filter" in filter_kwargs
        ), "If filter_type=='custom', must feed 2D FR_filter array"
        assert (
            "FR_frates" in filter_kwargs
        ), "If filter_type=='custom', must feed 1D FR_frates array"
        assert (
            "FR_freqs" in filter_kwargs
        ), "If filter_type=='custom', must feed 1D FR_freqs array"
        # interpolate FR_filter at frates and fqs
        mdl = RectBivariateSpline(
            filter_kwargs["FR_frates"],
            filter_kwargs["FR_freqs"],
            filter_kwargs["FR_filter"],
            kx=3,
            ky=3,
        )
        fringe_filter = np.fft.ifftshift(mdl(np.fft.fftshift(frates), fqs), axes=0)
        # set things close to zero to zero
        fringe_filter[np.isclose(fringe_filter, 0.0)] = 0.0

    else:
        raise ValueError("filter_type {} not recognized".format(filter_type))

    return fringe_filter


def rough_fringe_filter(
    data, lsts, fqs, ew_bl_len_ns, fringe_filter_type="tophat", **filter_kwargs
):
    """
    A rough fringe rate filter of data along zeroth axis.

    Args:
        data (ndarray): data to filter along zeroth axis
        lsts (ndarray): LST array [radians]
        fqs (ndarray): frequency array [GHz]
        ew_bl_len_ns (float): projected East-West baseline length [nanosec]
        filter_type (str): options=['tophat', 'gauss', 'custom', 'none']
        filter_kwargs: kwargs for different filter types
            filter_type == 'gauss'
                fr_width (float or array): Sets gaussian width in fringe-rate [Hz]
            filter_type == 'custom'
                FR_filter (ndarray): shape (Nfrates, Nfreqs) with custom filter (must be fftshifted, see below)
                FR_frates (ndarray): array of FR_filter fringe rates [Hz] (must be monotonically increasing)
                FR_freqs (ndarray): array of FR_filter freqs [GHz]

    Returns:
        filt_data (ndarray): filtered data

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
    dfft = np.fft.fft(data, axis=0)

    # get filter
    fringe_filter = gen_fringe_filter(
        lsts, fqs, ew_bl_len_ns, filter_type=fringe_filter_type, **filter_kwargs
    )

    # apply filter
    filt_data = np.fft.ifft(dfft * fringe_filter, axis=0)

    return filt_data


def calc_max_fringe_rate(fqs, ew_bl_len_ns):
    """
    Calculate the max fringe-rate seen by an East-West baseline.

    Args:
        fqs (ndarray) : frequency array [GHz]
        ew_bl_len_ns (float): projected East-West baseline length [ns]

    Returns:
        fr_max (float): fringe rate [Hz]
    """
    bl_wavelen = fqs * ew_bl_len_ns
    fr_max = 2 * np.pi / u.sday.to("s") * bl_wavelen
    return fr_max


def compute_ha(lsts, ra):
    """
    Compute hour angle from local sidereal time and right ascension.

    Arg:
        lsts: array-like, shape=(NTIMES,), radians
            local sidereal times of the observation to be generated.
        ra: float, radians
            the right ascension of a point source.

    Returns:
        ha: array-like, shape=(NTIMES,)
            hour angle corresponding to the provide ra and times'''
    """
    ha = lsts - ra
    ha = np.where(ha > np.pi, ha - 2 * np.pi, ha)
    ha = np.where(ha < -np.pi, ha + 2 * np.pi, ha)
    return ha


def gen_white_noise(size=1):
    """Produce complex Gaussian noise with unity variance.

    Parameters
    ----------
    size : int or tuple, optional
        Shape of output array.

    Returns
    -------
    noise : ndarray
        White noise realization with specified shape.
    """
    std = 1 / np.sqrt(2)
    return np.random.normal(scale=std, size=size) + 1j * np.random.normal(
        scale=std, size=size
    )


def Jy2T(freqs, omega_p):
    """Return Kelvin -> Jy conversion as a function of frequency.

    Parameters
    ----------
    freqs : ndarray
        Frequencies for which to calculate the conversion. Units of Hz.
    omega_p : ndarray or interpolators.Beam
        Beam area as a function of frequency. Must have the same shape
        as ``freqs`` if an ndarray. Otherwise, must be an interpolation
        object which converts frequencies (in Hz) to beam size.

    Returns
    -------
    Jy_to_K : ndarray
        Array for converting Jy to K, same shape as ``freqs``.
    """
    # get actual values of omega_p if it's an interpolation object
    if callable(omega_p):
        omega_p = omega_p(freqs)
    wavelengths = const.c.value / freqs
    # scaling went from 1e-23 -> 1e-34 in converting to SI
    # what is the point of this multiplicative constant?
    return 1e-34 * wavelengths ** 2 / (2 * const.k_B.value * omega_p)


def _listify(x):
    """Ensure a scalar/list is returned as a list.

    Taken from https://stackoverflow.com/a/1416677/1467820

    Copied from the pre-v1 hera_sim.rfi module.
    """
    try:
        basestring
    except NameError:
        basestring = (str, bytes)

    if isinstance(x, basestring):
        return [x]
    else:
        try:
            iter(x)
        except TypeError:
            return [x]
        else:
            return list(x)
