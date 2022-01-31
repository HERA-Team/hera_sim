"""Utility module."""
import numpy as np
import astropy.constants as const
import astropy.units as u
from scipy.interpolate import RectBivariateSpline
from typing import Sequence, Optional, Tuple, Union
import warnings
from .interpolators import Beam


def _get_bl_len_vec(bl_len_ns: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convert a baseline length in a variety of formats to a standard length-3 vector.

    Parameters
    ----------
    bl_len_ns
        The baseline length in nanosec (i.e. 1e9 * metres / c). If scalar, interpreted
        as E-W length, if len(2), interpreted as EW and NS length, otherwise the full
        [EW, NS, Z] length. Unspecified dimensions are assumed to be zero.

    Returns
    -------
    bl_vec
        A length-3 array. The full [EW, NS, Z] baseline vector.
    """
    if np.isscalar(bl_len_ns):
        return np.array([bl_len_ns, 0, 0])
    elif len(bl_len_ns) <= 3:
        # make a length-3 array
        return np.pad(bl_len_ns, pad_width=3 - len(bl_len_ns), mode="constant")[-3:]

    return bl_len_ns


def get_bl_len_magnitude(bl_len_ns: Union[float, np.ndarray, Sequence]) -> float:
    """
    Get the magnitude of the length of the given baseline.

    Parameters
    ----------
    bl_len_ns
        The baseline length in nanosec (i.e. 1e9 * metres / c). If scalar, interpreted
        as E-W length, if len(2), interpreted as EW and NS length, otherwise the full
        [EW, NS, Z] length. Unspecified dimensions are assumed to be zero.

    Returns
    -------
    mag
        The magnitude of the baseline length.
    """
    bl_len_ns = _get_bl_len_vec(bl_len_ns)
    return np.sqrt(np.sum(bl_len_ns**2))


def gen_delay_filter(
    freqs: np.ndarray,
    bl_len_ns: Union[float, np.ndarray, Sequence],
    standoff: float = 0.0,
    delay_filter_type: Optional[str] = "gauss",
    min_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    normalize: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a delay filter in delay space.

    Parameters
    ----------
    freqs
        Frequency array [GHz]
    bl_len_ns
        The baseline length in nanosec (i.e. 1e9 * metres / c). If scalar, interpreted
        as E-W length, if len(2), interpreted as EW and NS length, otherwise the full
        [EW, NS, Z] length. Unspecified dimensions are assumed to be zero.
    standoff
        Supra-horizon buffer [nanosec]
    delay_filter_type
        Options are ``['gauss', 'trunc_gauss', 'tophat', 'none']``.
        This sets the filter profile. ``gauss`` has a 1-sigma as horizon (+ standoff)
        divided by four, ``trunc_gauss`` is same but truncated above 1-sigma. ``'none'``
        means filter is identically one.
    min_delay
        Minimum absolute delay of filter
    max_delay
        Maximum absolute delay of filter
    normalize
        If set, will normalize the filter such that the power of the output
        matches the power of the input times the normalization factor.
        If not set, the filter merely has a maximum of unity.

    Returns
    -------
    delay_filter
        Delay filter in delay space (1D)
    """
    # setup
    delays = np.fft.fftfreq(freqs.size, freqs[1] - freqs[0])
    if isinstance(bl_len_ns, np.ndarray):
        bl_len_ns = np.linalg.norm(bl_len_ns)

    # add standoff: four sigma is horizon
    one_sigma = (bl_len_ns + standoff) / 4.0

    # create filter
    if delay_filter_type in [None, "none", "None"]:
        delay_filter = np.ones_like(delays)
    elif delay_filter_type in ["gauss", "trunc_gauss"]:
        delay_filter = np.exp(-0.5 * (delays / one_sigma) ** 2)
        if delay_filter_type == "trunc_gauss":
            delay_filter[np.abs(delays) > (one_sigma * 4)] = 0.0
    elif delay_filter_type == "tophat":
        delay_filter = np.ones_like(delays)
        delay_filter[np.abs(delays) > (one_sigma * 4)] = 0.0
    else:
        raise ValueError(f"Didn't recognize filter_type {delay_filter_type}")

    # set bounds
    if min_delay is not None:
        delay_filter[np.abs(delays) < min_delay] = 0.0
    if max_delay is not None:
        delay_filter[np.abs(delays) > max_delay] = 0.0

    # normalize
    if normalize is not None and np.any(delay_filter):
        norm = normalize / np.sqrt(np.sum(delay_filter**2))
        delay_filter *= norm * np.sqrt(len(delay_filter))

    return delay_filter


def rough_delay_filter(
    data: np.ndarray,
    freqs: Optional[np.ndarray] = None,
    bl_len_ns: Optional[np.ndarray] = None,
    *,
    delay_filter: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    A rough low-pass delay filter of data array along last axis.

    Parameters
    ----------
    data
        Data to be filtered along last axis
    freqs
        Frequencies of the filter [GHz]
    bl_len_ns
        The baseline length (see :func:`gen_delay_filter`).
    delay_filter
        The pre-computed filter to use. A filter can be created on-the-fly by
        passing kwargs.
    **kwargs
        Passed to :func:`gen_delay_filter`.

    Returns
    -------
    filt_data
        Filtered data array (same shape as ``data``).
    """
    # fft data across last axis
    dfft = np.fft.fft(data, axis=-1)

    # get delay filter
    if delay_filter is None:
        if freqs is None:
            raise ValueError(
                "If you don't provide a pre-computed delay filter, you must "
                "provide freqs"
            )
        if bl_len_ns is None:
            raise ValueError(
                "If you don't provide a pre-computed delay filter, you must provide "
                "bl_len_ns"
            )

        delay_filter = gen_delay_filter(freqs=freqs, bl_len_ns=bl_len_ns, **kwargs)

    # apply filtering and fft back
    filt_data = np.fft.ifft(dfft * delay_filter, axis=-1)

    return filt_data


def gen_fringe_filter(
    lsts: np.ndarray,
    freqs: np.ndarray,
    ew_bl_len_ns: float,
    fringe_filter_type: Optional[str] = "tophat",
    **filter_kwargs,
) -> np.ndarray:
    """
    Generate a fringe rate filter in fringe-rate & freq space.

    Parameters
    ----------
    lsts
        lst array [radians]
    freqs
        Frequency array [GHz]
    ew_bl_len_ns
        Projected East-West baseline length [nanosec]
    fringe_filter_type
        Options ``['tophat', 'gauss', 'custom', 'none']``
    **filter_kwargs
        These are specific to each ``fringe_filter_type``.

        For ``filter_type == 'gauss'``:

            * **fr_width** (float or array): Sets gaussian width in fringe-rate [Hz]

        For ``filter_type == 'custom'``:

            * **FR_filter** (ndarray): shape (Nfrates, Nfreqs) with custom filter (must
              be fftshifted, see below)
            * **FR_frates** (ndarray): array of FR_filter fringe rates [Hz] (must be
              monotonically increasing)
            * **FR_freqs** (ndarray): array of FR_filter freqs [GHz]

    Returns
    -------
    fringe_filter
        2D array in fringe-rate & freq space

    Notes
    -----
    If ``filter_type == 'tophat'`` filter is a tophat out to max fringe-rate set by
    ew_bl_len_ns.

    If ``filter_type == 'gauss'`` filter is a Gaussian centered on max fringe-rate
    with width set by kwarg fr_width in Hz

    If ``filter_type == 'custom'`` filter is a custom 2D (Nfrates, Nfreqs) filter fed as
    'FR_filter' its fringe-rate array is fed as "FR_frates" in Hz, its freq array is
    fed as "FR_freqs" in GHz. Note that input ``FR_filter`` must be fft-shifted along
    axis 0, but output filter is ``ifftshift``-ed back along axis 0.

    If ``filter_type == 'none'`` fringe filter is identically one.
    """
    # setup
    times = lsts / (2 * np.pi) * u.sday.to("s")
    fringe_rates = np.fft.fftfreq(times.size, times[1] - times[0])

    if fringe_filter_type in [None, "none", "None"]:
        fringe_filter = np.ones((len(times), len(freqs)), dtype=float)
    elif fringe_filter_type == "tophat":
        fr_max = np.repeat(
            calc_max_fringe_rate(freqs, ew_bl_len_ns)[None, :], len(lsts), axis=0
        )
        fringe_rates = np.repeat(fringe_rates[:, None], len(freqs), axis=1)
        fringe_filter = np.where(np.abs(fringe_rates) <= np.abs(fr_max), 1.0, 0)
    elif fringe_filter_type == "gauss":
        assert (
            "fr_width" in filter_kwargs
        ), "If filter_type=='gauss' must feed fr_width kwarg"
        fr_max = np.repeat(
            calc_max_fringe_rate(freqs, ew_bl_len_ns)[None, :], len(lsts), axis=0
        )
        fringe_rates = np.repeat(fringe_rates[:, None], len(freqs), axis=1)
        fringe_filter = np.exp(
            -0.5 * ((fringe_rates - fr_max) / filter_kwargs["fr_width"]) ** 2
        )
    elif fringe_filter_type == "custom":
        assert (
            "FR_filter" in filter_kwargs
        ), "If filter_type=='custom', must feed 2D FR_filter array"
        assert (
            "FR_frates" in filter_kwargs
        ), "If filter_type=='custom', must feed 1D FR_frates array"
        assert (
            "FR_freqs" in filter_kwargs
        ), "If filter_type=='custom', must feed 1D FR_freqs array"
        # interpolate FR_filter at fringe_rates and fqs
        mdl = RectBivariateSpline(
            filter_kwargs["FR_frates"],
            filter_kwargs["FR_freqs"],
            filter_kwargs["FR_filter"],
            kx=3,
            ky=3,
        )
        fringe_filter = np.fft.ifftshift(
            mdl(np.fft.fftshift(fringe_rates), freqs), axes=0
        )
        # set things close to zero to zero
        fringe_filter[np.isclose(fringe_filter, 0.0)] = 0.0

    else:
        raise ValueError(f"filter_type {fringe_filter_type} not recognized")

    return fringe_filter


def rough_fringe_filter(
    data: np.ndarray,
    lsts: Optional[np.ndarray] = None,
    freqs: Optional[np.ndarray] = None,
    ew_bl_len_ns: Optional[float] = None,
    *,
    fringe_filter: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    A rough fringe rate filter of data along zeroth axis.

    Parameters
    ----------
    data
        data to filter along zeroth axis
    fringe_filter
        A pre-computed fringe-filter to use. Computed on the fly if not given.
    **kwargs
        Passed to :func:`gen_fringe_filter` to compute the fringe
        filter on the fly (if necessary). If so, at least ``lsts``, ``freqs``, and
        ``ew_bl_len_ns`` are required.

    Returns
    -------
    filt_data
        Filtered data (same shape as ``data``).

    """
    # fft data along zeroth axis
    dfft = np.fft.fft(data, axis=0)

    # get filter
    if fringe_filter is None:
        if any(k is None for k in [lsts, freqs, ew_bl_len_ns]):
            raise ValueError(
                "Must provide 'lsts', 'freqs' and 'ew_bl_len_ns' if fringe_filter not "
                "given."
            )

        fringe_filter = gen_fringe_filter(
            freqs=freqs, lsts=lsts, ew_bl_len_ns=ew_bl_len_ns, **kwargs
        )

    # apply filter
    filt_data = np.fft.ifft(dfft * fringe_filter, axis=0)

    return filt_data


def calc_max_fringe_rate(fqs: np.ndarray, ew_bl_len_ns: float) -> np.ndarray:
    """
    Calculate the max fringe-rate seen by an East-West baseline.

    Parameters
    ----------
    fqs
        Frequency array [GHz]
        ew_bl_len_ns (float): projected East-West baseline length [ns]
    ew_bl_len_ns
        The EW baseline length, in nanosec.

    Returns
    -------
    fr_max
        Maximum fringe rate [Hz]
    """
    bl_wavelen = fqs * ew_bl_len_ns
    return 2 * np.pi / u.sday.to("s") * bl_wavelen


def compute_ha(lsts: np.ndarray, ra: float) -> np.ndarray:
    """
    Compute hour angle from local sidereal time and right ascension.

    Parameters
    ----------
    lsts
        Local sidereal times of the observation to be generated [radians].
        Shape=(NTIMES,)
    ra
        The right ascension of a point source [radians].

    Returns
    -------
    ha
        Hour angle corresponding to the provide ra and times. Shape=(NTIMES,)
    """
    ha = lsts - ra
    ha = np.where(ha > np.pi, ha - 2 * np.pi, ha)
    ha = np.where(ha < -np.pi, ha + 2 * np.pi, ha)
    return ha


def wrap2pipi(a):
    """
    Wrap values of an array to [-π; +π] modulo 2π.

    Parameters
    ----------
    a: array_like
        Array of values to be wrapped to [-π; +π].

    Returns
    -------
    res: array_like
        Array of 'a' values wrapped to [-π; +π].
    """
    # np.fmod(~, 2π) outputs values in [0; 2π] or [-2π; 0]
    res = np.fmod(a, 2 * np.pi)
    # wrap [π; 2π] to [-π; 0]...
    res[np.where(res > np.pi)] -= 2 * np.pi
    # ... and [-2π; -π] to [0; π]
    res[np.where(res < -np.pi)] += 2 * np.pi
    return res


def gen_white_noise(size: Union[int, Tuple[int]] = 1) -> np.ndarray:
    """Produce complex Gaussian noise with unity variance.

    Parameters
    ----------
    size
        Shape of output array. Can be an integer if a single dimension is required,
        otherwise a tuple of ints.

    Returns
    -------
    noise
        White noise realization with specified shape.
    """
    std = 1 / np.sqrt(2)
    return np.random.normal(scale=std, size=size) + 1j * np.random.normal(
        scale=std, size=size
    )


def jansky_to_kelvin(freqs: np.ndarray, omega_p: Union[Beam, np.ndarray]) -> np.ndarray:
    """Return Kelvin -> Jy conversion as a function of frequency.

    Parameters
    ----------
    freqs
        Frequencies for which to calculate the conversion. Units of GHz.
    omega_p
        Beam area as a function of frequency. Must have the same shape
        as ``freqs`` if an ndarray. Otherwise, must be an interpolation
        object which converts frequencies (in GHz) to beam size.

    Returns
    -------
    Jy_to_K
        Array for converting Jy to K, same shape as ``freqs``.
    """
    # get actual values of omega_p if it's an interpolation object
    if callable(omega_p):
        omega_p = omega_p(freqs)

    wavelengths = const.c.value / (freqs * 1e9)  # meters
    # The factor of 1e-26 converts from Jy to W/m^2/Hz.
    return 1e-26 * wavelengths**2 / (2 * const.k_B.value * omega_p)


def Jy2T(freqs, omega_p):
    """Convert Janskys to Kelvin.

    Deprecated in v1.0.0. Will be removed in v1.1.0
    """
    warnings.warn(
        "The function Jy2T has been renamed 'jansky_to_kelvin'. It will be removed in "
        "v1.1."
    )
    return jansky_to_kelvin(freqs, omega_p)


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
