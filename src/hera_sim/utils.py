"""Utility module."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pyuvdata.utils as uvutils
from astropy import constants, units
from astropy.coordinates import Longitude
from scipy.interpolate import RectBivariateSpline

from .interpolators import Beam

try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


def _get_bl_len_vec(bl_len_ns: float | np.ndarray) -> np.ndarray:
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


def get_bl_len_magnitude(bl_len_ns: float | np.ndarray | Sequence) -> float:
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
    bl_len_ns: float | np.ndarray | Sequence,
    standoff: float = 0.0,
    delay_filter_type: str | None = "gauss",
    min_delay: float | None = None,
    max_delay: float | None = None,
    normalize: float | None = None,
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
    freqs: np.ndarray | None = None,
    bl_len_ns: np.ndarray | None = None,
    *,
    delay_filter: np.ndarray | None = None,
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
    fringe_filter_type: str | None = "tophat",
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
    times = lsts / (2 * np.pi) * units.sday.to("s")
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
    lsts: np.ndarray | None = None,
    freqs: np.ndarray | None = None,
    ew_bl_len_ns: float | None = None,
    *,
    fringe_filter: np.ndarray | None = None,
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
    return 2 * np.pi / units.sday.to("s") * bl_wavelen


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


def gen_white_noise(
    size: int | tuple[int] = 1, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Produce complex Gaussian noise with unity variance.

    Parameters
    ----------
    size
        Shape of output array. Can be an integer if a single dimension is required,
        otherwise a tuple of ints.
    rng
        Random number generator.

    Returns
    -------
    noise
        White noise realization with specified shape.
    """
    # Split power evenly between real and imaginary components.
    std = 1 / np.sqrt(2)
    args = dict(scale=std, size=size)

    # Create a random number generator if needed, then generate noise.
    rng = rng or np.random.default_rng()
    return rng.normal(**args) + 1j * rng.normal(**args)


def jansky_to_kelvin(freqs: np.ndarray, omega_p: Beam | np.ndarray) -> np.ndarray:
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

    wavelengths = constants.c.value / (freqs * 1e9)  # meters
    # The factor of 1e-26 converts from Jy to W/m^2/Hz.
    return 1e-26 * wavelengths**2 / (2 * constants.k_B.value * omega_p)


def Jy2T(freqs, omega_p):
    """Convert Janskys to Kelvin.

    Deprecated in v1.0.0. Will be removed in v1.1.0
    """
    warnings.warn(
        "This function has been deprecated. Please use `jansky_to_kelvin` instead.",
        stacklevel=1,
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


def reshape_vis(
    vis: np.ndarray,
    ant_1_array: np.ndarray,
    ant_2_array: np.ndarray,
    pol_array: np.ndarray,
    antenna_numbers: np.ndarray,
    n_times: int,
    n_freqs: int,
    n_ants: int,
    n_pols: int,
    invert: bool = False,
    use_numba: bool = True,
) -> np.ndarray:
    """Reshaping helper for mutual coupling sims.

    The mutual coupling simulations take as input, and return, a data array with
    shape ``(Nblts, Nfreqs, Npols)``, but perform matrix multiplications on
    the data array reshaped to ``(Ntimes, Nfreqs, 2*Nants, 2*Nants)``. This
    function performs the reshaping between the matrix multiply shape and the
    input/output array shapes.

    Parameters
    ----------
    vis
        Input data array.
    ant_1_array
        Array specifying the first antenna in each baseline.
    ant_2_array
        Array specifying the second antenna in each baseline.
    pol_array
        Array specifying the observed polarizations via polarization numbers.
    antenna_numbers
        Array specifying all of the antennas to include in the reshaped data.
    n_times
        Number of integrations in the data.
    n_freqs
        Number of frequency channels in the data.
    n_ants
        Number of antennas.
    n_pols
        Number of polarizations in the data.
    invert
        Whether to reshape to :class:`pyuvdata.UVData`'s data array shape.
    use_numba
        Whether to use ``numba`` to speed up the reshaping.

    Returns
    -------
    reshaped_vis
        Input data reshaped to desired shape.
    """
    if invert:
        out = np.zeros((ant_1_array.size, n_freqs, n_pols), dtype=complex)
    else:
        out = np.zeros((n_times, n_freqs, 2 * n_ants, 2 * n_ants), dtype=complex)

    # If we have numba, then this is a bit faster.
    if HAVE_NUMBA and use_numba:  # pragma: no cover
        if invert:
            fnc = jit_reshape_vis_invert
        else:
            fnc = jit_reshape_vis
        fnc(
            vis=vis,
            out=out,
            ant_1_array=ant_1_array,
            ant_2_array=ant_2_array,
            pol_array=pol_array,
            antenna_numbers=antenna_numbers,
        )
        return out

    # We don't have numba, so we need to do this a bit more slowly.
    pol_slices = {"x": slice(None, None, 2), "y": slice(1, None, 2)}
    polnum2str = {pol: uvutils.polnum2str(pol) for pol in pol_array}
    for i, ai in enumerate(antenna_numbers):
        for j, aj in enumerate(antenna_numbers[i:]):
            j += i
            uvd_inds = np.argwhere((ant_1_array == ai) & (ant_2_array == aj)).flatten()
            flipped = uvd_inds.size == 0
            ii, jj = i, j
            if flipped:
                uvd_inds = np.argwhere(
                    (ant_2_array == ai) & (ant_1_array == aj)
                ).flatten()
                ii, jj = j, i

            if uvd_inds.size == 0:
                continue

            for k, pol in enumerate(pol_array):
                p1, p2 = polnum2str[pol]
                if flipped:
                    p1, p2 = p2, p1
                sl1, sl2 = (pol_slices[p.lower()] for p in (p1, p2))

                # NOTE: this is hard-coded to use the new-style UVData shapes!
                if invert:
                    # Going back to UVData shape
                    out[uvd_inds, :, k] = vis[:, :, sl1, sl2][:, :, ii, jj]
                else:
                    # Changing from UVData shape
                    out[:, :, sl1, sl2][:, :, ii, jj] = vis[uvd_inds, :, k]
                    out[:, :, sl2, sl1][:, :, jj, ii] = np.conj(vis[uvd_inds, :, k])
    return out


def matmul(left: np.ndarray, right: np.ndarray, use_numba: bool = False) -> np.ndarray:
    """Helper function for matrix multiplies used in mutual coupling sims.

    The :class:`~sigchain.MutualCoupling` class performs two matrix
    multiplications of arrays with shapes ``(1, Nfreqs, 2*Nant, 2*Nant)``
    and ``(Ntimes, Nfreqs, 2*Nant, 2*Nant)``. Typically the number of antennas
    is much less than the number of frequency channels, so the parallelization
    used by ``numpy``'s matrix multiplication routine tends to be sub-optimal.
    This routine--when used with ``numba``--produces a substantial speedup in
    matrix multiplication for typical HERA-sized problems.

    Parameters
    ----------
    left, right
        Input arrays to perform matrix multiplication left @ right.
    use_numba
        Whether to use ``numba`` to speed up the matrix multiplication.

    Returns
    -------
    prod
        Product of the matrix multiplication left @ right.

    Notes
    -----
    """
    if HAVE_NUMBA and use_numba:
        if left.shape[0] == 1:
            return _left_matmul(left, right)
        elif right.shape[0] == 1:
            return _right_matmul(left, right)
        elif left.shape == right.shape:
            return _matmul(left, right)
        else:
            raise ValueError("Inputs cannot be broadcast to a common shape.")
    else:
        return left @ right


def find_baseline_orientations(
    antenna_numbers: np.ndarray, enu_antpos: np.ndarray
) -> dict[tuple[int, int], float]:
    """Find the orientation of each redundant baseline group.

    Parameters
    ----------
    antenna_numbers
        Array containing antenna numbers corresponding to the provided
        antenna positions.
    enu_antpos
        ``(Nants,3)`` array containing the antenna positions in a local
        topocentric frame with basis (east, north, up).

    Returns
    -------
    antpair2angle
        Dictionary mapping antenna pairs ``(ai,aj)`` to baseline orientations.
        Orientations are defined on [0,2pi).
    """
    groups, baselines = uvutils.redundancy.get_antenna_redundancies(
        antenna_numbers, enu_antpos, include_autos=False
    )[:2]
    antpair2angle = {}
    for group, (e, n, _u) in zip(groups, baselines):
        angle = Longitude(np.arctan2(n, e) * units.rad).value
        conj_angle = Longitude((angle + np.pi) * units.rad).value
        for blnum in group:
            ai, aj = uvutils.baseline_to_antnums(
                blnum, Nants_telescope=antenna_numbers.size
            )
            antpair2angle[(ai, aj)] = angle
            antpair2angle[(aj, ai)] = conj_angle
    return antpair2angle


def tanh_window(x, x_min=None, x_max=None, scale_low=1, scale_high=1):
    if x_min is None and x_max is None:
        warnings.warn(
            "Insufficient information provided; you must provide either x_min or "
            "x_max. Returning uniform window.",
            stacklevel=1,
        )
        return np.ones(x.size)

    window = np.ones(x.size)
    if x_min is not None:
        window *= 0.5 * (1 + np.tanh((x - x_min) / scale_low))

    if x_max is not None:
        window *= 0.5 * (1 + np.tanh((x_max - x) / scale_high))

    return window


# Just some numba-fied helpful functions.
# Note that coverage can't see that these are run without disabling JIT,
# which kind of defeats the purpose of testing it.
if HAVE_NUMBA:  # pragma: no cover

    @numba.njit
    def jit_reshape_vis(vis, out, ant_1_array, ant_2_array, pol_array, antenna_numbers):
        """JIT-accelerated reshaping function.

        See :func:`~reshape_vis` for parameter information.
        """
        # This is basically the same as the non-numba reshape function,
        # but it's not as pretty.
        x_sl = slice(None, None, 2)
        y_sl = slice(1, None, 2)
        for i, ai in enumerate(antenna_numbers):
            for j, aj in enumerate(antenna_numbers[i:]):
                j += i
                uvd_inds = (ant_1_array == ai) & (ant_2_array == aj)

                flipped = False
                ii, jj = i, j
                if np.all(~uvd_inds):
                    uvd_inds = (ant_2_array == ai) & (ant_1_array == aj)
                    flipped = True
                    ii, jj = j, i

                # Don't do anything if this baseline isn't present.
                if np.all(~uvd_inds):
                    continue

                uvd_inds = np.argwhere(uvd_inds).flatten()
                for k, pol in enumerate(pol_array):
                    if pol == -5:
                        p1, p2 = x_sl, x_sl
                    elif pol == -6:
                        p1, p2 = y_sl, y_sl
                    elif pol == -7:
                        p1, p2 = x_sl, y_sl
                    else:
                        p1, p2 = y_sl, x_sl

                    if flipped:
                        p1, p2 = p2, p1

                    _p = out[:, :, p1, p2]
                    for tidx, uvd_ind in enumerate(uvd_inds):
                        _p[tidx, :, ii, jj] = vis[uvd_ind, :, k]
                        _p[tidx, :, jj, ii] = np.conj(vis[uvd_ind, :, k])
        return out

    @numba.njit
    def jit_reshape_vis_invert(
        vis, out, ant_1_array, ant_2_array, pol_array, antenna_numbers
    ):
        """JIT-accelerated reshaping function.

        See :func:`~reshape_vis` for parameter information.
        """
        # This is basically the same as the non-numba reshape function,
        # but it's not as pretty.
        x_sl = slice(None, None, 2)
        y_sl = slice(1, None, 2)
        for i, ai in enumerate(antenna_numbers):
            for j, aj in enumerate(antenna_numbers[i:]):
                j += i
                uvd_inds = (ant_1_array == ai) & (ant_2_array == aj)

                flipped = False
                ii, jj = i, j
                if np.all(~uvd_inds):
                    uvd_inds = (ant_2_array == ai) & (ant_1_array == aj)
                    flipped = True
                    ii, jj = j, i

                # Don't do anything if this baseline isn't present.
                if np.all(~uvd_inds):
                    continue

                uvd_inds = np.argwhere(uvd_inds).flatten()
                for k, pol in enumerate(pol_array):
                    if pol == -5:
                        p1, p2 = x_sl, x_sl
                    elif pol == -6:
                        p1, p2 = y_sl, y_sl
                    elif pol == -7:
                        p1, p2 = x_sl, y_sl
                    else:
                        p1, p2 = y_sl, x_sl

                    if flipped:
                        p1, p2 = p2, p1

                    # NOTE: This is hard-coded to use new-style UVData arrays!
                    # Go back to UVData shape
                    _p = vis[:, :, p1, p2]
                    for tidx, uvd_ind in enumerate(uvd_inds):
                        out[uvd_ind, :, k] = _p[tidx, :, ii, jj]
                        tidx += 1
        return out

    @numba.njit
    def _left_matmul(left, right):
        """JIT-accelerated matrix multiplication.

        This multiply assumes the zeroth axis of the ``left`` array is length 1.
        """
        out = np.zeros_like(right)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = left[0, j] @ right[i, j]
        return out

    @numba.njit
    def _right_matmul(left, right):
        """JIT-accelerated matrix multiplication.

        This multiply assumes the zeroth axis of the ``right`` array is length 1.
        """
        out = np.zeros_like(left)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = left[i, j] @ right[0, j]
        return out

    @numba.njit
    def _matmul(left, right):
        """JIT-accelerated matrix multiplication.

        This multiply assumes both arrays have the same shape. It should only
        provide a speedup over ``numpy``'s matrix multiplication for cases where
        the first two axes of the input arrays are much larger than the last two
        axes.
        """
        out = np.zeros_like(left)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = left[i, j] @ right[i, j]
        return out
