"""Object-oriented approach to signal chain systematics."""

import numpy as np
import warnings
from typing import Dict, Tuple, Union

from scipy import stats
from scipy.signal import blackmanharris

from . import interpolators
from . import utils
from .components import component
from . import DATA_PATH
from .defaults import _defaults


@component
class Gain:
    """Base class for systematic gains."""

    pass


class Bandpass(Gain, is_multiplicative=True):
    """Generate bandpass gains.

    Parameters
    ----------
    gain_spread : float, optional
        Standard deviation of random gains.
    dly_rng : tuple, optional
        Lower and upper range of delays which are uniformly sampled.
    bp_poly : callable or array_like, optional
        If an array, polynomial coefficients to evaluate. Otherwise, a function
        of frequency that can be evaluated to generate real numbers giving
        the bandpass gain.
    """

    _alias = ("gains", "bandpass_gain")

    def __init__(self, gain_spread=0.1, dly_rng=(-20, 20), bp_poly=None):
        super().__init__(gain_spread=gain_spread, dly_rng=dly_rng, bp_poly=bp_poly)

    def __call__(self, freqs, ants, **kwargs):
        """Generate the bandpass.

        Parameters
        ----------
        freqs : array_like of float
            Frequencies in GHz.
        ants : array_like of int
            Antenna numbers for which to produce gains.

        Returns
        -------
        dict
            Keys are antenna numbers and values are arrays of bandpass
            gains as a function of frequency.
        """
        # validate kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (gain_spread, dly_rng, bp_poly) = self._extract_kwarg_values(**kwargs)

        # get the bandpass gains
        bandpass = self._gen_bandpass(freqs, ants, gain_spread, bp_poly)

        # get the delay phases
        phase = self._gen_delay_phase(freqs, ants, dly_rng)

        return {ant: bandpass[ant] * phase[ant] for ant in ants}

    @_defaults
    def _gen_bandpass(self, freqs, ants, gain_spread=0.1, bp_poly=None):
        if bp_poly is None:
            # default to the H1C bandpass
            bp_poly = np.load(DATA_PATH / "HERA_H1C_BANDPASS.npy")
        elif isinstance(bp_poly, str):
            # make an interpolation object, assume it's a polyfit
            bp_poly = interpolators.Bandpass(bp_poly)
        if callable(bp_poly):
            # support for interpolation objects
            bp_base = bp_poly(freqs)
        else:
            bp_base = np.polyval(bp_poly, freqs)
        window = blackmanharris(freqs.size)
        modes = np.abs(np.fft.fft(window * bp_base))
        gains = {}
        for ant in ants:
            delta_bp = np.fft.ifft(
                utils.gen_white_noise(freqs.size) * modes * gain_spread
            )
            gains[ant] = bp_base + delta_bp
        return gains

    def _gen_delay_phase(self, freqs, ants, dly_rng=(-20, 20)):
        phases = {}
        for ant in ants:
            delay = np.random.uniform(*dly_rng)
            phases[ant] = np.exp(2j * np.pi * delay * freqs)
        return phases


class Reflections(Gain, is_multiplicative=True):
    """Produce multiplicative reflection gains.

    Parameters
    ----------
    amp : float, optional
        Mean Amplitude of the reflection gains.
    dly : float, optional
        Mean delay of the reflection gains.
    phs : float, optional
        Phase of the reflection gains.
    conj : bool, optional
        Whether to conjugate the gain.
    amp_jitter : float, optional
        Final amplitudes are multiplied by a normal variable with mean one, and
        with standard deviation of ``amp_jitter``.
    dly_jitter : float, optional
        Final delays are offset by a normal variable with mean
        zero and standard deviation ``dly_jitter``.
    """

    _alias = ("reflection_gains", "sigchain_reflections")

    def __init__(
        self, amp=None, dly=None, phs=None, conj=False, amp_jitter=0, dly_jitter=0
    ):
        super().__init__(
            amp=amp, dly=dly, phs=phs, conj=conj, amp_jitter=0, dly_jitter=0
        )

    def __call__(self, freqs, ants, **kwargs):
        """Generate the bandpass.

        Parameters
        ----------
        freqs : array_like of float
            Frequencies in units inverse to :attr:`dly`.
        ants : array_like of int
            Antenna numbers for which to produce gains.

        Returns
        -------
        dict
            Keys are antenna numbers and values are arrays of bandpass
            gains.
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        amp, dly, phs, conj, amp_jitter, dly_jitter = self._extract_kwarg_values(
            **kwargs
        )

        # fill in missing kwargs
        amp, dly, phs = self._complete_params(
            ants, amp, dly, phs, amp_jitter, dly_jitter
        )

        # determine gains iteratively
        gains = {}
        for j, ant in enumerate(ants):
            # calculate the reflection coefficient
            eps = self.gen_reflection_coefficient(
                freqs, amp[j], dly[j], phs[j], conj=conj
            )
            gains[ant] = 1 + eps

        return gains

    @staticmethod
    def gen_reflection_coefficient(freqs, amp, dly, phs, conj=False):
        """Randomly generate reflection coefficients.

        Parameters
        ----------
        freqs : array_like of float
            Frequencies, units are arbitrary but must be the inverse of ``dly``.
        amp : array_like of float
            Either a scalar amplitude, or 1D with size Nfreqs, or 2D
            with shape (Ntimes, Nfreqs).
        dly : [type]
            Either a scalar delay, or 1D with size Nfreqs, or 2D
            with shape (Ntimes, Nfreqs). Units are inverse of ``freqs``.
        phs : [type]
            Either a scalar phase, or 1D with size Nfreqs, or 2D
            with shape (Ntimes, Nfreqs). Units radians.
        conj : bool, optional
            Whether to conjugate the gain.

        Returns
        -------
        array_like
            The reflection gains as a 2D array of (Ntimes, Nfreqs).
        """
        # this is copied directly from the old sigchain module
        # TODO: make this cleaner

        # helper function for checking type/shape
        def _type_check(arr):
            if isinstance(arr, np.ndarray):
                if arr.ndim == 1 and arr.size > 1:
                    # reshape the array to (Ntimes, 1)
                    arr = arr.reshape(-1, 1)
                    # raise a warning if it's the same length as freqs
                    if arr.shape[0] == Nfreqs:
                        warnings.warn(
                            "The input array had lengths Nfreqs "
                            "and is being reshaped as (Ntimes,1)."
                        )
                elif arr.ndim > 1:
                    assert arr.shape[1] in (1, Nfreqs), (
                        "Frequency-dependent reflection coefficients must "
                        "match the input frequency array size."
                    )
            return arr

        Nfreqs = freqs.size
        amp = _type_check(amp)
        dly = _type_check(dly)
        phs = _type_check(phs)

        # actually make the reflection coefficient
        eps = amp * np.exp(1j * (2 * np.pi * freqs * dly + phs))

        # conjugate if desired
        return np.conj(eps) if conj else eps

    @staticmethod
    def _complete_params(
        ants, amp=None, dly=None, phs=None, amp_jitter=0, dly_jitter=0
    ):
        """
        Generate parameters to calculate a reflection coefficient.

        Parameters
        ----------
        ants: iterable
            Iterable providing information about antenna numbers. Only used to
            determine how many entries each parameter needs to have.
        amp: float or length-2 array-like of float, optional
            If a single number is provided, then every antenna is assigned that
            number as the amplitude of the reflection. Otherwise, it should
            specify the lower and upper bounds, respectively, of the uniform
            distribution from which to randomly assign an amplitude for each
            antenna. Default is to randomly choose a number between 0 and 1.
        dly: float or length-2 array-like of float
            If a single number provided, then the reflection shows up at that
            delay for every antenna. Otherwise, it should specify the lower and
            upper bounds, respectively, of the uniform distribution from which
            to randomly assign delays. This should be specified in units of ns.
            Default is to randomly choose a delay between -20 and 20 ns.
        phs: float or length-2 array-like of float
            The phase of the reflection, or the bounds to use for assigning
            random phases. Default is to randomly choose a phase on [-pi, pi).
        amp_jitter: float, optional
            Standard deviation of multiplicative jitter to apply to amplitudes.
            For example, setting this to 1e-4 will introduce, on average, 0.01%
            deviations to each amplitude. Default is to not add any jitter.
        dly_jitter: float, optional
            Standard deviation of additive jitter to apply to delays, in ns.
            For example, setting this to 10 will introduce, on average, delay
            deviations up to 10 ns. (This is drawn from a normal distribution, so
            it is possible that delays will exceed the value provided.)

        Returns
        -------
        amps: array-like of float
            Amplitude of reflection coefficient for each antenna.
        dlys: array-like of float
            Delay of each reflection coefficient, in ns, for each antenna.
        phases: array-like of float
            Phase of each reflection coefficient for each antenna.
        """

        def broadcast_param(param, lower_bound, upper_bound, size):
            if param is None:
                return stats.uniform.rvs(lower_bound, upper_bound, size)
            elif np.isscalar(param):
                return np.ones(size, dtype=float) * param
            else:
                if len(param) == size:
                    return np.array(param, dtype=float)
                else:
                    return stats.uniform.rvs(*param, size)

        # Transform parameters into arrays.
        amps = broadcast_param(amp, 0, 1, len(ants))
        dlys = broadcast_param(dly, -20, 20, len(ants))
        phases = broadcast_param(phs, -np.pi, np.pi, len(ants))

        # Apply jitter.
        amps *= stats.norm.rvs(1, amp_jitter, len(ants))
        dlys += stats.norm.rvs(0, dly_jitter, len(ants))

        return amps, dlys, phases


@component
class Crosstalk:
    """Base class for cross-talk models."""

    pass


class CrossCouplingCrosstalk(Crosstalk, Reflections):
    """Generate cross-coupling xtalk.

    Parameters
    ----------
    amp : float, optional
        Mean Amplitude of the reflection gains.
    dly : float, optional
        Mean delay of the reflection gains.
    phs : float, optional
        Phase of the reflection gains.
    conj : bool, optional
        Whether to conjugate the gain.
    amp_jitter : float, optional
        Final amplitudes are multiplied by a normal variable with mean one, and
        with standard deviation of ``amp_jitter``.
    dly_jitter : float, optional
        Final delays are offset by a normal variable with mean
        zero and standard deviation ``dly_jitter``.
    """

    _alias = ("cross_coupling_xtalk",)

    def __init__(
        self, amp=None, dly=None, phs=None, conj=False, amp_jitter=0, dly_jitter=0
    ):
        super().__init__(
            amp=amp, dly=dly, phs=phs, conj=conj, amp_jitter=0, dly_jitter=0
        )

    def __call__(self, freqs, autovis, **kwargs):
        """Copute the cross-correlations.

        Parameters
        ----------
        freqs : array_like of float
            Frequencies in units inverse to :attr:`dly`.
        autovis : array_like of float
            The autocorrelations as a function of frequency.

        Return
        ------
        array
            The cross-coupling contribution to the visibility,
            same shape as ``freqs``.
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # now unpack them
        amp, dly, phs, conj, amp_jitter, dly_jitter = self._extract_kwarg_values(
            **kwargs
        )

        # handle the amplitude, phase, and delay
        amp, dly, phs = self._complete_params(
            [1], amp, dly, phs, amp_jitter, dly_jitter
        )

        # Make reflection coefficient.
        eps = self.gen_reflection_coefficient(freqs, amp, dly, phs, conj=conj)

        # reshape if necessary
        if eps.ndim == 1:
            eps = eps.reshape((1, -1))

        # scale it by the autocorrelation and return the result
        return autovis * eps


class CrossCouplingSpectrum(Crosstalk):
    """Generate a cross-coupling spectrum.

    This generates multiple copies of :class:`CrossCouplingCrosstalk`
    into the visibilities.

    Parameters
    ----------
    Ncopies : int, optional
        Number of random cross-talk models to add.
    amp_range : tuple, optional
        Two-tuple of floats specifying the range of amplitudes
        to be sampled regularly in log-space.
    dly_range : tuple, optional
        Two-tuple of floats specifying the range of delays to be
        sampled at regular intervals.
    phs_range : tuple, optional
        Range of uniformly random phases.
    amp_jitter : int, optional
        Standard deviation of random jitter to be applied to the
        regular amplitudes.
    dly_jitter : int, optional
        Standard deviation of the random jitter to be applied to
        the regular delays.
    symmetrize : bool, optional
        Whether to also produce statistically equivalent cross-talk at
        negative delays. Note that while the statistics are equivalent,
        both amplitudes and delays will be different random realizations.
    """

    _alias = ("cross_coupling_spectrum", "xtalk_spectrum")

    def __init__(
        self,
        Ncopies=10,
        amp_range=(-4, -6),
        dly_range=(1000, 1200),
        phs_range=(-np.pi, np.pi),
        amp_jitter=0,
        dly_jitter=0,
        symmetrize=True,
    ):
        super().__init__(
            Ncopies=Ncopies,
            amp_range=amp_range,
            dly_range=dly_range,
            phs_range=phs_range,
            amp_jitter=amp_jitter,
            dly_jitter=dly_jitter,
            symmetrize=symmetrize,
        )

    def __call__(self, freqs, autovis, **kwargs):
        """Copute the cross-correlations.

        Parameters
        ----------
        freqs : array_like of float
            Frequencies in units inverse to :attr:`dly`.
        autovis : array_like of float
            The autocorrelations as a function of frequency.

        Return
        ------
        array
            The cross-coupling contribution to the visibility,
            same shape as ``freqs``.
        """
        self._check_kwargs(**kwargs)

        (
            Ncopies,
            amp_range,
            dly_range,
            phs_range,
            amp_jitter,
            dly_jitter,
            symmetrize,
        ) = self._extract_kwarg_values(**kwargs)

        # Construct the arrays of amplitudes and delays.
        amps = np.logspace(*amp_range, Ncopies)
        dlys = np.linspace(*dly_range, Ncopies)

        # Construct the spectrum of crosstalk.
        crosstalk_spectrum = np.zeros(autovis.shape, dtype=complex)
        for amp, dly in zip(amps, dlys):
            gen_xtalk = CrossCouplingCrosstalk(
                amp=amp,
                dly=dly,
                phs=phs_range,
                amp_jitter=amp_jitter,
                dly_jitter=dly_jitter,
            )

            crosstalk_spectrum += gen_xtalk(freqs, autovis)
            if symmetrize:
                # Note: this will have neither the same jitter realization nor
                # the same phase as the first crosstalk spectrum.
                crosstalk_spectrum += gen_xtalk(freqs, autovis, dly=-dly)

        return crosstalk_spectrum


class WhiteNoiseCrosstalk(Crosstalk):
    """Generate cross-talk that is simply white noise.

    Parameters
    ----------
    amplitude : float, optional
        The amplitude of the white noise spectrum (i.e. its standard deviation).
    """

    _alias = (
        "whitenoise_xtalk",
        "white_noise_xtalk",
    )

    def __init__(self, amplitude=3.0):
        super().__init__(amplitude=amplitude)

    def __call__(self, freqs, **kwargs):
        """Compute the cross-correlations.

        Parameters
        ----------
        freqs : array_like of float
            Frequencies in units inverse to :attr:`dly`.

        Return
        ------
        array
            The cross-coupling contribution to the visibility,
            same shape as ``freqs``.
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (amplitude,) = self._extract_kwarg_values(**kwargs)

        # why choose this size for the convolving kernel?
        kernel = np.ones(50 if freqs.size > 50 else int(freqs.size / 2))

        # generate the crosstalk
        xtalk = np.convolve(utils.gen_white_noise(freqs.size), kernel, "same")

        # scale the result and return
        return amplitude * xtalk


def apply_gains(
    vis: Union[float, np.ndarray],
    gains: Dict[int, Union[float, np.ndarray]],
    bl: Tuple[int, int],
) -> np.ndarray:
    """Apply antenna-based gains to a visibility.

    Parameters
    ----------
    vis
        The visibilities of the given baseline as a function of frequency.
    gains
        Dictionary where keys are antenna numbers and values are arrays of
        gains as a function of frequency.
    bl
        2-tuple of integers specifying the antenna numbers in the particular
        baseline.

    Returns
    -------
    vis
        The visibilities with gains applied.
    """
    # get the gains for each antenna in the baseline
    # don't apply a gain if the antenna isn't found
    gi = 1.0 if bl[0] not in gains else gains[bl[0]]
    gj = 1.0 if bl[1] not in gains else gains[bl[1]]

    # if neither antenna is in the gains dict, do nothing
    if bl[0] not in gains and bl[1] not in gains:
        return vis

    # form the gain term for the given baseline
    gain = gi * np.conj(gj)

    # reshape if need be
    if gain.ndim == 1:
        gain.shape = (1, -1)

    return vis * gain


def vary_gains_in_time(
    gains,
    times,
    freqs=None,
    delays=None,
    parameter="amp",
    variation_ref_time=None,
    variation_timescale=None,
    variation_amp=0.05,
    variation_mode="linear",
):
    r"""
    Vary gain amplitudes, phases, or delays in time.

    Notes
    -----
    If the gains initially have the form

    :math:`g(\\nu) = g_0(\\nu)\\exp(i2\\pi\\nu\\tau + i\\phi),`

    then the output gains have the form

    :math:`g(\\nu,t) = g_0(\\nu,t)\\exp\\bigl(i2\\pi\\nu\\tau(t) + i\\phi(t)\\bigr).`


    Parameters
    ----------
    gains: dict
        Dictionary mapping antenna numbers to gain spectra/waterfalls.
    times: array-like of float
        Times at which to simulate time variation. Should be the same length as
        the data to which the gains will be applied. Should also be in the same
        units as ``variation_ref_time`` and ``variation_timescale``.
    freqs: array-like of float, optional
        Frequencies at which the gains are evaluated, in GHz. Only needs to be
        specified for adding time variation to the delays.
    delays: dict, optional
        Dictionary mapping antenna numbers to gain delays, in ns.
    parameter: str, optional
        Which gain parameter to vary; must be one of ("amp", "phs", "dly").
    variation_ref_time: float or array-like of float, optional
        Reference time(s) used for generating time variation. For linear and
        sinusoidal variation, this is the time where the gains are equal to their
        original, time-independent values. Should be in the same units as the
        ``times`` array. Default is to use the center of the ``times`` provided.
    variation_timescale: float or array-like of float, optional
        Timescale(s) for one cycle of the variation(s), in the same units as
        the provided ``times``. Default is to use the duration of the entire
        ``times`` array.
    variation_amp: float or array-like of float, optional
        Amplitude(s) of the variation(s) introduced. This is *not* the peak-to-peak
        amplitude! This also does not have exactly the same interpretation for each
        type of variation mode. For amplitude and delay variation, this represents
        the amplitude of modulations--so it can be interpreted as a fractional
        variation. For phase variation, this represents an absolute, time-dependent
        phase offset to introduce to the gains; however, it is still *not* a
        peak-to-peak amplitude.
    variation_mode: str or array-like of str, optional
        Which type(s) of variation to simulate. Supported modes are "linear",
        "sinusoidal", and "noiselike". Default is "linear". Note that the "linear"
        mode produces a triangle wave variation with period twice the corresponding
        timescale; this ensures that the gains vary linearly over the entire set of
        provided times if the default variation timescale is used.

    Returns
    -------
    time_varied_gains: dict
        Dictionary mapping antenna numbers to gain waterfalls.
    """
    # Parameter checking/preparation.
    if np.isscalar(times) or not np.isrealobj(times):
        raise TypeError("times must be an array of real numbers.")
    if not isinstance(gains, dict):
        raise TypeError("gains must be provided as a dictionary.")
    if parameter not in ("amp", "phs", "dly"):
        raise ValueError("parameter must be one of 'amp', 'phs', or 'dly'.")

    times = np.array(times)
    gain_shapes = [np.array(gain).shape for gain in gains.values()]
    if any(gain_shape != gain_shapes[0] for gain_shape in gain_shapes):
        raise ValueError("Gains must all have the same shape.")
    gain_shape = gain_shapes[0]

    if parameter == "dly":
        if freqs is None or delays is None:
            raise ValueError(
                "In order to vary delays, you must provide both the corresponding "
                "frequency array and a dictionary mapping antenna numbers to delays."
            )

        freqs = np.array(freqs)
        if set(delays.keys()) != set(gains.keys()):
            raise ValueError("Delays and gains must have the same keys.")
        if len(gain_shape) == 2:
            if gain_shape != (times.size, freqs.size):
                raise ValueError("Gain waterfalls must have shape (Ntimes, Nfreqs).")
        elif len(gain_shape) == 1:
            if gain_shape[0] != freqs.size:
                raise ValueError(
                    "Gain spectra must be the same length as the provided frequencies."
                )
        else:
            raise ValueError("Gain dictionary values must be at most 2-dimensional.")

    # Setup for handling multiple modes of variation.
    if variation_ref_time is None:
        variation_ref_time = (np.median(times),)
    if variation_timescale is None:
        variation_timescale = (times[-1] - times[0],)
        if utils._listify(variation_mode)[0] == "linear":
            variation_timescale = (variation_timescale[0] * 2,)
    variation_ref_time = utils._listify(variation_ref_time)
    variation_timescale = utils._listify(variation_timescale)
    variation_amp = utils._listify(variation_amp)
    variation_mode = utils._listify(variation_mode)
    variation_settings = (
        variation_mode,
        variation_amp,
        variation_ref_time,
        variation_timescale,
    )

    # Check that everything is the same length.
    Nmodes = len(variation_mode)
    if any(len(settings) != Nmodes for settings in variation_settings):
        raise ValueError(
            "At least one of the variation settings does not have the same "
            "number of entries as the number of variation modes specified."
        )

    # Now generate a multiplicative envelope to use for applying time variation.
    iterator = zip(
        variation_mode, variation_amp, variation_ref_time, variation_timescale
    )
    envelope = 1
    for mode, amp, ref_time, timescale in iterator:
        phases = ((times - ref_time) / timescale) % 1  # Map times to [0, 1)
        if mode == "linear":
            phases = (phases + 0.25) % 1  # Shift left a quarter period.
            # Map phases to [-1, 1].
            response = np.where(phases <= 0.5, 4 * phases - 1, 3 - 4 * phases)
            envelope *= 1 + amp * response
        elif mode == "sinusoidal":
            envelope *= 1 + amp * np.sin(2 * np.pi * phases)
        elif mode == "noiselike":
            envelope *= stats.norm.rvs(1, amp, times.size)
        else:
            raise NotImplementedError(f"Variation mode '{mode}' not supported.")

    if parameter in ("amp", "phs"):
        envelope = np.outer(envelope, np.ones(gain_shape[-1]))
        if parameter == "phs":
            envelope = np.exp(1j * (envelope - 1))
        gains = {ant: np.atleast_2d(gain) * envelope for ant, gain in gains.items()}
    else:
        envelope = 2 * np.pi * np.outer(envelope - 1, freqs)
        gains = {
            ant: np.atleast_2d(gain) * np.exp(1j * delays[ant] * envelope)
            for ant, gain in gains.items()
        }

    return gains


# to minimize breaking changes
gen_gains = Bandpass()
gen_bandpass = gen_gains._gen_bandpass
gen_delay_phs = gen_gains._gen_delay_phase

gen_reflection_coefficient = Reflections.gen_reflection_coefficient
gen_reflection_gains = Reflections()

gen_whitenoise_xtalk = WhiteNoiseCrosstalk()
gen_cross_coupling_xtalk = CrossCouplingCrosstalk()
