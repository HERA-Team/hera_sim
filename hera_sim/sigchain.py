"""Object-oriented approach to signal chain systematics."""

import os
import numpy as np
import warnings

from scipy import stats

from . import interpolators
from . import utils
from .components import registry
from . import DATA_PATH
from .defaults import _defaults

import aipy


@registry
class Gain:
    # TODO: docstring
    pass


class Bandpass(Gain, is_multiplicative=True):
    _alias = ("gains", "bandpass_gain")

    def __init__(self, gain_spread=0.1, dly_rng=(-20, 20), bp_poly=None):
        # TODO: docstring
        """

        """
        super().__init__(gain_spread=gain_spread, dly_rng=dly_rng, bp_poly=bp_poly)

    def __call__(self, freqs, ants, **kwargs):
        # TODO: docstring
        """
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
        window = aipy.dsp.gen_window(freqs.size, "blackman-harris")
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
    _alias = ("reflection_gains", "sigchain_reflections")

    def __init__(self, amp=None, dly=None, phs=None, conj=False, randomize=False):
        # TODO: docstring
        """
        """
        super().__init__(amp=amp, dly=dly, phs=phs, conj=conj, randomize=randomize)

    def __call__(self, freqs, ants, **kwargs):
        # TODO: docstring
        """
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (amp, dly, phs, conj, randomize) = self._extract_kwarg_values(**kwargs)

        # fill in missing kwargs
        amp, dly, phs = self._complete_params(ants, amp, dly, phs, randomize)

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
        # TODO: docstring
        """
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
    def _complete_params(ants, amp, dly, phs, randomize):
        # TODO: docstring
        """
        """
        # if we're randomizing, then amp, dly, phs should define which
        # bounds to use for making random numbers
        if randomize:
            # convert these to bounds if they're None
            if amp is None:
                amp = (0, 1)
            if dly is None:
                dly = (-20, 20)
            if phs is None:
                phs = (-np.pi, np.pi)
            # now make sure that they're all correctly formatted
            assert all(
                [
                    isinstance(param, (list, tuple)) and len(param) == 2
                    for param in (amp, dly, phs)
                ]
            ), (
                "You have chosen to randomize the amplitude, delay, "
                "and phase parameters, but at least one parameter "
                "was not specified as None or a length-2 tuple or "
                "list. Please check your parameter settings."
            )

            # in the future, expand this to allow for freq-dependence?
            # randomly generate the parameters
            amp = [np.random.uniform(amp[0], amp[1]) for ant in ants]
            dly = [np.random.uniform(dly[0], dly[1]) for ant in ants]
            phs = [np.random.uniform(phs[0], phs[1]) for ant in ants]
        else:
            # set the amplitude to 1, delay and phase both to zero
            if amp is None:
                amp = [1.0 for ant in ants]
            if dly is None:
                dly = [0.0 for ant in ants]
            if phs is None:
                phs = [0.0 for ant in ants]

        return amp, dly, phs


@registry
class Crosstalk:
    pass


class CrossCouplingCrosstalk(Crosstalk, Reflections):
    _alias = ("cross_coupling_xtalk",)

    def __init__(self, amp=None, dly=None, phs=None, conj=False, randomize=False):
        # TODO: docstring
        """
        """
        super().__init__(amp=amp, dly=dly, phs=phs, conj=conj, randomize=randomize)

    def __call__(self, freqs, autovis, **kwargs):
        # TODO: docstring
        """
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # now unpack them
        (amp, dly, phs, conj, randomize) = self._extract_kwarg_values(**kwargs)

        # handle the amplitude, phase, and delay
        amp, dly, phs = self._complete_params([1], amp, dly, phs, randomize)

        # make a reflection coefficient
        eps = self.gen_reflection_coefficient(freqs, amp, dly, phs, conj=conj)

        # reshape if necessary
        if eps.ndim == 1:
            eps = eps.reshape((1, -1))

        # scale it by the autocorrelation and return the result
        return autovis * eps


class WhiteNoiseCrosstalk(Crosstalk):
    _alias = (
        "whitenoise_xtalk",
        "white_noise_xtalk",
    )

    def __init__(self, amplitude=3.0):
        # TODO: docstring
        """
        """
        super().__init__(amplitude=amplitude)

    def __call__(self, freqs, **kwargs):
        # TODO: docstring
        """
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


def apply_gains(vis, gains, bl):
    # TODO: docstring
    """
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
    variation_ref_times=None,
    variation_timescales=None,
    variation_amps=(0.05,),
    variation_modes=("linear",),
):
    """
    Vary gain amplitudes, phases, or delays in time.

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
        units as ``variation_ref_times`` and ``variation_timescales``.
    freqs: array-like of float, optional
        Frequencies at which the gains are evaluated, in GHz. Only needs to be
        specified for adding time variation to the delays.
    delays: dict, optional
        Dictionary mapping antenna numbers to gain delays, in ns.
    parameter: str, optional
        Which gain parameter to vary; must be one of ("amp", "phs", "dly").
    variation_ref_times: float or array-like of float, optional
        Reference time(s) used for generating time variation. For linear and
        sinusoidal variation, this is the time where the gains are equal to their
        original, time-independent values. Should be in the same units as the
        ``times`` array. Default is to use the center of the ``times`` provided.
    variation_timescales: float or array-like of float, optional
        Timescale(s) for one cycle of the variation(s), in the same units as
        the provided ``times``. Default is to use the duration of the entire
        ``times`` array.
    variation_amps: float or array-like of float, optional
        Amplitude(s) of the variation(s) introduced. This is *not* the peak-to-peak
        amplitude! This also does not have exactly the same interpretation for each
        type of variation mode. For amplitude and delay variation, this represents
        the amplitude of modulations--so it can be interpreted as a fractional
        variation. For phase variation, this represents an absolute, time-dependent
        phase offset to introduce to the gains; however, it is still *not* a
        peak-to-peak amplitude.
    variation_modes: str or array-like of str, optional
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
    gain_shape = np.array(list(gains.values())[0]).shape

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
    if variation_ref_times is None:
        variation_ref_times = (np.median(times),)
    if variation_timescales is None:
        variation_timescales = (times[-1] - times[0],)
        if utils._listify(variation_modes)[0] == "linear":
            variation_timescales = (variation_timescales[0] * 2,)
    variation_ref_times = utils._listify(variation_ref_times)
    variation_timescales = utils._listify(variation_timescales)
    variation_amps = utils._listify(variation_amps)
    variation_modes = utils._listify(variation_modes)
    variation_settings = (
        variation_modes,
        variation_amps,
        variation_ref_times,
        variation_timescales,
    )

    # Check that everything is the same length.
    Nmodes = len(variation_modes)
    if any(len(settings) != Nmodes for settings in variation_settings):
        raise ValueError(
            "At least one of the variation settings does not have the same "
            "number of entries as the number of variation modes specified."
        )

    # Now generate a multiplicative envelope to use for applying time variation.
    iterator = zip(
        variation_modes, variation_amps, variation_ref_times, variation_timescales
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
