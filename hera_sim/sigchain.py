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

    def __init__(
        self, amp=None, dly=None, phs=None, conj=False, amp_jitter=0, dly_jitter=0
    ):
        # TODO: docstring
        """
        """
        super().__init__(
            amp=amp, dly=dly, phs=phs, conj=conj, amp_jitter=0, dly_jitter=0
        )

    def __call__(self, freqs, ants, **kwargs):
        # TODO: docstring
        """
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
                return np.ones(size, dtype=np.float) * param
            else:
                if len(param) == size:
                    return np.array(param, dtype=np.float)
                else:
                    return stats.uniform.rvs(*param, size)

        # Transform parameters into arrays.
        amps = broadcast_param(amp, 0, 1, len(ants))
        dlys = broadcast_param(dly, -20, 20, len(ants))
        phases = broadcast_param(phs, -np.pi, np.pi, len(ants))

        # Apply jitter.
        amps *= stats.norm.rvs(1, amp_jitter, len(ants))
        dlys += stats.norm.rvs(0, dly_jitter, len(ants))

        return amp, dly, phases


@registry
class Crosstalk:
    pass


class CrossCouplingCrosstalk(Crosstalk, Reflections):
    _alias = ("cross_coupling_xtalk",)

    def __init__(
        self, amp=None, dly=None, phs=None, conj=False, amp_jitter=0, dly_jitter=0
    ):
        # TODO: docstring
        """
        """
        super().__init__(
            amp=amp, dly=dly, phs=phs, conj=conj, amp_jitter=0, dly_jitter=0
        )

    def __call__(self, freqs, autovis, **kwargs):
        # TODO: docstring
        """
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

        # Make reflection coefficient; enforce symmetry in delay.
        eps_1 = self.gen_reflection_coefficient(freqs, amp, dly, phs, conj=conj)
        eps_2 = self.gen_reflection_coefficient(freqs, amp, -dly, phs, conj=conj)
        eps = eps_1 + eps_2

        # reshape if necessary
        if eps.ndim == 1:
            eps = eps.reshape((1, -1))

        # scale it by the autocorrelation and return the result
        return autovis * eps


class CrossCouplingSpectrum(Crosstalk):
    _alias = ("cross_coupling_spectrum", "xtalk_spectrum")

    def __init__(
        self,
        Ncopies=10,
        amp_range=(-4, -6),
        dly_range=(1000, 1200),
        phs_range=(-np.pi, np.pi),
        amp_jitter=0,
        dly_jitter=0,
    ):
        super().__init__(
            Ncopies=Ncopies,
            amp_range=amp_range,
            dly_range=dly_range,
            phs_range=phs_range,
            amp_jitter=amp_jitter,
            dly_jitter=dly_jitter,
        )

    def __call__(self, freqs, autovis, **kwargs):
        # TODO: docstring
        """
        """
        self._check_kwargs(**kwargs)

        (
            Ncopies,
            amp_range,
            dly_range,
            phs_range,
            amp_jitter,
            dly_jitter,
        ) = self._extract_kwarg_values(**kwargs)

        # Construct the arrays of amplitudes and delays.
        amps = np.logspace(*amp_range, Ncopies)
        dlys = np.linspace(*dly_range, Ncopies)

        # Construct the spectrum of crosstalk.
        crosstalk_spectrum = np.zeros(autovis.shape, dtype=np.complex)
        for amp, dly in zip(amps, dlys):
            gen_xtalk = CrossCouplingCrosstalk(
                amp=amp,
                dly=dly,
                phs=phs_range,
                amp_jitter=amp_jitter,
                dly_jitter=dly_jitter,
            )

            crosstalk_spectrum += gen_xtalk(freqs, autovis)

        return crosstalk_spectrum


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


# to minimize breaking changes
gen_gains = Bandpass()
gen_bandpass = gen_gains._gen_bandpass
gen_delay_phs = gen_gains._gen_delay_phase

gen_reflection_coefficient = Reflections.gen_reflection_coefficient
gen_reflection_gains = Reflections()

gen_whitenoise_xtalk = WhiteNoiseCrosstalk()
gen_cross_coupling_xtalk = CrossCouplingCrosstalk()
