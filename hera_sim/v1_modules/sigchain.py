"""Object-oriented approach to signal chain systematics."""

import numpy as np
from abc import abstractmethod

from . import utils
from .components import registry

import aipy

@registry
class Gain:
    # TODO: docstring
    pass

class Bandpass(Gain, is_multiplicative=True):
    __aliases__ = ("gen_gains", "bandpass_gain")

    def __init__(self, gain_spread=0.1, dly_rng=(-20,20), bp_poly=None):
        # TODO: docstring
        """

        """
        super().__init__(
            gain_spread=gain_spread,
            dly_rng=dly_rng,
            bp_poly=bp_poly)

    def __call__(self, freqs, ants, **kwargs):
        # TODO: docstring
        """
        """
        # validate kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (gain_spread, dly_rng, 
            bp_poly) = self._extract_kwarg_values(**kwargs)

        # get the bandpass gains
        bandpass = self._gen_bandpass(freqs, ants, gain_spread, bp_poly)

        # get the delay phases
        phase = self._gen_delay_phase(freqs, ants, dly_rng)

        return {ant : bandpass[ant] * phase[ant] for ant in ants}

    def _gen_bandpass(self, freqs, ants, gain_spread=0.1, bp_poly=None):
        if bp_poly is None:
            # figure out how to deal with this
            pass
        bp_base = np.polyval(bp_poly, freqs)
        window = aipy.dsp.gen_window(freqs.size, "blackman-harris")
        modes = np.abs(np.fft.fft(window * bp_base))
        gains = {}
        for ant in ants:
            delta_bp = np.fft.ifft(utils.white_noise(freqs.size)
                                   * modes * gain_spread)
            gains[ant] = bp_base + delta_bp
        return gains

    def _gen_delay_phase(self, freqs, ants, dly_rng=(-20,20)):
        phases = {}
        for ant in ants:
            delay = np.random.uniform(*dly_rng)
            phases[ant] = np.exp(2j* np.pi * delay * freqs)
        return phases

class Reflections(Gain, is_multiplicative=True):
    __aliases__ = ("gen_reflection_gains", "sigchain_reflections")

    def __init__(self, amp=None, dly=None, phs=None, 
                       conj=False, randomize=False):
        # TODO: docstring
        """
        """
        super().__init__(
            amp=amp,
            dly=dly,
            phs=phs,
            conj=conj,
            randomize=randomize)

    def __call__(self, freqs, ants, **kwargs):
        # TODO: docstring
        """
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (amp, dly, phs, conj, 
            randomize) = self._extract_kwarg_values(**kwargs)

        # fill in missing kwargs
        amp, dly, phs = self._complete_params(ants, amp, dly, phs, randomize)

        # determine gains iteratively
        gains = {}
        for j, ant in enumerate(ants):
            # calculate the reflection coefficient
            eps = self.gen_reflection_coefficient(freqs, amp[j], dly[j],
                                                  phs[j], conj=conj)
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
                        warnings.warn("The input array had lengths Nfreqs "
                                      "and is being reshaped as (Ntimes,1).")
                elif arr.ndim > 1:
                    assert arr.shape[1] in (1, Nfreqs), \
                        "Frequency-dependent reflection coefficients must " \
                        "match the input frequency array size."
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
            assert all([isinstance(param, (list, tuple))
                        and len(param) == 2
                        for param in (amp, dly, phs)]), \
                "You have chosen to randomize the amplitude, delay, " \
                "and phase parameters, but at least one parameter " \
                "was not specified as None or a length-2 tuple or " \
                "list. Please check your parameter settings."

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

class CrossCouplingCrosstalk(Reflections, Crosstalk):
    __aliases__ = ("gen_cross_coupling_xtalk", "cross_coupling_xtalk")

    def __init__(self, amp=None, dly=None, phs=None, 
                       conj=False, randomize=False):
        # TODO: docstring
        """
        """
        super().__init__(
            amp=amp,
            dly=dly,
            phs=phs,
            conj=conj,
            randomize=randomize)


    def __call__(self, freqs, autovis, **kwargs):
        # TODO: docstring
        """
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # now unpack them
        (amp, dly, phs, conj, 
            randomize) = self._unpack_kwarg_values(**kwargs)

        # handle the amplitude, phase, and delay
        amp, dly, phs = self._complete_params([1], amp, dly, phs, randomize)

        # make a reflection coefficient
        eps = self.gen_reflection_coefficient(freqs, amp, dly, phs, conj=conj)

        # reshape if necessary
        if eps.ndim == 1:
            eps = eps.reshape((1,-1))

        # scale it by the autocorrelation and return the result
        return autovis * eps

class WhiteNoiseCrosstalk(Crosstalk):
    __aliases__ = ("gen_whitenoise_xtalk", "white_noise_xtalk", )

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
        amplitude = self._unpack_kwargs(**kwargs)

        # why choose this size for the convolving kernel?
        kernel = np.ones(50 if freqs.size > 50 else int(freqs.size/2))

        # generate the crosstalk
        xtalk = np.convolve(utils.white_noise(freqs.size), kernel, "same")

        # scale the result and return
        return amplitude * xtalk

# to minimize breaking changes
gen_gains = Bandpass()
gen_reflection_coefficient = Reflections.gen_reflection_coefficient
gen_reflection_gains = Reflections()
gen_whitenoise_xtalk = WhiteNoiseCrosstalk()
gen_cross_coupling_xtalk = CrossCouplingCrosstalk()
