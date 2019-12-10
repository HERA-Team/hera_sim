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

    def __call__(self):
        pass

@registry
class Crosstalk:
    pass

class CrossCouplingCrosstalk(Crosstalk):
    __aliases__ = ("gen_cross_coupling_xtalk", "cross_coupling_xtalk")

    def __call__(self):
        pass

class WhiteNoiseCrosstalk(Crosstalk):
    __aliases__ = ("gen_whitenoise_xtalk", "white_noise_xtalk", )

    def __call__(self):
        pass

gen_gains = Bandpass()
gen_sigchain_reflections = Reflections()
gen_whitenoise_xtalk = WhiteNoiseCrosstalk()
gen_cross_coupling_xtalk = CrossCouplingCrosstalk()
