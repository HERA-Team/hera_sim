"""Object-oriented approach to signal chain systematics."""

from .components import registry


@registry
class Gain:

    def apply(self, vis):
        pass

    pass

class Bandpass(Gain, is_multiplicative=True):
    __aliases__ = ("gen_gains", "bandpass_gain")

    def __call__(self):
        pass

class Reflections(Gain, is_multiplicative=True):
    __aliases__ = ("gen_sigchain_reflections", "sigchain_reflections")

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
