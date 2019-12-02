"""Object-oriented approach to signal chain systematics."""

from .components import registry


@registry
class Gain:

    def apply(self, vis):
        pass

    pass

class Bandpass(Gain, is_multiplicative=True):
    pass

class Reflections(Gain, is_multiplicative=True):
    pass

class Crosstalk(Gain, is_abstract=True):
    pass

class CrossCouplingCrosstalk(Crosstalk):
    pass

class WhiteNoiseCrosstalk(Crosstalk):
    pass

gen_gains = Bandpass()
gen_sigchain_reflections = Reflections()
gen_whitenoise_xtalk = WhiteNoiseCrosstalk()
gen_cross_coupling_xtalk = CrossCouplingCrosstalk()
