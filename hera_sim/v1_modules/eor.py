"""EoR from an object-oriented approach."""

from .components import registry 
from abc import abstractmethod
from cached_property import cached_property

@registry
class EoR:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class NoiselikeEoR(EoR):
    def __init__(self, eor_amp=1e-5, 
                 min_delay=None, max_delay=None, 
                 fringe_filter_type="tophat",
                 fringe_filter_kwargs={}, **kwargs):
        super().__init__(**kwargs)
    # TODO: docstrings

    def __call__(self, lsts, freqs, bl_vec, **kwargs):
        pass


noiselike_eor = NoiselikeEoR()
