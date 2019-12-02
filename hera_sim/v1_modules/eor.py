"""EoR from an object-oriented approach."""

from .components import registry 
from abc import abstractmethod
from cached_property import cached_property

@registry
class EoR:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class NoiselikeEoR(EoR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

noiselike_eor = NoiselikeEoR()
