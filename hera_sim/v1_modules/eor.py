"""EoR from an object-oriented approach."""

from .components import track 
from abc import abstractmethod
from cached_property import cached_property

@track
class EoR:
    def __init__(self, **kwargs):
        print("calling EoR initializor.")
        super().__init__(**kwargs)

class NoiselikeEoR(EoR):
    def __init__(self, **kwargs):
        print("calling NoiselikeEoR initializor.")
        super().__init__(**kwargs)

class Test(EoR, is_abstract=True):
    def __init__(self):
        print("calling Test initializor.")
        super().__init__()

print("Creating noiselike_eor instance of NoiselikeEoR.")
noiselike_eor = NoiselikeEoR()
print("Creating test instance of Test.")
test = Test()
