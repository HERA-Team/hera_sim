"""EoR from an object-oriented approach."""

from .components import ComponentBase
from abc import abstractmethod
from cached_property import cached_property

class EoR(ComponentBase):
    def __init_subclass__(cls, is_abstract=False, **kwargs):
        print("Calling EoR __init_subclass__ routine.")
        print("is_abstract is: {}".format(is_abstract))
        print("kwargs are: {}".format(kwargs))
        super().__init_subclass__(is_abstract=is_abstract, **kwargs)

    def __init__(self, **kwargs):
        print("calling EoR initializor.")
        super().__init__(**kwargs)

    @staticmethod
    def models():
        _models = {}
        for subclass in EoR.__subclasses__():
            name = subclass.__name__
            _models[name] = ComponentBase._models[name]
        return _models

class NoiselikeEoR(EoR):
    def __init__(self, **kwargs):
        print("calling NoiselikeEoR initializor.")
        super().__init__(**kwargs)

class Test(EoR):
    def __init__(self, is_abstract=True):
        print("calling Test initializor.")
        super().__init__(is_abstract=is_abstract)

print("Creating noiselike_eor instance of NoiselikeEoR.")
noiselike_eor = NoiselikeEoR()
print("Creating test instance of Test.")
test = Test()
