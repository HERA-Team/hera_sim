"""Make some noise."""

from .components import registry

@registry
class Noise:
    pass

class ThermalNoise(Noise):
    pass

