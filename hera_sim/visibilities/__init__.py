"""
A sub-package dedicated to defining various visibility simulators.

The focus is on simulating visibilities from a known sky-locked signal. These
include simulators of varying levels of accuracy and performance capabilities,
as well as varying expected input types -- eg. floating point sources,
temperature fields, rectilinear co-ordinates, spherical co-ordinates, healpix
maps etc. This package intends to unify the interfaces of these various kinds
of simulators.
"""

from .pyuvsim_wrapper import UVSim
from .simulators import (
    ModelData,
    VisibilitySimulation,
    VisibilitySimulator,
    load_simulator_from_yaml,
)

# Registered Simulators
SIMULATORS = {"UVSim": UVSim}

try:
    from .matvis import MatVis

    SIMULATORS["MatVis"] = MatVis
except (ImportError, NameError):  # pragma: no cover
    pass

try:
    from .fftvis import FFTVis

    SIMULATORS["FFTVis"] = FFTVis
except (ImportError, NameError):  # pragma: no cover
    pass
