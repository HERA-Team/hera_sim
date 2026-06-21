"""
A sub-package dedicated to defining various visibility simulators.

The focus is on simulating visibilities from a known sky-locked signal. These
include simulators of varying levels of accuracy and performance capabilities,
as well as varying expected input types -- eg. floating point sources,
temperature fields, rectilinear co-ordinates, spherical co-ordinates, healpix
maps etc. This package intends to unify the interfaces of these various kinds
of simulators.
"""

from .fftvis import HAVE_FFTVIS, FFTVis

# Always import these, because we need them for docs to compile.
from .matvis import HAVE_MATVIS, MatVis
from .pyuvsim_wrapper import UVSim
from .simulators import (
    ModelData,
    VisibilitySimulation,
    VisibilitySimulator,
    load_simulator_from_yaml,
)

# Registered Simulators
SIMULATORS = {"UVSim": UVSim}

if HAVE_MATVIS:
    SIMULATORS["MatVis"] = MatVis
if HAVE_FFTVIS:
    SIMULATORS["FFTVis"] = FFTVis
