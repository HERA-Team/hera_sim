"""
A sub-package dedicated to defining various visibility simulators, where the
focus is on simulating visibilities from a known sky-locked signal. These
include simulators of varying levels of accuracy and performance capabilities,
as well as varying expected input types -- eg. floating point sources,
temperature fields, rectilinear co-ordinates, spherical co-ordinates, healpix
maps etc. This package intends to unify the interfaces of these various kinds
of simulators.
"""
from .simulators import VisibilitySimulator
from .conversions import *

import warnings

# Registered Simulators
from .vis_cpu import VisCPU


try:
    from .healvis_wrapper import HealVis
except (ImportError, NameError):
    warnings.warn("HealVis failed to import.")

try:
    from .prisim_wrapper import PRISim
except ImportError:
    warnings.warn("PRISim failed to import.")

# GPU version of VisCPU
try:
    from .vis_gpu import VisGPU
except ImportError:
    warnings.warn("VisGPU failed to import.")
