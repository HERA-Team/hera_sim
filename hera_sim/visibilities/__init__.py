"""
A sub-package dedicated to defining various visibility simulators, where the
focus is on simulating visibilities from a known sky-locked signal. These
include simulators of varying levels of accuracy and performance capabilities,
as well as varying expected input types -- eg. floating point sources,
temperature fields, rectilinear co-ordinates, spherical co-ordinates, healpix
maps etc. This package intends to unify the interfaces of these various kinds
of simulators.
"""
from .simulators import *
from .conversions import *
