"""Re-imagining of the simulation module."""

import functools
import inspect
import os
import sys
import warnings
import yaml
import time

import numpy as np
from cached_property import cached_property
from pyuvdata import UVData
from astropy import constants as const

from . import io
from .version import version
from .components import SimulationComponent

class Simulator:
    """Class for managing a simulation.

    """
    def __init__(self, data=None, uvdata_kwargs={}, 
                       default_config=None, default_kwargs={},
                       **kwargs):
        """Initialize a Simulator object.

        Idea: Make Simulator object have three major components:
            sim.data -> UVData object for storing the "measured" data
                Also keep track of most metadata here
            sim.defaults -> Defaults object

        """
        self._initialize_uvd(data, **uvdata_kwargs)
        # might not actually want to handle defaults this way
        self.defaults = Defaults(default_config, **default_kwargs)
        self._components = {}
        self.extras = {}


    def _initialize_uvd(self, data, **uvdata_kwargs):
        if data is None:
            self.data = io.empty_uvdata(**uvdata_kwargs)
        elif isinstance(data, str):
            self._read_datafile(data)
            self.extras['data_file'] = data
        elif isinstance(data, UVData):
            self.data = data
        else:
            raise ValueError("Unsupported type.") # make msg better

    def _read_datafile(self, datafile):
        pass

    def add(self, component, conserve_memory=False, **kwargs):
        # search for the appropriate class
        # make an instance of the class w/ appropriate parameters
        # calculate the effect
        # log it in the components dictionary if not conserving memory
        # there might be a trick to get around this:
        # just log the parameters used to generate the component, then
        # re-calculate the component in the ``get`` class method
        # check if it's multiplicative
        # add it to the data array appropriately
        pass

    def get(self, component):
        # find which component to return
        # return the component
        pass

    def write(self, save_format="uvh5", save_seeds=True):
        pass

    def run_sim(self, config):
        pass

