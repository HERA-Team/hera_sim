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
            self.data = self._read_datafile(data, **uvdata_kwargs)
            self.extras['data_file'] = data
        elif isinstance(data, UVData):
            self.data = data
        else:
            raise ValueError("Unsupported type.") # make msg better

    @staticmethod
    def _read_datafile(datafile, **kwargs):
        # TODO: docstring
        uvd = UVData()
        uvd.read(datafile, read_data=True, **kwargs)
        return uvd

    @staticmethod
    def _get_component(component):
        # TODO: docstring
        for registry in SimulationComponent.__subclasses__():
            for model in registry.__subclasses__():
                aliases = [model.__name__,]
                aliases += list(getattr(model, "__aliases__", []))
                aliases = [alias.lower() for alias in aliases]
                if component.lower() in aliases:
                    return model

    def add(self, component, conserve_memory=False, **kwargs):
        self._components[component] = kwargs
        model = self._get_component(component)(**kwargs)
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
        assert component in self._components.keys()
        model = self._get_component(component)(**self._components[component])
        pass

    def write(self, filename, save_format="uvh5", save_seeds=True, **kwargs):
        # TODO: docstring
        try:
            getattr(self.data, "write_%s" % save_format)(filename, **kwargs)
        except AttributeError:
            msg = "The save_format must correspond to a write method in UVData."
            raise ValueError(msg)
        if save_seeds:
            seed_file = os.path.splitext(filename)[0] + "_seeds"
            np.save(seed_file, self.extras.get("seeds", None))

    def run_sim(self, config):
        pass

