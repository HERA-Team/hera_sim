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
        self.seeds = {}


    def _initialize_uvd(self, data, **uvdata_kwargs):
        # TODO: docstring
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
        try:
            if issubclass(component, SimulationComponent):
                # support passing user-defined classes that inherit from
                # the SimulationComponent base class to add method
                return component, True
        except TypeError:
            # this is raised if ``component`` is not a class
            if callable(component):
                # if it's callable, then it's either a user-defined 
                # function or a class instance
                return component, False
            else:
                assert isinstance(component, str), \
                        "``component`` must be either a class which " \
                        "derives from ``SimulationComponent`` or an " \
                        "instance of a callable class, or a function, " \
                        "whose signature is:\n" \
                        "func(lsts, freqs, *args, **kwargs)\n" \
                        "If it is none of the above, then it must be " \
                        "a string which corresponds to the name of a " \
                        "``hera_sim`` class or an alias thereof."
                # keep track of all known aliases in case desired 
                # component isn't found in the search
                all_aliases = []
                for registry in SimulationComponent.__subclasses__():
                    for model in registry.__subclasses__():
                        aliases = (model.__name__,)
                        aliases += getattr(model, "__aliases__", ())
                        aliases = [alias.lower() for alias in aliases]
                        for alias in aliases:
                            all_aliases.append(alias)
                        if component.lower() in aliases:
                            return model, True
                # if this part is executed, then the model wasn't found, so
                msg = "The component {component} wasn't found. The following "
                msg += "aliases are known: "
                msg += ", ".join(set(all_aliases))
                msg = msg.format(component=component)
                raise AttributeError(msg)

    def _generate_seeds(self, model):
        pass

    def _get_seed(self, model, *ants):
        # TODO: docstring
        if not isinstance(model, str):
            try:
                model = model.__name__
            except AttributeError:
                # in case it's an instance
                model = model.__class__.__name__
        
    def _sanity_check(self, model):
        # TODO: docstring
        has_data = not np.all(self.data.data_array == 0)
        is_multiplicative = model.is_multiplicative
        contains_multiplicative_effect = any([
                self._get_component(component).is_multiplicative
                for component in self._components])
        if is_multiplicative and not has_data:
            warnings.warn("You are trying to compute a multiplicative "
                          "effect, but no visibilities have been "
                          "simulated yet.")
        elif not is_multiplicative and contains_multiplicative_effect:
            warnings.warn("You are adding visibilities to a data array "
                          "*after* multiplicative effects have been "
                          "introduced.")

    def add(self, component, **kwargs):
        # log the component and its kwargs
        self._components[component] = kwargs
        # find out if the component should have its RNG be seeded
        seed_model = {key : value for key, value in kwargs.items()
                                  if key.startswith("seed")}
        if seed_model:
            for key in seed_model:
                kwargs.pop(key)
        # get the model for the desired component
        model, is_class = self._get_component(component)
        if is_class:
            # if the component returned is a class, instantiate it
            model = model(**kwargs)
        # check that there isn't an issue with component ordering
        self._sanity_check(model)
        # calculate the effect
        self._iteratively_apply(model, **seed_model)
        self._update_history(model, **kwargs)

    def get(self, component):
        assert component in self._components.keys()
        model, _ = self._get_component(component)(**self._components[component])
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
            np.save(seed_file, self.seeds)

    def run_sim(self, config):
        pass

