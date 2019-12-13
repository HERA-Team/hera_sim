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
from .defaults import defaults
from .version import version
from .components import SimulationComponent

# wrapper for the run_sim method, necessary for part of the CLI
def _generator_to_list(func, *args, **kwargs):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        result = list(func(*args, **kwargs))
        return None if result == [] else result
    return new_func

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
        uvdata_kwargs.update(kwargs)
        self._initialize_uvd(data, **uvdata_kwargs)
        self._components = {}
        self.extras = {}
        self.seeds = {}
        # apply and activate defaults if specified
        if default_config or default_kwargs:
            self.apply_defaults(default_config, **default_kwargs)

    @cached_property
    def antpos(self):
        # TODO: docstring
        """
        """
        antpos, ants = self.data.get_ENU_antpos(pick_data_ants=True)
        return dict(zip(ants, antpos))

    @cached_property
    def lsts(self):
        # TODO: docstring
        return np.unique(self.data.lst_array)

    @cached_property
    def freqs(self):
        # TODO: docstring
        """Frequencies in GHz
        """
        return np.unique(self.data.freq_array) / 1e9

    def apply_defaults(self, default_config, refresh=True, **default_kwargs):
        # TODO: docstring
        """
        """
        # ensure that only one of these is set
        assert bool(default_config) ^ bool(default_kwargs), \
            "If you wish to use a default configuration, please " \
            "only specify *either* a path to a configuration file " \
            "*or* a default parameter dictionary. You are seeing " \
            "this message because you specified both."

        # actually apply the default settings
        if default_config:
            defaults.set(default_config, refresh=refresh)
        else:
            defaults.set(**default_kwargs, refresh=refresh)

    def _initialize_uvd(self, data, **uvdata_kwargs):
        # TODO: docstring
        """
        """
        if data is None:
            self.data = io.empty_uvdata(**uvdata_kwargs)
        elif isinstance(data, str):
            self.data = self._read_datafile(data, **uvdata_kwargs)
            self.extras['data_file'] = data
        elif isinstance(data, UVData):
            self.data = data
        else:
            raise ValueError("Unsupported type.") # make msg better

    def _initialize_args_from_model(self, model):
        # TODO: docstring
        """
        """
        model_params = inspect.signature(model).parameters
        # pull the lst and frequency arrays as required
        args = list(getattr(self, param) for param in model_params
                    if param in ("lsts", "freqs"))
        # for antenna-based gains
        requires_ants = any([param.startswith("ant")
                             for param in model_params])
        # for sky components
        requires_bl_vec = any([param.startswith("bl") 
                               for param in model_params])
        # for cross-coupling xtalk
        requires_vis = any([param.find("vis") != -1
                            for param in model_params])
        return (args, requires_ants, requires_bl_vec, requires_vis)

    def _iterate_antpair_pols(self):
        # TODO: docstring
        """
        """
        for ant1, ant2, pol in self.data.get_antpairpols():
            blt_inds = self.data.antpair2ind((ant1, ant2))
            pol_ind = self.data.get_pols().index(pol)
            yield ant1, ant2, pol, blt_inds, pol_ind

    def _iteratively_apply(self, model, add_vis=True, ret_vis=False, **kwargs):
        # TODO: docstring
        """
        """
        # do nothing if neither adding nor returning the effect
        if not add_vis and not ret_vis:
            return
        # pull lsts/freqs if required and find out which extra 
        # parameters are required
        (args, requires_ants, requires_bl_vec, 
            requires_vis) = self._initialize_args_from_model(model)
        # figure out whether or not to seed the RNG
        seed_mode = kwargs.pop("seed_mode", None)
        # get the original data array just in case
        initial_data = self.data.data_array.copy()
        # do we really want to do it this way?
        for ant1, ant2, pol, blt_inds, pol_ind in self._iterate_antpair_pols():
            use_args = self._update_args(
                args, requires_ants, requires_bl_vec, requires_vis,
                ant1, ant2, pol
            )
            # determine whether or not to seed the RNG(s) used in 
            # simulating the model effects
            if seed_mode is not None:
                self._seed_rng(seed_mode, model, ant1, ant2)
            # check whether we're simulating a gain or a visibility
            if model.is_multiplicative:
                # get the gains for the entire array
                # this is sloppy, but ensures seeding works correctly
                gains = model(*use_args, **kwargs)
                # now get the product g_1g_2*
                gain = gains[ant1] * np.conj(gains[ant2])
                # apply the effect to the appropriate part of the data
                self.data.data_array[blt_inds, 0, :, pol_ind] *= gain
            else:
                # if it's not multiplicative, then it should be an 
                # actual visibility, so calculate it
                vis = model(*use_args, **kwargs)
                # and add it in
                self.data.data_array[blt_inds, 0, :, pol_ind] += vis

        # return the component if desired
        if ret_vis:
            # return the gain dictionary if gains are simulated
            if model.is_multiplicative:
                return gains
            # otherwise return the actual visibility simulated
            else:
                return self.data.data_array - initial_data
        
        # reset the data array if not adding the component
        if not add_vis:
            self.data.data_array = initial_data


    @staticmethod
    def _read_datafile(datafile, **kwargs):
        # TODO: docstring
        """
        """
        uvd = UVData()
        uvd.read(datafile, read_data=True, **kwargs)
        return uvd

    def _seed_rng(self, seed_mode, model, ant1=None, ant2=None):
        # TODO: docstring
        """
        """
        if seed_mode == "redundantly":
            assert ant1 is not None and ant2 is not None
            # generate seeds for each redundant group
            # this does nothing if the seeds already exist
            self._generate_redundant_seeds(model)
            # get the baseline integer for baseline (ant1, ant2)
            bl_int = self.data.antnums_to_baseline(ant1, ant2)
            # find out which redundant group the baseline is in
            key = [bl_int in reds 
                   for reds in self._get_reds()].index(True)
            # seed the RNG accordingly
            np.random.seed(self._get_seed(model, key))
        elif seed_mode == "once":
            # this should only be used for antenna-based gains
            # where it's most convenient to just seed the RNG 
            # once for the whole array
            np.random.seed(self._get_seed(model, 0))
        else:
            raise ValueError("Seeding mode not supported.")

    def _update_args(self, args, requires_ants, requires_bl_vec,
                     requires_vis, ant1=None, ant2=None, pol=None):
        # TODO: docstring
        """
        """
        # check if this is an antenna-dependent quantity; should
        # only ever be true for gains (barring future changes)
        if requires_ants:
            ants = self.antpos
            use_args = args + [ants]
        # check if this is something requiring a baseline vector
        # current assumption is that these methods require the
        # baseline vector to be provided in nanoseconds
        elif requires_bl_vec:
            bl_vec = self.antpos[ant1] - self.antpos[ant2]
            bl_vec_ns = bl_vec * 1e9 / const.c.value
            use_args = args + [bl_vec_ns]
        # check if this is something that depends on another
        # visibility. as of now, this should only be cross coupling
        # crosstalk
        elif requires_vis:
            autovis = self.data.get_data(ant1, ant1, pol)
            use_args = args + [autovis]
        else:
            use_args = args.copy()
        return use_args

    @staticmethod
    def _get_component(component):
        # TODO: docstring
        """
        """
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
                msg = "The component '{component}' wasn't found. The "
                msg += "following aliases are known: "
                msg += ", ".join(set(all_aliases))
                msg = msg.format(component=component)
                raise AttributeError(msg)

    def _generate_seed(self, model, key):
        # TODO: docstring
        """
        """
        model = self._get_model_name(model)
        # for the sake of randomness
        np.random.seed(int(time.time()))
        if model not in self.seeds:
            self.seeds[model] = {}
        self.seeds[model][key] = np.random.randint(2**32)

    def _generate_redundant_seeds(self, model):
        # TODO: docstring
        """
        """
        model = self._get_model_name(model)
        if model in self.seeds:
            return
        for j in range(len(self._get_reds())):
            self._generate_seed(model, j)

    def _get_reds(self):
        # TODO: docstring
        """
        """
        return self.data.get_baseline_redundancies()[0]

    def _get_seed(self, model, key):
        # TODO: docstring
        """
        """
        model = self._get_model_name(model)
        if model not in self.seeds:
            self._generate_seed(model, key)
        if key not in self.seeds[model]:
            self._generate_seed(model, key)
        return self.seeds[model][key]
    
    @staticmethod
    def _get_model_name(model):
        # TODO: docstring
        """
        """
        if isinstance(model, str):
            return model
        try:
            return model.__name__
        except AttributeError:
            # check if it's a user defined function
            if model.__class__.__name__ == "function":
                # get the name of it if so
                try:
                    func_name = [name for name, obj in globals().items()
                                      if id(obj) == id(model)][0]
                    return func_name
                except IndexError:
                    # it's not in the global namespace
                    msg = "The model is a function but is not in the "
                    msg += "global namespace. Please import the "
                    msg += "function and try again."
                    raise ValueError(msg)
            else:
                return model.__class__.__name__

    def _sanity_check(self, model):
        # TODO: docstring
        """
        """
        has_data = not np.all(self.data.data_array == 0)
        is_multiplicative = model.is_multiplicative
        contains_multiplicative_effect = any([
                self._get_component(component)[0].is_multiplicative
                for component in self._components])
        if is_multiplicative and not has_data:
            warnings.warn("You are trying to compute a multiplicative "
                          "effect, but no visibilities have been "
                          "simulated yet.")
        elif not is_multiplicative and contains_multiplicative_effect:
            warnings.warn("You are adding visibilities to a data array "
                          "*after* multiplicative effects have been "
                          "introduced.")

    def _update_history(self, model, **kwargs):
        # TODO: docstring
        """
        """
        model = self._get_model_name(model)
        msg = "hera_sim v{version}: Added {component} using kwargs:\n"
        if defaults._override_defaults:
            kwargs["defaults"] = defaults._config_name
        for param, value in kwargs.items():
            msg += "{param} = {value}\n".format(param=param, value=value)
        msg = msg.format(version=version, component=model)
        self.data.history += msg

    def add(self, component, **kwargs):
        # TODO: docstring
        """
        """
        # find out whether to add and/or return the component
        add_vis = kwargs.pop("add_vis", True)
        ret_vis = kwargs.pop("ret_vis", False)
        # take out the seed_mode kwarg so as not to break initializor
        seed_mode = kwargs.pop("seed_mode", -1)
        # get the model for the desired component
        model, is_class = self._get_component(component)
        if is_class:
            # if the component returned is a class, instantiate it
            model = model(**kwargs)
        # check that there isn't an issue with component ordering
        self._sanity_check(model)
        # re-add the seed_mode kwarg if it was specified
        if seed_mode != -1:
            kwargs["seed_mode"] = seed_mode
        # calculate the effect
        data = self._iteratively_apply(
            model, add_vis=add_vis, ret_vis=ret_vis, **kwargs
        )
        # log the component and its kwargs, if added to data
        if add_vis:
            self._components[component] = kwargs
            # update the history
            self._update_history(model, **kwargs)
        # return the data if desired
        if ret_vis:
            return data

    def get(self, component, ant1=None, ant2=None, pol=None):
        # TODO: docstring
        """
        """
        assert component in self._components.keys()
        # retrieve the model
        model, is_class = self._get_component(component)
        # get the kwargs
        kwargs = self._components[component]
        # figure out whether or not to seed the rng
        seed_mode = kwargs.pop("seed_mode", None)
        # instantiate the model if it's a class
        if is_class:
            model = model(**kwargs)
        # seed the RNG if desired
        if seed_mode is not None:
            self._seed_rng(seed_mode, model, ant1, ant2)
        # get the arguments necessary for the model
        init_args = self._initialize_args_from_model(model)
        use_args = self._update_args(*init_args, ant1, ant2, pol)
        # now calculate the effect and return it
        return model(*use_args, **kwargs)

    def write(self, filename, save_format="uvh5", save_seeds=True, **kwargs):
        # TODO: docstring
        """
        """
        try:
            getattr(self.data, "write_%s" % save_format)(filename, **kwargs)
        except AttributeError:
            msg = "The save_format must correspond to a write method in UVData."
            raise ValueError(msg)
        if save_seeds:
            seed_file = os.path.splitext(filename)[0] + "_seeds"
            np.save(seed_file, self.seeds)
    
    @_generator_to_list
    def run_sim(self, sim_file=None, **sim_params):
        # TODO: docstring
        """
        """
        # make sure that only sim_file or sim_params are specified
        assert bool(sim_file) ^ bool(sim_params), \
            "Either an absolute path to a simulation configuration " \
            "file or a dictionary of simulation parameters may be " \
            "passed, but not both. Please only pass one of the two."

        # read the simulation file if provided
        if sim_file is not None:
            with open(sim_file, 'r') as config:
                try:
                    sim_params = yaml.load(
                        config.read(), Loader=yaml.FullLoader
                    )
                except:
                    print("The configuration file was not able to be loaded.")
                    print("Please fix the file and try again.")
                    sys.exit()

        # loop over the entries in the configuration dictionary
        for component, params in sim_params.items():
            # make sure that the parameters are a dictionary
            assert isinstance(params, dict), \
                "The parameters for {component} are not formatted " \
                "properly. Please ensure that the parameters for " \
                "each component are specified using a " \
                "dictionary.".format(component=component)
            
            # add the component to the data
            value = self.add(component, **params)
        
            # if the user wanted to return the data, then
            if value is not None:
                yield (component, value)
