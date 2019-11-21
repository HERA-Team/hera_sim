"""
This module is designed to allow for easy interfacing with simulation default
parameters in an interactive environment.
"""

import yaml
import inspect
import functools
import sys
import warnings

from os import path
from .config import CONFIG_PATH

SEASON_CONFIGS = {'h1c': path.join(CONFIG_PATH, 'HERA_H1C_CONFIG.yaml'),
                  'h2c': path.join(CONFIG_PATH, 'HERA_H2C_CONFIG.yaml'),
                  }

class Defaults:
    """Class for dynamically changing hera_sim parameter defaults.

    This class handles the retreival of simulation default parameters from
    YAML files and the ability to switch the default settings while in an
    interactive environment. This class is intended to exist as a singleton;
    as such, an instance is created at the end of this module, and that
    instance is what is imported in the hera_sim constructor. See below
    for example usage within hera_sim.

    Examples
    --------
    To set the default parameters to those appropriate for the H2C
    observing season (and activate the use of those defaults):
        
    hera_sim.defaults.set('h2c')

    To set the defaults to a custom set of defaults, you must first
    create a configuration YAML. Assuming the path to the YAML is
    stored in the variable `config_path`, these defaults would be set
    via the following line:

    hera_sim.defaults.set(config_path)

    To revert back to using defaults defined in function signatures:

    hera_sim.defaults.deactivate()

    To view what the default value is for a particular parameter, do: 

    hera_sim.defaults(parameter),

    where `parameter` is a string with the name of the parameter as 
    listed in the configuration file. To view the entire set of default 
    parameters, use:

    hera_sim.defaults()
    """

    def __init__(self, config='h1c'):
        """Load in a configuration and check its formatting.

        Parameters
        ----------
        
        config : str or dict, optional (default 'h1c') 
            May either be an absolute path to a configuration YAML, one of
            the observing season keywords ('h1c', 'h2c'), or a dictionary 
            with the appropriate format.

        Notes
        -----
        
        The configuration file may be formatted in practically any way, 
        as long as it is parsable by `pyyaml`. That said, the resulting 
        configuration will *always* take the form {param : value} for 
        every item (param, value) such that `value` is not a dict. A 
        consequence of this is that any parameters whose names are not 
        unique will take on the value specified last in the config. The 
        raw configuration is kept in memory, but it currently is not 
        used for overriding any default values.

        Examples
        --------

        Consider the following contents of a configuration file:

        foregrounds:
            Tsky_mdl: !Tsky
                datafile: HERA_Tsky_Reformatted.npz
            seed_redundantly: True
            nsrcs: 500
        gains:
            gain_spread: 0.1
            dly_rng: [-10, 10]
            bp_poly: HERA_H1C_BANDPASS.npy

        This would result in the following set of defaults:

        {Tsky_mdl: <hera_sim.interpolators.Tsky instance>,
         seed_redundantly: True,
         nsrcs: 500, 
         gain_spread: 0.1,
         dly_rng: [-10,10]
         bp_poly: HERA_H1C_BANDPASS.npy
         }

        Now consider a different configuration file:

        sky:
            eor:
                eor_amp: 0.001
        systematics:
            rfi:
                rfi_stations:
                    stations: !!null
                rfi_impulse:
                    chance: 0.01
                rfi_scatter:
                    chance: 0.35
            crosstalk:
                amplitude: 1.25
            gains:
                gain_spread: 0.2
            noise:
                Trx: 150

        Since the parser recursively unpacks the raw configuration 
        dictionary until no entry is nested, the resulting config is:

        {eor_amp: 0.001,
         stations: None,
         chance: 0.35,
         amplitude: 1.25,
         gain_spread: 0.2,
         Trx: 150
         }
        """
        self._raw_config = {}
        self._config = {}
        self._config_name = None
        self._warn = False # remove this when season configs are updated
        self._set_config(config)
        self._warn = True # remove this when season configs are updated
        self._override_defaults = False

    def __call__(self, component=None):
        """Return the defaults dictionary, or just a component."""
        if component is not None:
            try:
                return self._config[component]
            except KeyError:
                raise KeyError("{} not found in configuration.".format(component))
        else:
            return self._config

    def set(self, new_config, refresh=False):
        """Set the defaults to those specified in `new_config`.

        Parameters
        ----------
        new_config : str or dict
            Absolute path to configuration file or dictionary of 
            configuration parameters formatted in the same way a 
            configuration would be loaded.

        refresh : bool, optional
            Choose whether to completely overwrite the old config or 
            just add new values to it.

        Notes
        -----
        Calling this method also activates the defaults.
        """
        if refresh:
            self._config = {}
        self._set_config(new_config)
        self.activate()

    def activate(self):
        """Activate the defaults."""
        self._override_defaults = True

    def deactivate(self):
        """Revert to function defaults."""
        self._override_defaults = False

    def _set_config(self, config):
        """Retrieve the configuration specified."""
        if isinstance(config, str):
            # set the name of the configuration used
            self._config_name = config
            # retrieve the season configuration file if appropriate
            if config in SEASON_CONFIGS.keys():
                self._config_name = config
                config = SEASON_CONFIGS[config]
            # load in the raw configuration
            with open(config, 'r') as conf:
                self._raw_config = yaml.load(conf.read(), Loader=yaml.FullLoader)
        elif isinstance(config, dict):
            # set the raw configuration dictionary to config
            self._raw_config = config
            # unpack the raw configuration dictionary
            self._config = self._unpack_dict(self._raw_config, self._config)
            # note that a custom configuration is used
            self._config_name = "custom"
        else:
            raise ValueError(
                    "The configuration must be a dictionary, an absolute " \
                    "path to a configuration YAML, or a season keyword." )
        
        # unpack the raw configuration dictionary
        self._config = self._unpack_dict(self._raw_config, self._config)
        # check if any items are repeated
        self._check_config()

    def _unpack_dict(self, nested_dict, new_dict):
        """Extract individual components from a (partially) nested dictionary.
        
        Parameters
        ----------
        nested_dict : dict
            A dictionary that may either be fully, partially, or not 
            nested. May have any degree of nesting.

        new_dict : dict
            A dictionary, empty or not, to fill with the (key, value) 
            pairs in `nested_dict` such that `value` is not a dict.

        Returns
        -------
        new_dict : dict
            The fully unpacked dictionary of (key, value) pairs from
            `nested_dict`. No values in this dictionary will be 
            dictionaries themselves.

        Examples
        --------
        Input: nested_dict = {key1 : {k1 : v1, k2 : v2}, key2 : val2}
               new_dict = {}
        Output: new_dict = {k1 : v1, k2 : v2, key2 : val2}

        Input: nested_dict = {key1 : val1, key2 : val2}
               new_dict = {key0 : val0}
        Output: new_dict = {key0 : val0, key1 : val1, key2 : val2}
        """
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                self._unpack_dict(value, new_dict)
            else:
                new_dict[key] = value
        return new_dict

    def _recursive_enumerate(self, counts, values, dictionary):
        """Recursively enumerate the entries in `dictionary`."""
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self._recursive_enumerate(counts, values, value)
            else:
                counts[key] += 1
                values[key].append(value)

    def _check_config(self):
        """Check and warn if any keys in the configuration are repeated."""
        # initialize dictionaries that enumerate the key, value pairs
        # in the raw configuration dictionary
        counts = {key : 0 for key in self().keys()}
        values = {key : [] for key in self().keys()}
        
        # actually do the enumeration
        self._recursive_enumerate(counts, values, self._raw_config)

        # flag any parameters that show up more than once
        flags = {key : (1 if count > 1 else 0) for key, count in counts.items()}
        # warn the user if any configuration parameters are repeated
        if any(flags.values()):
            warning = "The following parameters have multiple values defined " \
                      "in the configuration:\n"
            for param, flag in flags.items():
                if flag:
                    warning += "{}\n".format(param)
            warning += "Please check your configuration, as only the last " \
                       "value specified for each parameter will be used."
            if self._warn: # remove this after season defaults update
                warnings.warn(warning)

    @property
    def _version_is_compatible(self):
        """Check that the version of Python used is sufficiently new."""
        version = sys.version_info
        if version.major < 3 or version.major > 3 and version.minor < 4:
            warnings.warn("You are using a version of Python that is not " \
                          "compatible with the Defaults class. If you would " \
                          "like to use the features of the Defaults class, " \
                          "then please use a version of Python newer than 3.4.")
            return False
        else:
            return True

    def _handler(self, func, *args, **kwargs):
        """Decorator for applying new function parameter defaults."""
        if self._version_is_compatible:
            @functools.wraps(func)
            def new_func(*args, **kwargs):
                # get the full argspec
                argspec = inspect.getfullargspec(func)

                # get dictionary of kwargs and their defaults
                try:
                    offset = len(argspec.args) - len(argspec.defaults)
                    old_kwargs = {arg : default for arg, default 
                                  in zip(argspec.args[offset:], argspec.defaults)}
                except TypeError:
                    # if there are no defaults in the argspec
                    old_kwargs = {}

                # make the args list into a dictionary
                args = {argspec.args[i] : arg for i, arg in enumerate(args)}
                
                # make a dictionary of everything passed to func
                passed_args = {param : value for args in (args, kwargs)
                                             for param, value in args.items()}

                # get the new defaults
                new_args = self()
                # keep only the parameters from the function signature
                new_args = {param : value for param, value in new_args.items()
                                          if param in argspec.args}
                # add the variable kwargs if they're there
                if argspec.varkw is not None and argspec.varkw in self().keys():
                    for kwarg, value in self(argspec.varkw).items():
                        new_args[kwarg] = value

                # choose which set of args will be used
                if self._override_defaults:
                    keys = set(list(new_args.keys()) + list(passed_args.keys()))
                else:
                    keys = set(list(old_kwargs.keys()) + list(passed_args.keys()))

                # make a final dictionary to pass to func
                final_args = {arg: (
                            passed_args[arg] if arg in passed_args.keys() else
                            new_args[arg] if self._override_defaults else
                            old_kwargs[arg]) for arg in keys}

                # use the final set of arguments/kwargs
                return func(**final_args)
        else:
            @functools.wraps(func)
            def new_func(*args, **kwargs):
                return func(*args, **kwargs)
        
        return new_func

defaults = Defaults()
_defaults = defaults._handler

