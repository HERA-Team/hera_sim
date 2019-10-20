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

class _Defaults:
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

    To view what the default parameter values for a `model` in a given
    `module` are:

    hera_sim.defaults(module, model)
    """

    def __init__(self, config='h1c'):
        self._raw_config = {}
        self._config = {}
        self._config_name = None
        self._set_config(config)
        self._override_defaults = False
    """Load in a configuration and check its formatting.

    Parameters
    ----------
    config : str or dict, optional (default 'h1c') 
        May either be an absolute path to a configuration YAML, one of
        the observing season keywords ('h1c', 'h2c'), or a dictionary 
        with the appropriate format.

        TODO: rewrite the following bit in accordance w/ new config format

        The loaded YAML file is intended to have the following form:
        {module: {model: {param: default}}}, where 'module' is any one
        of the `hera_sim` modules, `model` is any of the functions within
        the specified module, and `param` is any of the model function's 
        default parameters.

    Examples
    --------
    An example configuration YAML should be formatted as follows:
    module1:
        model1:
            param1: default1
            param2: default2
            ...
        model2:
            ...
    module2:
        ...
    """

    def __call__(self, component=None):
        """Return the defaults dictionary, or just a component if specified."""
        if component is not None:
            try:
                return self._config[component]
            except KeyError:
                raise KeyError("{} not found in configuration.".format(component))
        else:
            return self._config

    def set(self, new_config):
        """Set the defaults to those specified in `new_config`.

        Parameters
        ----------
        new_config : str or dict
            Absolute path to configuration file or dictionary of configuration
            parameters formatted in the same way a configuration would be loaded.

        Notes
        -----
        Calling this method also activates the defaults.
        """
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
            if config in SEASON_CONFIGS.keys():
                self._config_name = config
                config = SEASON_CONFIGS[config]
            with open(config, 'r') as conf:
                self._raw_config = yaml.load(conf.read(), Loader=yaml.FullLoader)
            # raw configuration dictionary should be nested; this line pulls
            # out the individual terms in the nested dictionary
            self._unpack_raw_config()
        elif isinstance(config, dict):
            # set the raw configuration dictionary to config
            self._raw_config = config
            # check if config is formatted like a config file
            isnested = all([isinstance(entry, dict) for entry in config.values()])
            # if it's formatted like a config file, then unpack it
            if isnested:
                self._unpack_raw_config()
            else:
                self._config = config
            self._config_name = "custom"
        else:
            raise ValueError(
                    "The configuration must be a dictionary, an absolute " \
                    "path to a configuration YAML, or a season keyword." )
        self._check_config()

    def _unpack_raw_config(self):
        """Extract individual components from raw configuration dictionary."""
        self._config = {param : value for component in self._raw_config.values()
                                      for param, value in component.items()}

    def _check_config(self):
        """Check if any keys in the configuration are repeated, warn if so."""
        counts = {key : 0 for key in self().keys()}
        values = {key : [] for key in self().keys()}
        for param, value in self._raw_config.items():
            if isinstance(value, dict):
                for key, val in value.items():
                    counts[key] += 1
                    values[key].append(val)
            else:
                counts[param] += 1
                values[param].append(value)
        flags = {key : (1 if count > 1 else 0) for key, count in counts.items()}
        if any(flags.values()):
            warning = "The following parameters have multiple values defined " \
                      "in the configuration:\n"
            for param, flag in flags.items():
                if flag:
                    warning += "{}\n".format(param)
            warning += "Please check your configuration, as only the last " \
                       "value specified for each parameter will be used."
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

defaults = _Defaults()
_defaults = defaults._handler

