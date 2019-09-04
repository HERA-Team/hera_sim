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
from .interpolators import _check_path

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
        self._config = self._get_config(config)
        self._check_config()
        self._use_season_defaults = False
    """Load in a configuration and check its formatting.

    Parameters
    ----------
    config : str or dict, optional (default 'h1c') 
        May either be an absolute path to a configuration YAML, one of
        the observing season keywords ({}), or a dictionary with the
        appropriate format.

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

    def __call__(self, module, model):
        """Return the defaults for the given `model` in `module`."""
        return self._config[module][model]
        #with open(self._config, 'r') as config:
        #    defaults = yaml.load(config.read(), Loader=yaml.FullLoader)[module][model]
        #return defaults

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
        self._config = self._get_config(new_config)
        self._check_config()
        self.activate()

    def activate(self):
        """Activate the defaults."""
        self._use_season_defaults = True

    def deactivate(self):
        """Revert to function defaults."""
        self._use_season_defaults = False

    def _get_config(self, config):
        """Validate the choice of configuration file."""
        #assert isinstance(config, str), \
        #        "Default configurations are set by passing a path to a " \
        #        "configuration file or one of the season keys. The " \
        #        "currently supported season configurations are " \
        #        "{}.".format(SEASON_CONFIGS.keys())
        if isinstance(config, str):
            if config in SEASON_CONFIGS.keys():
                config = SEASON_CONFIGS[config]
                # return SEASON_CONFIGS[config]
            with open(config, 'r') as conf:
                defaults = yaml.load(conf.read(), Loader=yaml.FullLoader)
            return defaults
        elif isinstance(config, dict):
            return config
        else:
            raise ValueError(
                    "The configuration must be a dictionary or an absolute " \
                    "path to a configuration YAML." )
        #else:
        #    return config

    def _check_config(self):
        """Confirm that the specified configuration file can be found."""
        #assert path.exists(self._config), \
        #        "Please ensure that a path to the specified configuration " \
        #        "file exists. {} could not be found".format(self._config)
        error_message = "Your configuration should be formatted as a nested " \
                "dictionary: {module: {model: {params: values}}}"
        try:
            module = list(self._config.keys())[0]
            model = list(self._config[module].keys())[0]
        except KeyError:
            raise AssertionError(error_message)
        # seems a little stupid to use self._config rather than self()
        assert isinstance(self._config[module][model], dict), error_message

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
                # get model name and module name
                model = func.__name__
                module = func.__module__

                # peel off the actual module name
                module = module[module.index('.')+1:]

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

                # get the season defaults
                season_kwargs = self(module, model).copy()

                # choose which set of kwargs will be used
                if self._use_season_defaults:
                    keys = set(list(season_kwargs.keys()) + list(kwargs.keys()))
                else:
                    keys = set(list(old_kwargs.keys()) + list(kwargs.keys()))

                # make a new dictionary of kwargs to pass to func
                new_kwargs = {
                        kwarg: (
                            kwargs[kwarg] if kwarg in kwargs.keys() else
                            season_kwargs[kwarg] if self._use_season_defaults else
                            old_kwargs[kwarg]
                            )
                        for kwarg in keys}
                # return the function using the new kwargs
                return func(*args, **new_kwargs)
        else:
            @functools.wraps(func)
            def new_func(*args, **kwargs):
                return func(*args, **kwargs)
        
        return new_func

defaults = _Defaults()
_defaults = defaults._handler

