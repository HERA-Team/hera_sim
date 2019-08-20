"""
This module is designed to allow for easy interfacing with simulation default
parameters in an interactive environment.
"""

import yaml
import inspect
import functools
import sys
import warnings
import numpy as np

from os import path
from .config import CONFIG_PATH
from .interpolators import _check_path

SEASON_CONFIGS = {'h1c': path.join(CONFIG_PATH, 'HERA_H1C_CONFIG.yaml'),
                  'h2c': path.join(CONFIG_PATH, 'HERA_H2C_CONFIG.yaml'),
                  }

class _Defaults:
    """
    This class handles the retreival of simulation default parameters from
    YAML files and the ability to switch the default settings while in an
    interactive environment. This class is intended to exist as a singleton;
    as such, an instance is created at the end of this module, and that
    instance is what is imported in the hera_sim __init__ script. See below
    for example usage within hera_sim.

    Examples:
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

        To view what the default parameter values for a model in a given
        module are:

        hera_sim.defaults(module, model)
    """

    def __init__(self, config_file='h1c'):
        self._config = self._get_config(config_file)
        self._check_config()
        self._use_season_defaults = False

    """
    Instantiate a Defaults object with a hook to a configuration file.

    Args:
        config_file (str, optional): hook to a YAML file or a season name
            The currently supported observing seasons are 'h1c' and 'h2c'.
            The loaded YAML file is intended to have the following form:
            {module: {model: {param: default}}}, where 'module' is any one
            of the modules
    """

    def __call__(self, module, model):
        with open(self._config, 'r') as config:
            defaults = yaml.load(config.read(), Loader=yaml.FullLoader)[module][model]
        # handle instances where default parameters are related to interpolators
        return self._retrieve_models(defaults)

    def set(self, new_config):
        self._config = self._get_config(new_config)
        self._check_config()
        self.activate()

    def activate(self):
        self._use_season_defaults = True

    def deactivate(self):
        self._use_season_defaults = False

    def _get_config(self, config_file):
        assert isinstance(config_file, str), \
                "Default configurations are set by passing a hook to a " \
                "configuration file or one of the season keys. The " \
                "currently supported season configurations are " \
                "{}.".format(SEASON_CONFIGS.keys())

        if config_file in SEASON_CONFIGS.keys():
            return SEASON_CONFIGS[config_file]
        else:
            return config_file

    def _check_config(self):
        assert path.exists(self._config), \
                "Please ensure that a path to the specified configuration " \
                "file exists. {} could not be found".format(self._config)

    @property
    def _version_is_compatible(self):
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

                # find out how many required arguments there are
                try:
                    offset = len(argspec.args) - len(argspec.defaults)
                except TypeError:
                    # this will be raised if there are no defaults; really
                    # the only thing that raises this is io.empty_uvdata (I think?)
                    offset = 0

                # make a dictionary of the function kwargs and their defaults
                try:
                    old_kwargs = {arg : default for arg, default 
                                  in zip(argspec.args[offset:], argspec.defaults)}
                except TypeError:
                    # if there are no defaults in the argspec
                    old_kwargs = {}

                # get the season defaults
                season_kwargs = self(module, model).copy()

                # choose which set of kwargs will be used
                if self._use_season_defaults:
                    keys = np.unique(list(season_kwargs.keys()) + list(kwargs.keys()))
                else:
                    keys = np.unique(list(old_kwargs.keys()) + list(kwargs.keys()))

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

    def _retrieve_models(self, defaults):
        # first, get a list of the interpolators supported
        clsmembs = dict(inspect.getmembers(sys.modules['hera_sim.interpolators'],
                                           inspect.isclass)
                      )
        interps = {}
        for cls, ref in clsmembs.items():
            if ref.__module__[:8]=="hera_sim":
                interps[cls] = ref

        for cls, ref in interps.items():
            # find out what the interpolator is usually referenced as
            f = getattr(ref, '_names')
            names = f()
            
            for name in names:
                # check if it's in the defaults
                if name in defaults.keys():
                    # if it is, then we need to make an instance of the interpolator
                    # this requires the datafile and interpolation kwargs
                    datafile = _check_path(defaults[name]['datafile'])
                    interp_kwargs = defaults[name]['interp_kwargs']
                    interp = ref(datafile, **interp_kwargs)
                
                    # now replace the default parameter with the interpolator
                    defaults[name] = interp
        
        return defaults

defaults = _Defaults()
_defaults = defaults._handler
