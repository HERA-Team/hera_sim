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
    """
    This class handles the retreival of simulation default parameters from
    YAML files and the ability to switch the default settings while in an
    interactive environment.
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
                use_func_defaults = kwargs.pop("use_func_defaults",
                        not self._use_season_defaults)

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

                # initialize a dictionary of new kwargs to pass to func from defaults
                new_kwargs = self(module, model).copy()

                rm_kwargs = []

                # now cycle through the kwargs, see if the user has set them
                # TODO: see if this can be done using list comprehension instead
                for kwarg in new_kwargs.keys():
                    # set any kwargs the user set to what they set them to
                    if kwarg in kwargs.keys():
                        new_kwargs[kwarg] = kwargs[kwarg]
                    elif use_func_defaults:
                        # use the defaults from the function signature
                        try:
                            new_kwargs[kwarg] = argspec.defaults[argspec.args.index(kwarg) 
                                                                 - offset]
                        except ValueError:
                            # this will get triggered if kwarg is not in argspec.args
                            # since this block is only entered if function defaults are
                            # desired, we shouldn't be overwriting things with defaults
                            rm_kwargs.append(kwarg)

                # check if any entries in the kwargs dict aren't in new_kwargs
                for kwarg, val in kwargs.items():
                    # add them to the new_kwarg dict if so
                    if kwarg not in new_kwargs.keys():
                        new_kwargs[kwarg] = val

                # remove any kwargs that shouldn't be in new_kwargs
                for kwarg in rm_kwargs:
                    del new_kwargs[kwarg]

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
