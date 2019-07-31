"""
This module is designed to allow for easy interfacing with simulation default
parameters in an interactive environment.
"""

import yaml
import inspect
import functools
from .config import CONFIG_PATH
from os import path

SEASON_CONFIGS = {'h1c': path.join(CONFIG_PATH, 'HERA_H1C_config.yaml'),
                  'h2c': path.join(CONFIG_PATH, 'HERA_H2C_config.yaml')
                  }

class Defaults:
    """
    This class handles the retreival of simulation default parameters from
    YAML files and the ability to switch the default settings while in an
    interactive environment.
    """

    def __init__(self, config_file='h1c'):
        self._config = self._get_config(config_file)
        self._check_config()

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
        with open(self.config_file, 'r') as config:
            return yaml.load(config.read(), Loader=yaml.FullLoader)[module][model]

    def set_defaults(self, new_config):
        self._config = self._get_config(new_config)
        self._check_config()

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

    def _handler(self, func, *args, **kwargs):

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            # XXX do we want to use the season defaults by default?
            # XXX or do we want to default to the defaults from the func signature?
            use_func_defaults = kwargs.pop('use_func_defaults', True)

            # get model name and module name
            model = func.__name__
            module = func.__module__

            # peel off the actual module name
            module = module[module.index('.')+1:]

            # get the full argspec
            argspec = inspect.getfullargspec(func)

            # find out how many required arguments there are
            offset = len(argspec.args) - len(argspec.defaults)

            # initialize a dictionary of new kwargs to pass to func from defaults
            new_kwargs = self(module, model).copy()

            # now cycle through the kwargs, see if the user has set them
            for kwarg in new_kwargs.keys():
                # set any kwargs the user set to what they set them to
                if kwarg in kwargs.keys():
                    new_kwargs[kwarg] = kwargs[kwarg]
                elif use_func_defaults:
                    # use the defaults from the function signature
                    new_kwargs[kwarg] = argspec.defaults[argspec.args.index(kwarg) - offset]

            return func(*args, **new_kwargs)
        return new_func

defaults = Defaults()
_defaults = defaults._handler
