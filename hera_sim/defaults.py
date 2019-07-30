"""
This module is designed to allow for easy interfacing with simulation default
parameters in an interactive environment.
"""

import yaml
from .config import CONFIG_PATH
from os.path import join

SEASON_CONFIGS = {'h1c': join(CONFIG_PATH, 'HERA_H1C_config.yaml'),
                  'h2c': join(CONFIG_PATH, 'HERA_H2C_config.yaml')
                  }

class Defaults:
    """
    This class handles the retreival of simulation default parameters from
    YAML files and the ability to switch the default settings while in an
    interactive environment.
    """

    def __init__(self, config_file='h1c'):
        if config_file in SEASON_CONFIGS.keys():
            self._config_file = SEASON_CONFIGS[config_file]
        else:
            self._config_file = config_file

    """
    Instantiate a Defaults object with a hook to a configuration file.

    Args:
        config_file (str, optional): hook to a YAML file or a season name
            The currently supported observing seasons are 'h1c' and 'h2c'.
            The loaded YAML file is intended to have the following form:
            {module: {model: {param: default}}}, where 'module' is any one
            of the modules
    """

    @property
    def defaults(self, module, model):
        with open(self._config_file, 'r') as config:
            return yaml.load(config.read(), Loader=yaml.FullLoader)[module][model]

    def set_defaults(self, new_config):
        assert isinstance(new_config, str)
        self._config_file = new_config

defaults = Defaults()
