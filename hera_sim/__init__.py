"""A package for running instrument and systematic simulations for HERA."""
from pathlib import Path

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    DATA_PATH = Path(__file__).parent / "data"
    CONFIG_PATH = Path(__file__).parent / "config"
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass


from . import __yaml_constructors
from . import adjustment
from . import antpos
from . import cli_utils
from . import foregrounds
from . import interpolators
from . import io
from . import noise
from . import rfi
from . import sigchain
from .visibilities import simulators, load_simulator_from_yaml
from . import eor
from . import utils
from . import simulate
from . import beams
from .simulate import Simulator
from .defaults import defaults
from .components import SimulationComponent, component
from .components import get_all_components, get_model, get_models
from .interpolators import Tsky, Bandpass, Beam
