"""A package for running instrument and systematic simulations for HERA."""

from pathlib import Path

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    DATA_PATH = Path(__file__).parent / "data"
    CONFIG_PATH = Path(__file__).parent / "config"
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass


from . import (
    __yaml_constructors,
    adjustment,
    antpos,
    beams,
    cli_utils,
    eor,
    foregrounds,
    interpolators,
    io,
    noise,
    rfi,
    sigchain,
    simulate,
    utils,
)
from .components import (
    SimulationComponent,
    component,
    get_all_components,
    get_model,
    get_models,
)
from .defaults import defaults
from .interpolators import Bandpass, Beam, Tsky
from .simulate import Simulator
from .visibilities import load_simulator_from_yaml, simulators
