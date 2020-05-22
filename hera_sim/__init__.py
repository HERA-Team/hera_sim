from pathlib import Path

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    DATA_PATH = Path(__file__).parent / "data"
    __version__ = version(__name__)
except PackageNotFoundError:
    print("package not found")
    # package is not installed
    pass


from . import __yaml_constructors
from . import antpos
from . import foregrounds
from . import interpolators
from . import io
from . import noise
from . import rfi
from . import sigchain
from .visibilities import simulators
from . import eor
from . import utils
from . import simulate
from .cli import run  # for testing purposes
from .simulate import Simulator
from .defaults import defaults
from .components import SimulationComponent, registry
from .components import list_discoverable_components
from .interpolators import Tsky, Bandpass, Beam
