from . import __yaml_constructors
from . import antpos
from . import foregrounds
from . import interpolators
from . import io
from . import noise
from . import rfi
from . import sigchain
from hera_sim.visibilities import simulators
from . import version
from . import eor
from . import utils
from . import simulate
from .simulate import Simulator
from .defaults import defaults

__version__ = version.version

# warn the user about upcoming deprecations
import warnings
warnings.warn(
        "\nIn the next major release, all HERA-specific variables will be " \
        "removed from the codebase. The following variables will need to be " \
        "accessed through new class-like structures to be introduced in the " \
        "next major release: \n\n" \
        "noise.HERA_Tsky_mdl\n" \
        "noise.HERA_BEAM_POLY\n" \
        "sigchain.HERA_NRAO_BANDPASS\n" \
        "rfi.HERA_RFI_STATIONS\n\n" \
        "Additionally, the next major release will involve modifications " \
        "to the package's API, which move toward a regularization of the " \
        "way in which hera_sim methods are interfaced with; in particular, " \
        "changes will be made such that the Simulator class is the most " \
        "intuitive way of interfacing with the hera_sim package features.",
        FutureWarning)

