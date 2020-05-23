"""
CLI for hera_sim
"""
import argparse
import copy
import os
import yaml
import warnings

import hera_sim
from hera_sim import cli_utils
from pyuvsim.simsetup import _parse_layout_csv
from astropy.coordinates import Angle
try:
    import bda
    from bda import bda_tools
except ImportError:
    warnings.warn("bda failed to import. Baseline-dependent averaging not available.")
    bda = None

parser = argparse.ArgumentParser(
    description="Run a hera_sim-managed simulation from the command line."
)
parser.add_argument("config", type=str, help="Path to configuration file.")
parser.add_argument(
    "-o",
    "--outfile",
    type=str,
    default=None,
    help="Where to save simulated data. Overrides outfile specified in config."
)
parser.add_argument(
    "-v",
    "--verbose",
    default=False,
    action="store_true",
    help="Print progress updates."
)
parser.add_argument(
    "-sa",
    "--save-all",
    default=False,
    action="store_true",
    help="Save each simulation component."
)
parser.add_argument(
    "--clobber",
    default=False,
    action="store_true",
    help="Overwrite existing files in case of name conflicts."
)
args = parser.parse_args()

if args.verbose:
    print("Reading configuration file and validating contents...")

with open(args.config, 'r') as cfg:
    config = yaml.load(cfg.read(), loader=yaml.FullLoader)

bda_params = config.get("bda", {})
if bda_params and bda is None:
    raise ModuleNotFoundError("Please ensure bda is installed and try again.")


