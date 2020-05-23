"""
Module containing useful helper functions and argparsers for running 
simulations with hera_sim via the command line.
"""

import argparse

from . import interpolators

def simulate_argparser():
    """Argparser for running hera_sim from the command line."""

    desc = "Run a hera_sim-managed simulation from the command line."
    a = argparse.ArgumentParser(description=desc)
    a.add_argument("config", type=str, help="Path to configuration file.")
    a.add_argument(
        "-o",
        "--outfile",
        type=str,
        default=None,
        help="Where to save simulation. Overrides outfile in config."
    )
    a.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Print progress updates."
    )
    a.add_argument(
        "-sa",
        "--save-all",
        default=False,
        action="store_true",
        help="Save each simulation component."
    )
    a.add_argument(
        "--clobber",
        default=False,
        action="store_true",
        help="Overwrite existing files in case of name conflicts."
    )
    args = a.parse_args()
    return args
