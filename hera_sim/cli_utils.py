"""
Module containing useful helper functions and argparsers for running 
simulations with hera_sim via the command line.
"""
import os
from pathlib import Path

def get_filing_params(config):
    """Extract filing parameters from a configuration dictionary."""

    filing_params = dict(
        outdir=os.getcwd(),
        outfile_name="hera_sim_simulation.uvh5",
        output_format="uvh5",
        clobber=False,
    )
    filing_params.update(config.get("filing", {}))
    if filing_params["output_format"] not in ("miriad", "uvfits", "uvh5"):
        raise ValueError(
            "Output format not supported. Please use miriad, uvfits, or uvh5."
        )

    return filing_params
