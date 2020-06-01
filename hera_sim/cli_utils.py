"""
Module containing useful helper functions and argparsers for running 
simulations with hera_sim via the command line.
"""
import os
from pathlib import Path
from .defaults import SEASON_CONFIGS
from .simulate import Simulator

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

def validate_config(config):
    """Validate the contents of a loaded configuration file."""
    freqs_ok = False
    times_ok = False
    array_ok = False
    if config.get("defaults", None) is not None:
        if config["defaults"] in SEASON_CONFIGS.keys():
            return
        else:
            # TODO: figure out a clean way of checking this
            pass

    # skip the validation until this is figured out
    freqs_ok, times_ok, array_ok = (True,) * 3
    if not all(freqs_ok, times_ok, array_ok):
        raise ValueError("Insufficient information to initialize simulation.")

def write_calfits(
    gains, 
    filename, 
    sim=None, 
    freqs=None, 
    times=None, 
    clobber=False
):
    """Write gains to disk as a calfits file."""
    from hera_cal.io import write_cal
    if sim is not None:
        if isinstance(sim, Simulator):
            freqs = sim.freqs * 1e9
            times = sim.times
        else:
            freqs = np.unique(sim.freq_array)
            times = np.unique(sim.time_array)
    else:
        if freqs is None or times is None:
            raise ValueError(
                "If a simulation is not provided, then both frequencies and "
                "times must be specified."
            )

    write_cal(filename, gains, freqs, times, overwrite=clobber, return_uvc=False)
