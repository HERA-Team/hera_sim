"""
Module containing useful helper functions and argparsers for running 
simulations with hera_sim via the command line.
"""
import copy
import os
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
    if config.get("defaults", None) is not None:
        if type(config["defaults"]) is not str:
            raise ValueError(
                "Defaults in the CLI may only be specified using a string."
            )
        
        if config["defaults"] in SEASON_CONFIGS.keys():
            return
        else:
            raise ValueError("Default configuration string not recognized.")

    freq_params = config.get("freq", {})
    time_params = config.get("time", {})
    array_params = config.get("telescope", {}).get("array_layout", {})
    if any(param == {} for param in (freq_params, time_params, array_params)):
        raise ValueError("Insufficient information for initializing simulation.")

    freqs_ok = _validate_freq_params(freq_params)
    times_ok = _validate_time_params(time_params)
    array_ok = _validate_array_params(array_params)
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

    # Update gain keys to conform to write_cal assumptions.
    gains = {(ant, 'x'): gain for ant, gain in gains.items()}
    write_cal(filename, gains, freqs, times, overwrite=clobber, return_uvc=False)

def write_vis(ref_sim, vis, filename, save_format="uvh5", bda_params=None, **kwargs):
    """Write a simulated visibility to disk."""
    tmp_sim = copy.deepcopy(ref_sim)
    if not isinstance(tmp_sim, Simulator):
        tmp_sim = Simulator(data=tmp_sim)

    tmp_sim.data.data_array = vis
    if bda_params:
        # This should be installed if this clause is executed.
        from bda import bda_tools 
        tmp_sim.data = bda_tools.apply_bda(tmp_sim.data, **bda_params)

    tmp_sim.write(filename, save_format, **kwargs)
    return

def _validate_freq_params(freq_params):
    """Ensure frequency parameters specified are sufficient."""
    allowed_params = (
        "Nfreqs", "start_freq", "bandwidth", "freq_array", "channel_width"
    )
    allowed_combinations = list(
        combo for combo in itertools.combinations(all_params, 3)
        if "start_freq" in combo and "freq_array" not in combo
    ) + [("freq_array",)]
    for combination in allowed_combinations:
        if all(freq_params.get(param, None) is not None for param in combination):
            return True
    
    # None of the minimum necessary combinations are satisfied if we get here
    return False

def _validate_time_params(time_params):
    """Ensure time parameters specified are sufficient."""
    allowed_params = ("Ntimes", "start_time", "integration_time", "time_array")
    if time_params.get("time_array", None) is not None:
        return True
    elif all(
        time_params.get(param, None) is not None for param in allowed_params[:-1]
    ):
        # Technically, start_time doesn't need to be specified, since it has a
        # default setting in io.py, but that might not be set in stone.
        return True
    else:
        return False

def _validate_array_params(array_params):
    """Ensure array layout is OK."""
    if isinstance(array_params, dict):
        # Shallow check; make sure each antenna position is a 3-vector.
        if all(len(pos) == 3 for pos in array_params.values()):
            return True
    elif isinstance(array_params, str):
        # Shallow check; just make sure the file exists.
        return os.path.exists(array_params)
    else:
        return False
