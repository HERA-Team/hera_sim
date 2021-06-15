"""Useful helper functions and argparsers for running simulations via CLI."""
import itertools
import os
import numpy as np
from .defaults import SEASON_CONFIGS
from .simulate import Simulator
from pyuvdata import UVData


def get_filing_params(config: dict):
    """Extract filing parameters from a configuration dictionary.

    Parameters
    ----------
    config
        The full configuration dict.

    Returns
    -------
    dict
        Filing parameter from the config, with default entries
        filled in.

    Raises
    ------
    ValueError
        If ``output_format`` not in "miriad", "uvfits", or "uvh5".
    """
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


def validate_config(config: dict):
    """Validate the contents of a loaded configuration file.

    Parameters
    ----------
    config
        The full configuration dict.

    Raises
    ------
    ValueError
        If either insufficient information is provided, or the info
        is not valid.
    """
    if config.get("defaults") is not None:
        if type(config["defaults"]) is not str:
            raise ValueError(
                "Defaults in the CLI may only be specified using a string. "
                "The string used may specify either a path to a configuration "
                "yaml or one of the named default configurations."
            )

        if config["defaults"] in SEASON_CONFIGS.keys():
            return
        else:
            raise ValueError("Default configuration string not recognized.")

    freq_params = config.get("freq", {})
    time_params = config.get("time", {})
    array_params = config.get("telescope", {}).get("array_layout", {})
    if {} in (freq_params, time_params, array_params):
        raise ValueError("Insufficient information for initializing simulation.")

    freqs_ok = _validate_freq_params(freq_params)
    times_ok = _validate_time_params(time_params)
    array_ok = _validate_array_params(array_params)
    if not all([freqs_ok, times_ok, array_ok]):
        raise ValueError("Insufficient information for initializing simulation.")


def write_calfits(gains, filename, sim=None, freqs=None, times=None, clobber=False):
    """
    Write gains to disk as a calfits file.

    Parameters
    ----------
    gains: dict
        Dictionary mapping antenna numbers or (ant, pol) tuples to gains. Gains
        may either be spectra or waterfalls.
    filename: str
        Name of file, including desired extension.
    sim: :class:`pyuvdata.UVData` instance or :class:`~.simulate.Simulator` instance
        Object containing metadata pertaining to the gains to be saved. Does not
        need to be provided if both ``freqs`` and ``times`` are provided.
    freqs: array-like of float
        Frequencies corresponding to gains, in Hz. Does not need to be provided
        if ``sim`` is provided.
    times: array-like of float
        Times corresponding to gains, in JD. Does not need to be provided if
        ``sim`` is provided.
    clobber: bool, optional
        Whether to overwrite existing file in the case of a name conflict.
        Default is to *not* overwrite conflicting files.
    """
    from hera_cal.io import write_cal

    gains = gains.copy()

    if sim is not None:
        if not isinstance(sim, (Simulator, UVData)):
            raise TypeError("sim must be a Simulator or UVData object.")

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
    if all(np.issctype(type(ant)) for ant in gains.keys()):
        # Make sure that gain keys are (ant, jpol) tuples.
        gains = {(ant, "Jee"): gain for ant, gain in gains.items()}

    # Ensure that all of the gains have the right shape.
    for antpol, gain in gains.items():
        if gain.ndim == 1:
            gains[antpol] = np.outer(np.ones(times.size), gain)

    write_cal(filename, gains, freqs, times, overwrite=clobber, return_uvc=False)


def _validate_freq_params(freq_params):
    """Ensure frequency parameters specified are sufficient."""
    allowed_params = (
        "Nfreqs",
        "start_freq",
        "bandwidth",
        "freq_array",
        "channel_width",
    )
    allowed_combinations = [
        combo
        for combo in itertools.combinations(allowed_params, 3)
        if "start_freq" in combo and "freq_array" not in combo
    ] + [("freq_array",)]
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
    elif all(time_params.get(param, None) is not None for param in allowed_params[:-1]):
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
        raise TypeError("Array layout must be a dictionary or path to a layout csv.")
