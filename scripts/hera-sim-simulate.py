#!/bin/env python

"""
Command-line interface for ``hera_sim``.

This script may be used to run a simulation from a configuration file and write
the result, with the option to write intermediate results, to disk. This script
uses the quick-and-dirty sky simulation tools from ``hera_sim``, as well as the
various systematics simulation tools available. This includes bandpass gains,
reflection gains, cross-coupling crosstalk, RFI, and thermal noise. For more
information on the simulation tools available, please see the tutorial notebooks.
There is a tutorial RST document for using this command-line interface available
in the readthedocs, and there is a template configuration file for reference in
the ``config_examples`` folder in the top-level ``hera_sim`` directory.
"""

import argparse
import copy
import os
import sys

import numpy as np
import yaml
from astropy import units
from astropy.coordinates import Angle

import hera_sim
from hera_sim import cli_utils

try:
    import bda
    from bda import bda_tools
except ImportError:
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
    help="Where to save simulated data. Overrides outfile specified in config.",
)
parser.add_argument(
    "-v",
    "--verbose",
    default=False,
    action="store_true",
    help="Print progress updates.",
)
parser.add_argument(
    "-sa",
    "--save_all",
    default=False,
    action="store_true",
    help="Save each simulation component.",
)
parser.add_argument(
    "--clobber",
    default=None,  # To allow the user to specify clobber in the config file.
    action="store_true",
    help="Overwrite existing files in case of name conflicts.",
)
args = parser.parse_args()

if __name__ == "__main__":
    if args.verbose:
        print("Reading configuration file and validating contents...")

    with open(args.config) as cfg:
        config = yaml.load(cfg.read(), Loader=yaml.FullLoader)

    cli_utils.validate_config(config)
    bda_params = config.get("bda", {})
    if bda_params:
        bda_params["corr_fov_angle"] = Angle(
            bda_params.get("corr_fov_angle", 20 * units.deg)
        )

    if bda_params and bda is None:
        raise ModuleNotFoundError("Please ensure bda is installed and try again.")

    filing_params = cli_utils.get_filing_params(config)
    args.outfile = args.outfile or os.path.join(
        filing_params["outdir"], filing_params["outfile_name"]
    )
    args.clobber = filing_params["clobber"] if args.clobber is None else args.clobber
    if os.path.exists(args.outfile) and not args.clobber:
        print("File exists and clobber=False; exiting.")
        sys.exit()

    if args.verbose:
        print("Preparing simulation...")

    instrument_parameters = {}
    defaults = config.get("defaults", None)
    if defaults is not None:
        if args.verbose:
            print(f"Using default configuration: {defaults}")
        hera_sim.defaults.set(defaults, refresh=True)

    telescope_params = config.get("telescope", {})
    array_layout = telescope_params.get("array_layout", None)
    if array_layout is not None:
        instrument_parameters["array_layout"] = array_layout

    # Finish filling out the instrument parameters and initialize the Simulator.
    frequency_parameters = config.get("freq", {})
    instrument_parameters.update(frequency_parameters)
    time_parameters = config.get("time", {})
    instrument_parameters.update(time_parameters)
    hera_sim.defaults.set(instrument_parameters, refresh=False)
    sim = hera_sim.Simulator()

    # Now handle the extras.
    if telescope_params.get("omega_p", None) is not None:
        hera_sim.defaults.set({"omega_p": telescope_params["omega_p"]}, refresh=False)

    sky_parameters = config.get("sky", {})
    Tsky_mdl = sky_parameters.pop("Tsky_mdl", None)
    if Tsky_mdl is not None:
        hera_sim.defaults.set({"Tsky_mdl": Tsky_mdl}, refresh=False)

    systematics_parameters = config.get("systematics", {})
    default_components = ["foregrounds", "eor", "noise", "rfi", "sigchain"]
    simulation_info = config.get(
        "simulation", {"components": default_components, "exclude": []}
    )
    all_simulation_parameters = {
        component: parameter_dict
        for parameters in (sky_parameters, systematics_parameters)
        for component, parameter_dict in parameters.items()
    }
    simulation_parameters = {}
    for component in simulation_info["components"]:
        parameters = all_simulation_parameters.get(component, {})
        for this_component, settings in parameters.items():
            if this_component in simulation_info.get("exclude", []):
                continue

            simulation_parameters[this_component] = settings

    if args.verbose:
        print("Running simulation...")

    for component, parameters in simulation_parameters.items():
        if args.verbose:
            print(f"Now simulating: {component}")
        if args.save_all:
            filename = f"_{component}".join(os.path.splitext(args.outfile))
            data = sim.add(component, ret_vis=True, **parameters)
            # BDA will modify the time structure of the data, so we'll need to
            # apply it to each component prior to saving.
            this_sim = copy.deepcopy(sim)
            if type(data) is np.ndarray:
                this_sim.data.data_array = data
            if bda_params:
                this_sim.data = bda_tools.apply_bda(this_sim.data, **bda_params)
            if isinstance(data, dict):
                # The component is a gain-like term, so save as a calfits file.
                ext = os.path.splitext(filename)[1]
                if ext == "":
                    filename += ".calfits"
                else:
                    filename = filename.replace(ext, ".calfits")
                # XXX when time-varying gains are supported, we'll need special
                # handling here for data that has had BDA applied.
                cli_utils.write_calfits(data, filename, sim=this_sim)
            else:
                this_sim.write(
                    filename,
                    filing_params.get("output_format", "uvh5"),
                    **filing_params.get("kwargs", {}),
                )
            del this_sim
        else:
            sim.add(component, ret_vis=False, **parameters)

    if bda_params:
        if args.verbose:
            print("Applying BDA...")

        sim.data = bda_tools.apply_bda(sim.data, **bda_params)

    if args.verbose:
        print("Writing simulation to disk...")

    sim.data.history += "\nSimulation performed with hera_sim, using configuration "
    sim.data.history += f"file {args.config}"
    sim.write(
        args.outfile,
        save_format=filing_params.get("output_format", "uvh5"),
        **filing_params.get("kwargs", {}),
    )
