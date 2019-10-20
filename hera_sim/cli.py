"""
CLI for hera_sim
"""

import click
import os
import yaml
import warnings

from pyuvsim.simsetup import _parse_layout_csv
from astropy.coordinates import Angle
try:
    import bda
    from bda import bda_tools
except ImportError:
    bda = None

import hera_sim

main = click.Group()


@main.command()
@click.argument('input', type=click.Path(exists=True, dir_okay=False))
#                help="input YAML configuration file")
@click.option('-o', '--outfile', type=click.Path(dir_okay=False),
                help='path to output file. Over-rides config if passed.', default=None)
@click.option("-v", '--verbose', count=True)
def run(input, outfile, verbose):
    """
    Run a full simulation with systematics.
    """
    if verbose:
        print("Loading configuration file...")

    # load in config
    with open(input, 'r') as fl:
        yaml_contents = yaml.load(fl.read(), Loader=yaml.FullLoader)


    # figure out whether or not to do BDA
    bda_params = yaml_contents.get("bda", {})
    # make sure bda is installed if the user wants to do BDA
    if bda_params and bda is None:
        raise ImportError("You have defined BDA parameters but do not have "
                          "bda installed. Please install bda to proceed.")

    if verbose:
        print("Checking validity of filing parameters...")

    # extract parameters for saving to disk
    filing_params = yaml_contents.get("filing", {})

    # construct outfile name if not passed from command line
    if outfile is None:
        outfile = os.path.join(filing_params["outdir"], filing_params["outfile_name"])
    
    # get the filing format
    fmt = filing_params.get("output_format", None)
    assert fmt is not None, \
        "The output file format must be specified in the configuration file " \
        "under the 'filing' section."
    
    # assume miriad files have the extension "uv"; others are same as name
    fmt_to_ext = {"miriad" : "uv", 
                  "uvfits" : "uvfits", 
                  "uvh5" : "uvh5"}
    
    # make sure the output format is supported; only miriad, uvfits, uvh5
    # are currently supported by UVData objects
    supported_fmts = tuple(fmt_to_ext.keys())
    assert fmt in supported_fmts, \
        "UVData objects currently only support writing to the following " \
        "datatypes: {}".format(supported_fmts)
    
    # add appropriate extension if not specified already
    if not outfile.endswith('.%s'%fmt_to_ext[fmt]):
        outfile += ".%s"%fmt

    # XXX consider adding clobber as a click option
    if os.path.exists(outfile) and not filing_params['clobber']:
        print("Nothing to do: %s already exists and clobber=False"%outfile)
        return

    if verbose:
        print("Determining whether to use a default configuration...")

    # determine whether to use season defaults
    defaults = yaml_contents.get("defaults", {})
    if defaults:
        # this assertion is made to keep the configuration file as neat as
        # possible; it is confusing to have a full configuration nested
        # inside another configuration
        assert isinstance(defaults["default_config"], str), \
               "If a default configuration is set with the default_config " \
               "option in the configuration YAML, then it must be specified " \
               "by a string which is either an absolute path to a config " \
               "file compatible with hera_sim.defaults, or one of the season " \
               "configuration keywords."
        hera_sim.defaults.set(defaults["default_config"])

    if verbose:
        print("Constructing Simulator object...")

    # extract instrument parameters
    if isinstance(yaml_contents["telescope"]["array_layout"], str):
        # assume it's an antenna layout csv
        # XXX: revisit _parse_layout_csv documentation to see how
        # beams are handled with this
        antennas = _parse_layout_csv(yaml_contents["telescope"]["array_layout"])
    else:
        # assume it's constructed using antpos and the YAML tag !antpos
        antennas = yaml_contents["telescope"]["array_layout"]
    instrument_params = {"antennas": antennas}
    for parameter in ("freq", "time",):
        for key, value in yaml_contents[parameter].items():
            instrument_params[key] = value

    sim = hera_sim.Simulator(**instrument_params)

    # for this application, we only want the sky temperature model and
    # beam size; this will need to be updated in the future when the
    # defaults handling is changed, but we'll leave that to the version
    # 1 release
    config_params = {}
    # look for Tsky_mdl, omega_p, and integration_time; extract these
    for content in yaml_contents.values():
        if "Tsky_mdl" in content.keys():
            config_params["Tsky_mdl"] = content["Tsky_mdl"]
        if "omega_p" in content.keys():
            config_params["omega_p"] = content["omega_p"]
        if "integration_time" in content.keys():
            config_params["inttime"] = content["integration_time"]
    # warn the user if both defaults and configuration parameters specified
    if defaults and config_params:
        warnings.warn("You have chosen to use a default configuration in "
                      "addition to listing configuration parameters. The "
                      "configuration parameters will override the default "
                      "parameters.")
    hera_sim.defaults.set(config_params)

    if verbose:
        print("Extracting simulation parameters...")
    # extract the simulation parameters from the configuration file
    # the configuration file should only specify any particular parameter once
    # i.e. the same sky temperature model should be used throughout
    sim_params = {}
    sim_details = yaml_contents["simulation"]
    for component in sim_details["components"]:
        for content in yaml_contents.values():
            if component in content.keys():
                for model, params in content[component].items():
                    if model in sim_details["exclude"]:
                        continue
                    sim_params[model] = params
            continue

    if verbose:
        print("Running simulation...")
    sim.run_sim(**sim_params)

    # if the user wants to do BDA, then apply BDA
    if bda_params:
        # convert corr_FoV_angle to an Angle object
        bda_params["corr_FoV_angle"] = Angle(bda_params["corr_FoV_angle"])
        if verbose:
            print("Performing BDA...")
        sim.data = bda_tools.apply_bda(sim.data, **bda_params)

    # save the simulation
    # note that we may want to allow the functionality for the user to choose some
    # kwargs to pass to the write method
    if verbose:
        print("Writing simulation results to disk...")
    sim.write_data(outfile,
                   file_type=filing_params["output_format"],
                   **filing_params["kwargs"])

    if verbose:
        print("Simulation complete.")
    return
