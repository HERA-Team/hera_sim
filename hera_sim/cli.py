"""
CLI for hera_sim
"""

import click
import os
import yaml

from pyuvsim.simsetup import _parse_layout_csv
try:
    import bda
    from bda import bda_tools
except ImportError:
    bda = None

import hera_sim

main = click.Group()


@main.command()
@click.argument('input', type=click.Path(exists=True, dir_okay=False),
                help="input YAML configuration file")
@click.option('-o', '--outfile', type=click.Path(dir_okay=False),
                help='path to output file. Over-rides config if passed.', default=None)
@click.option("-v", '--verbose', count=True)
def run(input, outfile, verbose):
    """
    Run a full simulation with systematics.
    """
    # load in config
    with open(input, 'r') as fl:
        yaml_contents = yaml.load(fl.read(), Loader=yaml.FullLoader)


    # figure out whether or not to do BDA
    bda_params = yaml_contents.get("bda", {})
    # make sure bda is installed if the user wants to do BDA
    if bda_params and bda is None:
        raise ImportError("You have defined BDA parameters but do not have "
                          "bda installed. Please install bda to proceed.")

    # extract parameters for saving to disk
    filing_params = yaml_contents["filing"]
    if outfile is None:
        outfile = os.path.join(filing_params["outdir"], filing_params["outfile_name"])
    fmt = filing_params["output_format"]
    if not outfile.endswith('.%s'%fmt):
        outfile += ".%s"%fmt

    if os.path.exists(outfile) and not filing_params['clobber']:
        print("Nothing to do: %s already exists and clobber=False"%outfile)
        return

    # XXX add a check to make sure the save type is OK?

    # determine whether to use season defaults
    defaults = yaml_contents.get("defaults", {})
    if defaults:
        if isinstance(defaults["default_config"], str):
            hera_sim.defaults.set(defaults["default_config"])
        elif isinstance(defaults["default_config"], dict):
            hera_sim.defaults.set(**defaults["default_config"])
        else:
            raise ValueError("If you wish to override function defaults, then "
                             "you must do one of the following:\n"
                             "specify a path to the default configuration file\n"
                             "specify the name of a set of defaults ('h1c' or 'h2c')\n"
                             "specify an appropriately formatted dictionary of default values.")

    # extract instrument parameters
    if isinstance(yaml_contents["telescope"]["array_layout"], str):
        # assume it's an antenna layout csv
        antennas = _parse_layout_csv(yaml_contents["telescope"]["array_layout"])
    else:
        # assume it's constructed using antpos and the YAML tag !antpos
        antennas = yaml_contents["telescope"]["array_layout"]
    instrument_params = {"antennas": antennas}
    for parameter in ("freq", "time",):
        for key, value in yaml_contents[parameter].items():
            instrument_params[key] = value

    sim = hera_sim.Simulator(**instrument_params)

    config_params = {}
    # need to figure out a mapping of this to how defaults works
    for component in ("analysis", "systematics",):
        for key, value in yaml_contents[component].items():
            config_params[key] = value
    # this currently isn't ideal. there should be checks to see if there
    # is any overlap between defaults and config_params; if there is
    # an overlap, a warning should be raised. if there is no overlap,
    # then the defaults and config_params dictionaries should be merged
    # and the resulting dictionary should be applied as the new defaults
    if not defaults and config_params:
        hera_sim.defaults.set(**config_params)

    sim_params = {}
    for key, value in yaml_contents["simulation"].items():
        sim_params[key] = value

    sim.run_sim(**sim_params)

    # if the user wants to do BDA, then apply BDA
    if bda_params:
        sim.data = bda_tools.apply_bda(sim.data, **bda_params)

    # save the simulation
    # note that we may want to allow the functionality for the user to choose some
    # kwargs to pass to the write method
    sim.write_data(outfile,
                   file_type=filing_params["output_format"],
                   **filing_params["kwargs"])
