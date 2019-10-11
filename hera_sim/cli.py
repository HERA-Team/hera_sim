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
    if bda_params:
        raise ImportError("You have defined BDA parameters but do not have "
                          "bda installed. Please install bda to proceed.")

    # extract parameters for saving to disk
    # XXX: either click options need to be changed, or this does
    filing_params = yaml_contents["filing"]
    if outfile is None:
        outfile = os.path.join(filing_params["outdir"], filing_params["outfile_name"])
    fmt = filing_params["output_format"]
    if not outfile.endswith('.%s'%fmt):
        outfile += ".%s"%fmt

    if os.path.exists(outfile) and not filing_params['clobber']:
        print("Nothing to do: %s already exists and clobber=False"%outfile)
        return

    # TODO: add check for validity of save type; see pyuvdata documentation

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

    # XXX: this section will take careful thought to implement correctly
    config_params = {}
    # the following choice of syntax was scrapped and will be temporarily
    # commented out before being removed entirely
    #for component in ("analysis", "systematics",):
    #    for key, value in yaml_contents[component].items():
    #        config_params[key] = value
    # default behavior should be such that anything specified in the
    # configuration parameters will override whatever the current default
    # settings are, but perhaps warn the user that something is being
    # specified multiple times if defaults have been set and some of the
    # defaults conflict with the configuration settings
    # in the end, we want to handle the configuration stuff with the
    # functionality offered by defaults
    hera_sim.defaults.set(**config_params)

    # extract the simulation parameters from the configuration file
    # the configuration file should only specify any particular parameter once
    # i.e. the same sky temperature model should be used throughout
    sim_params = {}
    for component in yaml_contents["simulation"]["include"]:
        for content in yaml_contents.values():
            if component in content.keys():
                for particular_component, parameters in content[component].items():
                    sim_params[particular_component] = parameters

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

