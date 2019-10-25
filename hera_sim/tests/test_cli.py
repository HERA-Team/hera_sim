"""Test the command line interface."""

import os
import tempfile

from pyuvdata import UVData
from nose.tools import raises
from click.testing import CliRunner
from hera_sim import run

# general idea: make a temporary directory, write a config file,
# then run the simulation using the config file, saving the product
# in the same directory. check that the uvh5 file has the correct properties

def construct_base_config(outdir, outfile_name, output_format):
    """Create a minimal working configuration file."""
    base_config = """
filing:
    outdir: %s
    outfile_name: %s
    output_format: %s
    clobber: True
    kwargs: {}
freq:
    n_freq: 50
    channel_width: 122070.3125
    start_freq: 100234567.333
time:
    n_times: 10
    integration_time: 8.59
    start_time: 2457458.174
telescope:
    array_layout: !antpos
        array_type: hex
        hex_num: 3
        sep: 14.6
        split_core: False
        outriggers: 0
    omega_p: !Beam
        datafile: HERA_H2C_BEAM_MODEL.npz
        interp_kwargs:
            interpolator: interp1d
            fill_value: extrapolate
    
""" % (outdir, outfile_name, output_format)
    return base_config[1:]

def add_bda(base_config):
    """Add BDA to config file with hard-coded parameter values."""
    bda_config = """
bda:
    max_decorr: 0
    pre_fs_int_time: !dimensionful
        value: 0.1
        units: s
    corr_FoV_angle: !dimensionful
        value: 20
        units: deg
    max_time: !dimensionful
        value: 16
        units: s
    corr_int_time: !dimensionful
        value: 2
        units: s
"""
    return bda_config[1:] + base_config

# make a temporary directory to write files to
tempdir = tempfile.mkdtemp()

@raises(AssertionError)
def test_bad_formats():
    config = construct_base_config(tempdir, "test", None)
    config_file = os.path.join(tempdir, "test_config.yaml")
    with open(config_file, 'w') as cfg:
        cfg.write(config)
    runner = CliRunner()
    results = runner.invoke(run, [config_file])
    if results.exit_code:
        raise results.exception
