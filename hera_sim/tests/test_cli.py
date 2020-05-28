"""Test the command line interface."""

import hera_sim
import os
import pathlib
import pytest
import tempfile

from pyuvdata import UVData
from nose.tools import raises
from click.testing import CliRunner
from hera_sim import run

# general idea: make a temporary directory, write a config file,
# then run the simulation using the config file, saving the product
# in the same directory. check that the uvh5 file has the correct properties
tempdir = pathlib.Path(tempfile.mkdtemp())


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
    Nfreqs: 50
    channel_width: 122070.3125
    start_freq: 100234567.333
time:
    Ntimes: 10
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

""" % (
        outdir,
        outfile_name,
        output_format,
    )
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


def set_defaults(config, defaults):
    """Choose to use a default set of function parameters."""
    new_config = """
defaults:
    default_config: {defaults}
""".format(
        defaults=defaults
    )
    return config + new_config[1:]


def add_systematics(config):
    """Add systematics to a config file, using default settings."""
    sim_config = """
systematics:
    rfi:
        rfi_stations: {}
        rfi_impulse: {}
        rfi_scatter: {}
        rfi_dtv: {}
    sigchain:
        gains: {}
    crosstalk:
        gen_whitenoise_xtalk: {}
"""
    return config + sim_config[1:]


def add_sky(config):
    """Add sky temperature model, EoR, and foregrounds, using defaults."""
    sky_config = """
sky:
    Tsky_mdl: !Tsky
        datafile: HERA_Tsky_Reformatted.npz
        interp_kwargs:
            pol: xx
    eor:
        noiselike_eor: {}
    foregrounds:
        diffuse_foreground: {}
        pntsrc_foreground: {}
"""
    return config + sky_config[1:]


def set_simulation(config, components, exclude):
    """Choose which components to simulate, and which parts to exclude."""
    sim_config = """
simulation:
    components: [{components}]
    exclude: [{exclude}]
""".format(
        components=", ".join(components), exclude=", ".join(exclude)
    )
    return config + sim_config[1:]


@raises(AssertionError)
def test_bad_formats():
    config = construct_base_config(str(tempdir), "test", None)
    config_file = tempdir / "test_config.yaml"
    with open(config_file, "w") as cfg:
        cfg.write(config)
    runner = CliRunner()
    results = runner.invoke(run, [str(config_file)])
    if results.exit_code:
        raise results.exception


def test_verbose_statements():
    config = construct_base_config(str(tempdir), "test", "uvh5")
    config = set_defaults(config, "h1c")
    sim_cmp = [
        "foregrounds",
    ]
    exclude = []
    config = set_simulation(config, sim_cmp, exclude)

    config_file = os.path.join(tempdir, "test_verbose.yaml")
    with open(config_file, "w") as cfg:
        cfg.write(config)

    runner = CliRunner()
    # test with --verbose
    results = runner.invoke(run, [str(config_file), "--verbose"])
    stdout = results.stdout
    # check that the output has some expected statements
    assert "Loading configuration file..." in stdout
    assert "Checking validity of filing parameters..." in stdout

    # test with -v
    results = runner.invoke(run, [str(config_file), "-v"])
    stdout = results.stdout
    # check for some expected output
    assert "Constructing Simulator object..." in stdout
    assert "Running simulation..." in stdout


@pytest.mark.skip("Haven't figured out issue related to checking stdout.")
def test_save_all():
    # set up config
    config = construct_base_config(str(tempdir), "test", "uvh5")
    config = add_systematics(add_sky(config))
    sim_cmp = ["foregrounds", "rfi", "sigchain"]
    exclude = []
    config = set_simulation(config, sim_cmp, exclude)

    # write to file
    config_file = tempdir / "test_save_all.yaml"
    with open(config_file, "w") as cfg:
        cfg.write(config)

    # CliRunner does not ever reach the save phase for some reason
    # so let's do this using os.system
    os.system("hera_sim run --save-all %s" % config_file)

    # now check that all of the correct files are saved
    dir_contents = os.listdir(tempdir)
    assert "test.diffuse_foreground.uvh5" in dir_contents
    assert "test.pntsrc_foreground.uvh5" in dir_contents
    assert "test.gains.uvh5" in dir_contents
    assert "test.rfi_impulse.uvh5" in dir_contents


@pytest.mark.skip("Haven't figured out issue related to checking stdout.")
def test_no_clobber():
    config = construct_base_config(tempdir, "test", "uvh5").replace("True", "False")
    config = set_defaults(config, "h1c")
    sim_cmp = [
        "foregrounds",
    ]
    exclude = []
    config = set_simulation(config, sim_cmp, exclude)

    config_file = os.path.join(tempdir, "test_clobber.yaml")
    with open(config_file, "w") as cfg:
        cfg.write(config)

    runner = CliRunner()
    results = runner.invoke(run, [str(config_file),])
    stdout = results.stdout
    assert "Nothing to do:" in stdout
