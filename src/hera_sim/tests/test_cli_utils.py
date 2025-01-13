"""Test different utilities for the command-line interface."""

import os

import numpy as np
import pytest
from astropy import units
from pyuvdata import UVCal

from hera_sim import Simulator, cli_utils
from hera_sim.sigchain import gen_gains


@pytest.fixture
def freqs():
    return [1, 2, 3, 4]


@pytest.fixture
def times():
    return [1, 2, 3, 4]


@pytest.fixture
def array():
    return {0: [1, 2, 3]}


@pytest.fixture(params=[0, 1, 2, 3], ids=["Nf0df", "Nf0B", "f0dfB", "fi"])
def freq_params(request, freqs):
    return (
        dict(Nfreqs=100, start_freq=1e6, channel_width=1e4),
        dict(Nfreqs=100, start_freq=1e6, bandwidth=1e6),
        dict(start_freq=1e6, channel_width=1e4, bandwidth=1e6),
        dict(freq_array=freqs),
    )[request.param]


@pytest.fixture(params=[0, 1], ids=["Nt0dt", "ti"])
def time_params(request, times):
    return (
        dict(Ntimes=50, start_time=2458711.235, integration_time=10.7),
        dict(time_array=times),
    )[request.param]


@pytest.fixture(params=[0, 1], ids=["file", "dict"])
def array_params(tmp_path, request, array):
    if request.param == 0:
        tmp_csv = tmp_path / "test_layout.csv"
        tmp_csv.touch()
        return {"array_layout": f"{str(tmp_csv)}"}
    else:
        return {"array_layout": array}


def test_freq_param_validation(freq_params):
    assert cli_utils._validate_freq_params(freq_params)


def test_time_param_validation(time_params):
    assert cli_utils._validate_time_params(time_params)


def test_array_param_validation(array_params):
    assert cli_utils._validate_array_params(array_params["array_layout"])


def test_config_validation(freq_params, time_params, array_params):
    config = {"freq": freq_params, "time": time_params, "telescope": array_params}
    cli_utils.validate_config(config)


def test_filing_params_parser_custom_config():
    base_params = {
        "filing": dict(
            outdir="tmp", outfile_name="test.uvh5", output_format="uvh5", clobber=True
        )
    }

    # Check that the parameters haven't been changed.
    parsed_params = cli_utils.get_filing_params(base_params)
    assert base_params["filing"] == parsed_params


def test_filing_params_parser_default_config():
    # Check that the default settings are correct.
    default_params = cli_utils.get_filing_params({})
    expected_params = {
        "outdir": os.getcwd(),
        "outfile_name": "hera_sim_simulation.uvh5",
        "output_format": "uvh5",
        "clobber": False,
    }
    assert default_params == expected_params


def test_filing_params_parser_bad_config():
    # Check that it handles bad formats correctly
    params = {"filing": {"output_format": "not_supported"}}
    with pytest.raises(ValueError) as err:
        cli_utils.get_filing_params(params)
    assert "Output format not supported." in err.value.args[0]


def test_config_validation_default_config():
    cli_utils.validate_config({"defaults": "h2c"})


def test_config_validation_bad_defaults_type():
    with pytest.raises(ValueError) as err:
        cli_utils.validate_config({"defaults": 123})
    assert "Defaults in the CLI may only be" in err.value.args[0]


def test_config_validation_bad_defaults_string():
    with pytest.raises(ValueError) as err:
        cli_utils.validate_config({"defaults": "bad_setting"})
    assert err.value.args[0] == "Default configuration string not recognized."


@pytest.mark.parametrize("remove", ["freq", "time", "telescope"])
def test_validate_config_missing_params(remove, freqs, times, array):
    config = {
        "freq": {"freq_array": freqs},
        "time": {"time_array": times},
        "telescope": {"array_layout": array},
    }
    del config[remove]
    with pytest.raises(ValueError) as err:
        cli_utils.validate_config(config)
    assert err.value.args[0] == "Insufficient information for initializing simulation."


@pytest.mark.parametrize("corrupt", ["freq", "time"])
def test_validate_config_bad_params(corrupt, freqs, times, array):
    config = {
        "freq": {"freq_array": freqs},
        "time": {"time_array": times},
        "telescope": {"array_layout": array},
    }
    config[corrupt][f"{corrupt}_array"] = None
    with pytest.raises(ValueError) as err:
        cli_utils.validate_config(config)
    assert err.value.args[0] == "Insufficient information for initializing simulation."


def test_validate_array_params_bad_type():
    with pytest.raises(TypeError) as err:
        cli_utils._validate_array_params(5555)
    assert "Array layout must be" in err.value.args[0]


@pytest.fixture
def sim_freqs():
    return np.linspace(100e6, 200e6, 100)


@pytest.fixture
def sim_times():
    return 2458799 + np.arange(0, 60, 10.7) * units.s.to("day")


@pytest.fixture
def gains(sim_freqs):
    return gen_gains(freqs=sim_freqs / 1e9, ants=[0])


@pytest.fixture
def sim(sim_freqs, sim_times):
    return Simulator(
        freq_array=sim_freqs, time_array=sim_times, array_layout={0: [0, 0, 0]}
    )


@pytest.mark.parametrize("save_method", ["arrays", "sim", "data"])
def test_write_calfits(sim_freqs, sim_times, gains, sim, tmp_path, save_method):
    # Write the file.
    cal_file = str(tmp_path / f"from_{save_method}.calfits")
    if save_method == "arrays":
        kwargs = {"freqs": sim_freqs, "times": sim_times}
    else:
        if save_method == "data":
            sim = sim.data
        kwargs = {"sim": sim}
    cli_utils.write_calfits(gains, cal_file, **kwargs)

    # Check the parameters.
    uvc = UVCal()
    uvc.read_calfits(cal_file)
    assert all(
        [
            np.allclose(uvc.freq_array.flatten(), sim_freqs),
            np.allclose(uvc.time_array, sim_times),
            np.allclose(uvc.get_gains(0, "Jee").mean(axis=1), gains[0]),
        ]
    )


def test_write_calfits_bad_kwarg(sim_freqs, gains, tmp_path):
    cal_file = str(tmp_path / "bad_kwarg.calfits")
    with pytest.raises(TypeError) as err:
        cli_utils.write_calfits(gains, cal_file, sim=sim_freqs)
    assert err.value.args[0] == "sim must be a Simulator or UVData object."


def test_write_calfits_no_kwargs(gains, tmp_path):
    cal_file = str(tmp_path / "no_kwargs.calfits")
    with pytest.raises(ValueError) as err:
        cli_utils.write_calfits(gains, cal_file)
    assert "frequencies and times must be specified" in err.value.args[0]
