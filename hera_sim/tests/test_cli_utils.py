"""Test different utilities for the command-line interface."""

import os
import pytest

from astropy import units
import numpy as np

from hera_sim import cli_utils
from hera_sim import Simulator
from hera_sim.sigchain import gen_gains
from pyuvdata import UVCal


def test_filing_params_parser():
    base_params = {
        "filing": dict(
            outdir="tmp", outfile_name="test.uvh5", output_format="uvh5", clobber=True
        )
    }

    # Check that the parameters haven't been changed.
    parsed_params = cli_utils.get_filing_params(base_params)
    assert base_params["filing"] == parsed_params

    # Check that the default settings are correct.
    default_params = cli_utils.get_filing_params({})
    assert default_params["outdir"] == os.getcwd()
    assert default_params["outfile_name"] == "hera_sim_simulation.uvh5"
    assert default_params["output_format"] == "uvh5"
    assert default_params["clobber"] is False

    # Check that it handles bad formats correctly
    bad_params = base_params.copy()
    bad_params["filing"]["output_format"] = "not_supported"
    with pytest.raises(ValueError):
        cli_utils.get_filing_params(bad_params)


def test_config_validation(tmp_path):
    # Construct various possible configurations.
    freq_params = {
        "A": dict(Nfreq=100, start_freq=1e6, channel_width=1e4),
        "B": dict(Nfreq=100, start_freq=1e6, bandwidth=1e6),
        "C": dict(start_freq=1e6, channel_width=1e4, bandwidth=1e6),
        "D": dict(freq_array=[1, 2, 3, 4]),
    }
    time_params = {
        "A": dict(Ntimes=50, start_time=2458711.235, integration_time=10.7),
        "B": dict(time_array=[1, 2, 3, 4]),
    }
    # Make sure a mock array layout file exists.
    tmp_csv = tmp_path / "test_layout.csv"
    tmp_csv.touch()
    array_params = {
        "A": {"array_layout": f"{str(tmp_csv)}"},
        "B": {"array_layout": {0: [1, 2, 3]}},
    }

    # Check that every possible choice of configuration works.
    for freq in freq_params.values():
        for time in time_params.values():
            for array in array_params.values():
                assert cli_utils._validate_freq_params(freq)
                assert cli_utils._validate_time_params(time)
                assert cli_utils._validate_array_params(array["array_layout"])
                config = {"freq": freq, "time": time, "telescope": array}
                cli_utils.validate_config(config)

    # Check that passing a season default key doesn't raise an error.
    cli_utils.validate_config({"defaults": "h2c"})

    # Now check that exceptions are raised as expected.
    with pytest.raises(ValueError) as err:
        cli_utils.validate_config({})
    assert "Insufficient information" in err.value.args[0]

    with pytest.raises(ValueError) as err:
        cli_utils.validate_config({"defaults": {"not": "valid"}})
    assert "may only be specified using a string." in err.value.args[0]

    with pytest.raises(ValueError) as err:
        cli_utils.validate_config({"defaults": "invalid"})
    assert err.value.args[0] == "Default configuration string not recognized."

    with pytest.raises(ValueError) as err:
        cli_utils.validate_config(
            {"freq": freq_params["A"], "time": time_params["A"],}
        )
    assert "Insufficient information" in err.value.args[0]

    with pytest.raises(ValueError) as err:
        cli_utils.validate_config(
            {"freq": freq_params["A"], "telescope": array_params["A"],}
        )
    assert "Insufficient information" in err.value.args[0]

    with pytest.raises(ValueError) as err:
        cli_utils.validate_config(
            {"time": time_params["A"], "telescope": array_params["A"],}
        )
    assert "Insufficient information" in err.value.args[0]

    with pytest.raises(ValueError) as err:
        cli_utils.validate_config(
            {
                "freq": freq_params["A"],
                "time": time_params["A"],
                "telescope": {"array_layout": "nonexistent.csv"},
            }
        )
    assert "Insufficient information" in err.value.args[0]

    with pytest.raises(ValueError) as err:
        config = {
            "freq": freq_params["A"],
            "time": time_params["A"],
            "telescope": array_params["A"],
        }
        config["time"]["start_time"] = None
        cli_utils.validate_config(config)
    assert "Insufficient information" in err.value.args[0]

    with pytest.raises(ValueError) as err:
        config = {
            "freq": freq_params["A"],
            "time": time_params["A"],
            "telescope": array_params["A"],
        }
        config["freq"]["start_freq"] = None
        cli_utils.validate_config(config)
    assert "Insufficient information" in err.value.args[0]

    with pytest.raises(TypeError) as err:
        cli_utils.validate_config(
            {
                "freq": freq_params["A"],
                "time": time_params["A"],
                "telescope": {"array_layout": None},
            }
        )
    assert "Array layout" in err.value.args[0]


def test_write_calfits(tmp_path):
    # Mock up antenna gains
    freqs = np.linspace(100e6, 200e6, 100)
    freqs_GHz = freqs / 1e9
    times = 2458799 + np.arange(0, 60, 10.7) * units.s.to("day")
    ants = {0: np.array([0, 0, 0])}
    sim = Simulator(freq_array=freqs, time_array=times, array_layout=ants)
    gains = gen_gains(freqs=freqs_GHz, ants=ants)

    # Write the gains to disk three ways
    cal_file_1 = str(tmp_path / "test1.calfits")
    cal_file_2 = str(tmp_path / "test2.calfits")
    cal_file_3 = str(tmp_path / "test3.calfits")
    cli_utils.write_calfits(gains, cal_file_1, freqs=freqs, times=times)
    cli_utils.write_calfits(gains, cal_file_2, sim=sim)
    cli_utils.write_calfits(gains, cal_file_3, sim=sim.data)

    # Check that everything checks out
    for cal_file in (cal_file_1, cal_file_2, cal_file_3):
        uvc = UVCal()
        uvc.read_calfits(cal_file)
        assert np.allclose(uvc.freq_array.flatten(), freqs)
        assert np.allclose(uvc.time_array, times)
        assert np.allclose(uvc.get_gains(0, "Jee").mean(axis=1), gains[0])

    # Now check that the appropriate errors are raised.
    with pytest.raises(TypeError) as err:
        cli_utils.write_calfits(gains, cal_file_1, sim=freqs)
    assert err.value.args[0] == "sim must be a Simulator or UVData object."

    with pytest.raises(ValueError) as err:
        cli_utils.write_calfits(gains, cal_file_1)
    assert "frequencies and times must be specified" in err.value.args[0]
