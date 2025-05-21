from os.path import join
from warnings import catch_warnings

import numpy as np
import pytest

from hera_sim.config import CONFIG_PATH
from hera_sim.defaults import defaults
from hera_sim.interpolators import Beam, Tsky
from hera_sim.sigchain import gen_bandpass


def test_config_swap():
    defaults.set("h1c")
    config1 = defaults().copy()
    defaults.set("h2c", refresh=True)
    assert config1 != defaults()


def test_direct_config_path():
    config = join(CONFIG_PATH, "H2C.yaml")
    defaults.set(config, refresh=True)
    # check some of the parameters
    assert defaults()["integration_time"] == 8.59
    assert isinstance(defaults()["Tsky_mdl"], Tsky)
    assert isinstance(defaults()["omega_p"], Beam)


def test_null_config():
    defaults.set(None, refresh=True)
    assert defaults() == {}
    defaults.deactivate()


def test_multiple_param_specification():
    config = {0: {"Nfreqs": 100}, 1: {"Nfreqs": 200}}
    with catch_warnings(record=True) as w:
        defaults.set(config, refresh=True)
        # make sure that there's an error message
        assert w[0].message != ""
    defaults.deactivate()


def test_bandpass_changes():
    defaults.set("h1c", refresh=True)
    fqs = np.linspace(0.1, 0.2, 100)
    bp = gen_bandpass(fqs, [0], rng=np.random.default_rng(0))[0]
    defaults.set("h2c", refresh=True)
    assert not np.all(bp == gen_bandpass(fqs, [0], rng=np.random.default_rng(0))[0])
    defaults.deactivate()


def test_activate_and_deactivate():
    defaults.activate()
    assert defaults._override_defaults
    defaults.deactivate()
    assert not defaults._override_defaults


def test_dict_unpacking():
    config = {
        "setup": {
            "frequency_array": {"Nfreqs": 100, "start_freq": 100e6},
            "time_array": {"Ntimes": 50, "start_time": 2e6},
        },
        "telescope": {"omega_p": np.ones(100)},
    }
    defaults.set(config, refresh=True)
    for value in defaults().values():
        assert not isinstance(value, dict)
    defaults.deactivate()


def test_refresh():
    # choose some defaults to start with
    defaults.set("h1c")
    # now use new, simple defaults
    config = {"Nfreqs": 100}
    defaults.set(config, refresh=True)
    # check that refresh is working
    assert "Ntimes" not in defaults()
    assert "Nfreqs" in defaults()
    # default behavior is to not refresh, just update
    config = {"Nfreqs": 200, "Ntimes": 50}
    defaults.set(config)
    assert "Ntimes" in defaults()
    assert defaults("Nfreqs") == 200
    defaults.deactivate()


def test_call_with_bad_component():
    defaults.set("h1c")
    with pytest.raises(KeyError) as err:
        defaults("not_a_valid_key")
    assert "not_a_valid_key not found in configuration." == err.value.args[0]
    defaults.deactivate()


def test_bad_config_type():
    not_a_string = 1
    with pytest.raises(ValueError) as err:
        defaults.set(not_a_string)
    assert "The configuration must be a" in err.value.args[0]
    defaults.deactivate()


def test_bad_config_file():
    with pytest.raises(FileNotFoundError):
        defaults.set("not_a_file")
    defaults.deactivate()
