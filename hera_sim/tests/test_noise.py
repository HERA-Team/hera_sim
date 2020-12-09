import os
import pytest
import unittest

import numpy as np
from astropy import units

from hera_sim import noise, utils
from hera_sim import DATA_PATH
from hera_sim.interpolators import Beam
from hera_sim.defaults import defaults

# Ensure that defaults aren't subtly overwritten.
defaults.deactivate()


@pytest.fixture(scope="function")
def freqs():
    return np.linspace(0.1, 0.2, 100)


@pytest.fixture(scope="function")
def lsts():
    return np.linspace(0, 2 * np.pi, 200)


@pytest.fixture(scope="function")
def omega_p(freqs):
    return Beam(DATA_PATH / "HERA_H1C_BEAM_POLY.npy")(freqs)


@pytest.fixture(scope="function")
def Jy2T(freqs, omega_p):
    return utils.Jy2T(freqs, omega_p).reshape(1, -1)


@pytest.fixture(scope="function")
def tsky_powerlaw(freqs, lsts):
    return noise.resample_Tsky(
        lsts, freqs, Tsky_mdl=None, Tsky=180.0, mfreq=0.18, index=-2.5
    )


@pytest.fixture(scope="function")
def tsky_from_model(lsts, freqs):
    return noise.resample_Tsky(lsts, freqs, noise.HERA_Tsky_mdl["xx"])


@pytest.mark.parametrize("model", ["powerlaw", "HERA"])
def test_resample_Tsky_shape(freqs, lsts, tsky_powerlaw, tsky_from_model, model):
    tsky = tsky_powerlaw if model == "powerlaw" else tsky_from_model
    assert tsky.shape == (lsts.size, freqs.size)


@pytest.mark.parametrize("model", ["powerlaw", "HERA"])
def test_resample_Tsky_time_behavior(tsky_powerlaw, tsky_from_model, model):
    if model == "powerlaw":
        assert all(
            np.all(tsky_powerlaw[0] == tsky_powerlaw[i])
            for i in range(1, tsky_powerlaw.shape[0])
        )
    else:
        assert all(
            np.all(tsky_from_model[0] != tsky_from_model[i])
            for i in range(1, tsky_from_model.shape[0])
        )


@pytest.mark.parametrize("model", ["powerlaw", "HERA"])
def test_resample_Tsky_freq_behavior(tsky_powerlaw, tsky_from_model, model):
    tsky = tsky_powerlaw if model == "powerlaw" else tsky_from_model
    assert all(np.all(tsky[:, 0] != tsky[:, i]) for i in range(1, tsky.shape[1]))


@pytest.mark.parametrize("channel_width", [None, 1e6, 1e8])
@pytest.mark.parametrize("integration_time", [None, 10.7, 33.7])
@pytest.mark.parametrize("aspect", ["mean", "std"])
def test_sky_noise_jy(
    freqs, lsts, tsky_powerlaw, Jy2T, omega_p, channel_width, integration_time, aspect
):
    np.random.seed(0)
    noise_Jy = noise.sky_noise_jy(
        lsts=lsts,
        freqs=freqs,
        Tsky_mdl=None,
        omega_p=omega_p,
        channel_width=channel_width,
        integration_time=integration_time,
    )

    # Calculate expected noise level based on radiometer equation.
    if aspect == "mean":
        expected_noise_Jy = 0
        atol = 0.7
        rtol = 0
    else:
        channel_width = channel_width or np.mean(np.diff(freqs)) * units.GHz.to("Hz")
        integration_time = integration_time or units.day.to("s") / lsts.size
        expected_noise_Jy = np.mean(tsky_powerlaw, axis=0) / Jy2T
        expected_noise_Jy /= np.sqrt(channel_width * integration_time)
        atol = 0
        rtol = 0.1
    assert np.allclose(
        getattr(np, aspect)(noise_Jy, axis=0), expected_noise_Jy, rtol=rtol, atol=atol
    )
