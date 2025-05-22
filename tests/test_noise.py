from contextlib import contextmanager

import numpy as np
import pytest
from astropy import units

from hera_sim import DATA_PATH, noise, utils
from hera_sim.defaults import defaults
from hera_sim.interpolators import Beam

# Ensure that defaults aren't subtly overwritten.
defaults.deactivate()


@contextmanager
def does_not_raise():
    yield


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
    noise_Jy = noise.sky_noise_jy(
        lsts=lsts,
        freqs=freqs,
        Tsky_mdl=None,
        omega_p=omega_p,
        channel_width=channel_width,
        integration_time=integration_time,
        rng=np.random.default_rng(0),
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


@pytest.mark.parametrize(
    "autovis,expectation",
    [(None, pytest.raises(NotImplementedError)), (True, does_not_raise())],
)
def test_thermal_noise_with_phase_wrap(freqs, omega_p, autovis, expectation):
    dlst = np.pi / 180
    wrapped_lsts = np.linspace(2 * np.pi - dlst, 2 * np.pi + dlst, 50)
    integration_time = (
        np.mean(np.diff(wrapped_lsts)) * units.day.to("s") * units.rad.to("cycle")
    )
    wrapped_lsts %= 2 * np.pi
    channel_width = np.mean(np.diff(freqs)) * units.GHz.to("Hz")
    expected_SNR = np.sqrt(integration_time * channel_width)
    Trx = 0
    rng = np.random.default_rng(0)
    if autovis is not None:
        autovis = np.ones((wrapped_lsts.size, freqs.size), dtype=complex)
    noise_sim = noise.ThermalNoise(
        Tsky_mdl=noise.HERA_Tsky_mdl["xx"],
        omega_p=omega_p,
        Trx=Trx,
        autovis=autovis,
        rng=rng,
    )
    with expectation:
        vis = noise_sim(lsts=wrapped_lsts, freqs=freqs)
        np.testing.assert_allclose(
            np.std(vis), 1 / expected_SNR, rtol=1 / np.sqrt(vis.size)
        )
