import numpy as np
import pytest

from hera_sim import eor


@pytest.fixture(scope="function")
def freqs():
    return np.linspace(0.1, 0.2, 500)


@pytest.fixture(scope="function")
def lsts():
    return np.linspace(0, 1, 500)


@pytest.fixture(scope="function")
def bl_vec():
    return np.array([50.0, 0, 0])


@pytest.fixture(scope="function")
def base_eor(freqs, lsts, bl_vec, request):
    if request.param == "auto":
        bl_vec = np.array([0, 0, 0])
    return eor.noiselike_eor(
        lsts=lsts,
        freqs=freqs,
        bl_vec=bl_vec,
        eor_amp=1e-5,
        fringe_filter_type="tophat",
        rng=np.random.default_rng(0),
    )


@pytest.fixture(scope="function")
def scaled_eor(freqs, lsts, bl_vec, request):
    if request.param == "auto":
        bl_vec = np.array([0, 0, 0])
    return eor.noiselike_eor(
        lsts=lsts,
        freqs=freqs,
        bl_vec=bl_vec,
        eor_amp=1e-3,
        fringe_filter_type="tophat",
        rng=np.random.default_rng(0),
    )


@pytest.mark.parametrize("base_eor", ["auto"], indirect=True)
def test_noiselike_eor_autocorr_is_real(base_eor):
    assert np.all(base_eor.imag == 0)


@pytest.mark.parametrize("base_eor", ["cross"], indirect=True)
def test_noiselike_eor_is_noiselike_in_freq(base_eor):
    covariance = np.cov(base_eor.T)  # Compute covariance across freq axis.
    mean_diagonal = np.mean(covariance.diagonal())
    mean_offdiagonal = np.mean(
        covariance - np.eye(len(covariance)) * covariance.diagonal()
    )
    assert np.abs(mean_diagonal / mean_offdiagonal) > 1000
    # To manually check: plt.matshow(np.abs(covariance))


@pytest.mark.parametrize("base_eor", ["cross"], indirect=True)
def test_noiselike_eor_is_sky_locked(base_eor):
    covariance = np.cov(base_eor)  # Compute covariance across time axis.
    mean_diagonal = np.mean(covariance.diagonal())
    mean_offdiagonal = np.mean(
        covariance - np.eye(len(covariance)) * covariance.diagonal()
    )
    assert np.abs(mean_diagonal / mean_offdiagonal) < 20


@pytest.mark.parametrize("base_eor, scaled_eor", [("cross",) * 2], indirect=True)
def test_noiselike_eor_crosscorr_scales_appropriately(base_eor, scaled_eor):
    assert np.isclose(
        np.mean(np.abs(base_eor / scaled_eor)) / np.sqrt(2), 1e-2, atol=0.01
    )


@pytest.mark.parametrize("base_eor, scaled_eor", [("auto",) * 2], indirect=True)
def test_noiselike_eor_autocorr_scales_appropriately(base_eor, scaled_eor):
    assert np.isclose(np.mean(np.abs(base_eor / scaled_eor)) / 2, 1e-2, atol=0.01)
