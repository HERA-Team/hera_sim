import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline, interp1d

from hera_sim.interpolators import Bandpass, Beam, Reflection, Tsky, _check_path, _read

INTERPOLATORS = {"beam": Beam, "bandpass": Bandpass, "tsky": Tsky}


@pytest.fixture(scope="function")
def freqs():
    return np.linspace(0.1, 0.2, 100, endpoint=False)


@pytest.fixture(scope="function")
def lsts():
    return np.linspace(0, 2 * np.pi, 200, endpoint=False)


@pytest.fixture(scope="function")
def Tsky_mdl(freqs, lsts):
    mfreq = 0.150
    Trx = 100
    dT = 5
    beta = -1.5
    # Tsky array needs to have shape (lsts.size, freqs.size)
    lsts, freqs = np.meshgrid(lsts, freqs, indexing="ij")
    return (freqs / mfreq) ** beta * (dT * np.sin(lsts) + Trx)


@pytest.fixture(scope="function")
def tsky(freqs, lsts, Tsky_mdl, tmp_path):
    # Write the data to disk to load later.
    data_file = str(tmp_path / "sample_data.npz")
    pols = ("xx",)
    meta = {"pols": pols}
    np.savez(data_file, tsky=Tsky_mdl[None, :, :], lsts=lsts, freqs=freqs, meta=meta)
    # Load the data in a Tsky object.
    return Tsky(data_file)


def test_tsky_interpolator_type(tsky):
    assert isinstance(tsky._interpolator, RectBivariateSpline)


def test_tsky_interpolator_accuracy(freqs, lsts, Tsky_mdl, tsky):
    tsky = tsky(lsts, freqs)
    assert np.allclose(tsky, Tsky_mdl)


def test_tsky_interpolator_shape(tsky):
    freqs = np.linspace(0.12, 0.18, 500)
    lsts = np.linspace(np.pi / 4, 3 * np.pi / 4, 130)
    resampled_tsky = tsky(lsts, freqs)
    assert resampled_tsky.shape == (lsts.size, freqs.size)


def construct_interpolator(freqs, model, interpolator, file_base):
    # Save the data to be loaded into an interpolation object
    data = np.ones(freqs.size)
    if interpolator == "poly1d":
        coefficients = np.polyfit(freqs, data, 5)
        data_file = f"{file_base}.npy"
        np.save(data_file, coefficients)
    elif interpolator == "interp1d":
        params = {"freqs": freqs, model: data}
        data_file = f"{file_base}.npz"
        np.savez(data_file, **params)
    return INTERPOLATORS[model](data_file, interpolator=interpolator)


@pytest.mark.parametrize("model", ["beam", "bandpass"])
@pytest.mark.parametrize("interpolator", ["poly1d", "interp1d"])
def test_1d_interpolator_types(freqs, model, interpolator, tmp_path):
    data_file = str(tmp_path / f"type_test_{model}_{interpolator}")
    expected_type = {"poly1d": np.poly1d, "interp1d": interp1d}[interpolator]
    interpolator = construct_interpolator(freqs, model, interpolator, data_file)
    assert type(interpolator._interpolator) is expected_type


@pytest.mark.parametrize("model", ["beam", "bandpass"])
@pytest.mark.parametrize("interpolator", ["poly1d", "interp1d"])
def test_1d_interpolator_shape(freqs, model, interpolator, tmp_path):
    data_file = str(tmp_path / f"shape_test_{model}_{interpolator}")
    interpolator = construct_interpolator(freqs, model, interpolator, data_file)
    new_freqs = np.linspace(0.12, 0.18, 720)
    resampled_data = interpolator(new_freqs)
    assert resampled_data.size == new_freqs.size


def test_reflection_interpolator(freqs, tmp_path):
    data_file = str(tmp_path / "sample_reflections.npz")
    mock_data = np.exp(2j * np.pi * (freqs - freqs.mean()) / freqs.max())
    np.savez(data_file, freqs=freqs[::2], reflection=mock_data[::2])
    interpolator = Reflection(data_file)
    interp_data = interpolator(freqs[1:-2:2])
    assert np.allclose(interp_data.real, mock_data[1:-2:2].real)
    assert np.allclose(interp_data.imag, mock_data[1:-2:2].imag)


def test_tsky_exception_no_freqs(lsts, Tsky_mdl, tmp_path):
    data_file = str(tmp_path / "test_tsky_no_freqs.npz")
    np.savez(data_file, lsts=lsts, tsky=Tsky_mdl[None, :, :], meta={"pols": ("xx",)})
    with pytest.raises(AssertionError) as err:
        Tsky(data_file)
    assert "frequencies corresponding to the sky temperature" in err.value.args[0]


def test_tsky_exception_no_lsts(freqs, Tsky_mdl, tmp_path):
    data_file = str(tmp_path / "test_tsky_no_lsts.npz")
    np.savez(data_file, freqs=freqs, tsky=Tsky_mdl[None, :, :], meta={"pols": ("xx",)})
    with pytest.raises(AssertionError) as err:
        Tsky(data_file)
    assert "LSTs corresponding to the sky temperature" in err.value.args[0]


def test_tsky_exception_no_tsky(freqs, lsts, tmp_path):
    data_file = str(tmp_path / "test_tsky_no_tsky.npz")
    np.savez(data_file, freqs=freqs, lsts=lsts, meta={"pols": ("xx",)})
    with pytest.raises(AssertionError) as err:
        Tsky(data_file)
    assert "sky temperature array must be saved" in err.value.args[0]


def test_tsky_exception_no_meta_dict(freqs, lsts, Tsky_mdl, tmp_path):
    data_file = str(tmp_path / "test_tsky_no_meta.npz")
    np.savez(data_file, freqs=freqs, lsts=lsts, tsky=Tsky_mdl[None, :, :])
    with pytest.raises(AssertionError) as err:
        Tsky(data_file)
    assert "npz file must contain a metadata dictionary" in err.value.args[0]


def test_tsky_exception_bad_tsky_shape(freqs, lsts, Tsky_mdl, tmp_path):
    data_file = str(tmp_path / "test_tsky_bad_tsky_shape.npz")
    np.savez(data_file, freqs=freqs, lsts=lsts, tsky=Tsky_mdl, meta={"pols": ("xx",)})
    with pytest.raises(AssertionError) as err:
        Tsky(data_file)
    assert "tsky array is incorrectly shaped." in err.value.args[0]


def test_tsky_exception_pol_not_found(freqs, lsts, Tsky_mdl, tmp_path):
    data_file = str(tmp_path / "test_tsky_pol_not_found.npz")
    np.savez(
        data_file,
        freqs=freqs,
        lsts=lsts,
        tsky=Tsky_mdl[None, :, :],
        meta={"pols": ("xx",)},
    )
    with pytest.raises(AssertionError) as err:
        Tsky(data_file, pol="yy")
    assert "Polarization must be in the metadata's" in err.value.args[0]


@pytest.mark.parametrize("model", ["beam", "bandpass"])
def test_1d_interpolators_bad_interp_type(model, tmp_path):
    data_file = str(tmp_path / f"test_{model}_bad_interp_type.npy")
    data = np.array([1, 2, 3])
    np.save(data_file, data)
    with pytest.raises(AssertionError) as err:
        {"beam": Beam, "bandpass": Bandpass}[model](data_file, interpolator="bad_type")
    assert (
        err.value.args[0]
        == "Interpolator choice must either be 'poly1d' or 'interp1d'."
    )


@pytest.mark.parametrize("model", ["beam", "bandpass"])
@pytest.mark.parametrize("interpolator", ["poly1d", "interp1d"])
def test_1d_interpolators_bad_file_ext(freqs, model, interpolator, tmp_path):
    data_file = str(tmp_path / f"test_{model}_{interpolator}_bad_ext")
    if interpolator == "poly1d":
        kwds = {"freqs": freqs, model: np.ones(freqs.size)}
        data_file = f"{data_file}.npz"
        np.savez(data_file, **kwds)
    elif interpolator == "interp1d":
        data_file = f"{data_file}.npy"
        np.save(data_file, np.array([1, 2, 3, 4, 5]))
    with pytest.raises(AssertionError) as err:
        INTERPOLATORS[model](data_file, interpolator=interpolator)
    article = {"poly1d": "a", "interp1d": "an"}[interpolator]
    assert f"In order to use {article} {interpolator!r} object" in err.value.args[0]


@pytest.mark.parametrize("model", ["beam", "bandpass"])
@pytest.mark.parametrize("param", ["freqs", "model"])
def test_1d_interpolators_missing_npz_keys(freqs, model, param, tmp_path):
    data_file = str(tmp_path / f"test_{model}_missing_{param}_key.npz")
    kwds = {"freqs": {model: np.ones(freqs.size)}, "model": {"freqs": freqs}}
    np.savez(data_file, **kwds[param])
    with pytest.raises(AssertionError) as err:
        INTERPOLATORS[model](data_file, interpolator="interp1d")
    assert "Please ensure that the `.npz` archive has" in err.value.args[0]


def test_path_checking_nonexistent_file():
    data_file = "does_not_exist.npz"
    with pytest.raises(AssertionError) as err:
        _check_path(data_file)
    assert "If datafile is not an absolute path" in err.value.args[0]


def test_file_parsing_unsupported_type():
    data_file = "something.jpg"
    with pytest.raises(ValueError) as err:
        _read(data_file)
    assert err.value.args[0] == "File type '.jpg' not supported."
