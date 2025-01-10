import numpy as np
import pytest

from hera_sim import io


@pytest.fixture(scope="function")
def time_params():
    return {"start_time": 2458119.5, "integration_time": 10.7, "Ntimes": 20}


@pytest.fixture(scope="function")
def freq_params():
    return {"start_freq": 1e8, "channel_width": 1e6, "Nfreqs": 150}


@pytest.fixture(scope="function")
def array_layout():
    return {0: [0, 0, 0], 1: [14.6, 0, 0], 2: [0, 14.6, 0]}


@pytest.fixture(scope="function")
def sample_uvd(time_params, freq_params, array_layout):
    full_params = dict(**time_params, **freq_params, array_layout=array_layout)
    return io.empty_uvdata(**full_params)


@pytest.mark.parametrize("axis", ["time", "freq"])
def test_empty_uvdata_basics(time_params, freq_params, sample_uvd, axis):
    all_params = dict(**time_params, **freq_params)
    start_ok = np.isclose(
        getattr(sample_uvd, f"{axis}_array").min(), all_params[f"start_{axis}"]
    )
    diff = {"time": "integration_time", "freq": "channel_width"}[axis]
    diff_ok = np.isclose(np.mean(getattr(sample_uvd, diff)), all_params[diff])
    count_ok = getattr(sample_uvd, f"N{axis}s") == all_params[f"N{axis}s"]
    assert all([start_ok, diff_ok, count_ok])


def test_empty_uvdata_antenna_positions(array_layout, sample_uvd):
    assert all(
        any(np.allclose(position, expectation) for expectation in array_layout.values())
        for position in sample_uvd.get_ENU_antpos()[0]
    )


def test_conj_convention(time_params, freq_params, array_layout):
    uvd = io.empty_uvdata(
        array_layout=array_layout, conjugation="ant1<ant2", **time_params, **freq_params
    )
    for ai, aj in uvd.get_antpairs():
        assert ai <= aj


@pytest.mark.parametrize(
    "old_param,new_param",
    [("n_times", "Ntimes"), ("n_freq", "Nfreqs"), ("antennas", "array_layout")],
)
def test_deprecation_warning(
    time_params, freq_params, array_layout, old_param, new_param
):
    all_params = dict(**time_params, **freq_params, array_layout=array_layout)
    all_params[old_param] = all_params[new_param]
    del all_params[new_param]
    with pytest.deprecated_call():
        io.empty_uvdata(**all_params)


def test_chunker_using_ref_files(sample_uvd, tmp_path):
    io.chunk_sim_and_save(sample_uvd, tmp_path, Nint_per_file=5, prefix="zen")
    ref_files = sorted(
        tmp_path / f for f in tmp_path.iterdir() if f.name.startswith("zen")
    )
    io.chunk_sim_and_save(sample_uvd, tmp_path, ref_files=ref_files, prefix="hor")
    new_files = sorted(
        tmp_path / f for f in tmp_path.iterdir() if f.name.startswith("hor")
    )
    print([f.name[4:] for f in ref_files])
    print([f.name[4:] for f in new_files])
    assert all(
        ref_file.name[4:] == new_file.name[4:]
        for ref_file, new_file in zip(ref_files, new_files)
    )


def test_chunker_error_uvdata_only():
    with pytest.raises(ValueError) as err:
        io.chunk_sim_and_save(None, ".")
    assert err.value.args[0] == "sim_uvd must be a UVData object."


def test_chunker_error_bad_filetype(sample_uvd):
    with pytest.raises(ValueError) as err:
        io.chunk_sim_and_save(sample_uvd, ".", filetype="unsupported")
    assert err.value.args[0] == "Write method not supported."


def test_chunker_error_no_reference(sample_uvd):
    with pytest.raises(ValueError) as err:
        io.chunk_sim_and_save(sample_uvd, ".")
    assert "reference files or the number of" in err.value.args[0]
