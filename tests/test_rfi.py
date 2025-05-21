import numpy as np
import pytest

from hera_sim import DATA_PATH, rfi


@pytest.fixture(scope="function")
def freqs():
    return np.round(np.linspace(0.1, 0.2, 101), 3)


@pytest.fixture(scope="function")
def lsts():
    return np.linspace(0, 2 * np.pi, 200)


@pytest.mark.parametrize("station_freq", [0.150, 0.1505])
def test_rfi_station_strength(freqs, lsts, station_freq):
    # Generate RFI for a single station.
    station = rfi.RfiStation(station_freq, std=0.0, rng=np.random.default_rng(0))
    rfi_vis = station(lsts, freqs)

    # Check that the RFI is inserted where it should be at the correct level.
    base_index = list(freqs).index(0.15)
    test_indices = [base_index - 1, base_index, base_index + 1]
    if station_freq == 0.150:
        expected_values = [0, station.strength, 0]
    else:
        expected_values = [0, station.strength / 2, station.strength / 2]

    for index, expected_value in zip(test_indices, expected_values):
        assert np.allclose(rfi_vis[:, index], expected_value, 4)


@pytest.mark.parametrize("rfi_type", ["stations", "impulse", "scatter", "dtv"])
def test_rfi_shape(freqs, lsts, rfi_type):
    if rfi_type == "stations":
        kwargs = {"stations": [[0.150, 1.0, 100.0, 10.0, 100.0]]}
    else:
        kwargs = {}

    rfi_vis = getattr(rfi, f"rfi_{rfi_type}")(lsts, freqs, **kwargs)
    assert rfi_vis.shape == (lsts.size, freqs.size)


def test_rfi_station_from_file(freqs, lsts):
    filename = DATA_PATH / "HERA_H1C_RFI_STATIONS.npy"
    station_params = np.load(filename)
    Nstations = station_params.shape[0]
    rfi_vis = rfi.rfi_stations(
        lsts, freqs, stations=filename, rng=np.random.default_rng(0)
    )
    assert np.sum(np.sum(np.abs(rfi_vis), axis=0).astype(bool)) >= Nstations


@pytest.mark.parametrize("rfi_type", ["scatter", "impulse", "dtv"])
@pytest.mark.parametrize("chance", [0.2, 0.5, 0.8])
def test_rfi_occupancy(freqs, lsts, rfi_type, chance):
    kwargs = {f"{rfi_type}_chance": chance, "rng": np.random.default_rng(0)}
    # Modify expected chance appropriately if simulating DTV.
    if rfi_type == "dtv":
        fmin, fmax = (0.15, 0.20)
        kwargs["dtv_band"] = (fmin, fmax)
        freq_cut = (freqs >= fmin) & (freqs < fmax)
        chance *= freqs[freq_cut].size / freqs.size
    rfi_vis = getattr(rfi, f"rfi_{rfi_type}")(lsts, freqs, **kwargs)
    assert np.isclose(np.mean(np.abs(rfi_vis).astype(bool)), chance, atol=0.1, rtol=0)


@pytest.mark.parametrize("alignment", ["aligned", "misaligned"])
def test_rfi_dtv_constant_across_subband(freqs, lsts, alignment):
    dtv_band = (0.15, 0.20) if alignment == "aligned" else (0.1502, 0.2002)
    channel_width = 0.01
    Nbands = 5  # Just hardcode this to avoid stupid numerical problems.
    subband_edges = np.linspace(dtv_band[0], dtv_band[1], Nbands + 1)
    rfi_vis = rfi.rfi_dtv(
        lsts,
        freqs,
        dtv_band=dtv_band,
        dtv_channel_width=channel_width,
        dtv_chance=0.5,
        dtv_strength=10,
        dtv_std=1,
        rng=np.random.default_rng(0),
    )
    for band_edges in zip(subband_edges[:-1], subband_edges[1:]):
        channels = np.argwhere(
            (freqs >= band_edges[0]) & (freqs < band_edges[1])
        ).flatten()
        for reference_channel in channels:
            assert np.allclose(
                np.mean(rfi_vis[:, channels], axis=1),
                rfi_vis[:, reference_channel],
                rtol=0.01,
                atol=0,
            )


def test_rfi_dtv_occupancy_variable_chance(freqs, lsts):
    dtv_band = (0.15, 0.20)
    channel_width = 0.01
    Nbands = 5  # See above note about hardcoding.
    subband_edges = np.linspace(0.15, 0.20, Nbands + 1)
    chances = 0.2 + 0.1 * np.arange(Nbands)
    rfi_vis = rfi.rfi_dtv(
        lsts,
        freqs,
        dtv_band=dtv_band,
        dtv_channel_width=channel_width,
        dtv_chance=chances,
        dtv_strength=10,
        dtv_std=1,
        rng=np.random.default_rng(0),
    )
    expected_occupancy = sum(
        chance
        * freqs[(freqs >= subband_edges[i]) & (freqs < subband_edges[i + 1])].size
        / freqs.size
        for i, chance in enumerate(chances)
    )
    assert np.isclose(
        np.mean(np.abs(rfi_vis).astype(bool)), expected_occupancy, atol=0.05, rtol=0
    )


def test_rfi_dtv_warning_when_no_overlap(freqs, lsts):
    dtv_band = (0.25, 0.37)
    with pytest.warns(UserWarning) as warning:
        rfi.rfi_dtv(lsts, freqs, dtv_band=dtv_band)
    assert "DTV band does not overlap" in warning[0].message.args[0]


def test_rfi_dtv_bad_parameters(freqs, lsts):
    with pytest.raises(ValueError) as err:
        rfi.rfi_dtv(
            lsts, freqs, dtv_band=(0.15, 0.2), dtv_chance=[0.1, 0.2], dtv_std=[1, 2, 3]
        )
    assert "not formatted properly." in err.value.args[0]


def test_rfi_station_out_of_band(freqs, lsts):
    freq = 0.27
    station = rfi.RfiStation(freq)
    rfi_vis = station(lsts, freqs)
    assert np.allclose(rfi_vis, 0, atol=1e-7, rtol=0)


def test_rfi_stations_unspecified_stations(freqs, lsts):
    with pytest.warns(UserWarning) as warning:
        rfi_vis = rfi.rfi_stations(lsts, freqs, stations=None)
    assert warning[0].message.args[0] == "You did not specify any stations to simulate."
    assert np.allclose(rfi_vis, 0, atol=1e-7, rtol=0)


def test_rfi_station_bad_station_parameters(freqs, lsts):
    stations = [[0.1, 0.2, 0.3]]
    with pytest.raises(ValueError) as err:
        rfi.rfi_stations(lsts, freqs, stations=stations)
    assert "Please check the format of your stations." in err.value.args[0]
