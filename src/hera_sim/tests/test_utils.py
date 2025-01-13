from itertools import combinations

import numpy as np
import pytest
from astropy import units
from pyuvdata import utils as uvutils

from hera_sim import DATA_PATH, Simulator, defaults, utils
from hera_sim.interpolators import Beam


@pytest.fixture(scope="function")
def lsts():
    return np.linspace(0, 2 * np.pi, 200)


@pytest.fixture(scope="function")
def fringe_rates(lsts):
    dlst = np.mean(np.diff(lsts))
    dlst_sec = dlst * units.rad.to("cycle") * units.sday.to("s")
    return np.fft.fftfreq(lsts.size, dlst_sec)


@pytest.fixture(scope="function")
def freqs():
    return np.linspace(0.1, 0.2, 100)


@pytest.fixture(scope="function")
def delays(freqs):
    df = np.mean(np.diff(freqs))
    return np.fft.fftfreq(freqs.size, df)


@pytest.mark.parametrize("bl_len_ns", [50, 100])
@pytest.mark.parametrize("standoff", [0, 20])
def test_gen_delay_filter_tophat(freqs, delays, bl_len_ns, standoff):
    delay_filter = utils.gen_delay_filter(
        freqs, bl_len_ns, standoff=standoff, delay_filter_type="tophat"
    )
    expected_filter = np.ones_like(delays)
    expected_filter[np.abs(delays) > bl_len_ns + standoff] = 0
    assert np.allclose(delay_filter, expected_filter, atol=1e-7)


@pytest.mark.parametrize("filter_type", [None, "none", "None"])
def test_gen_delay_filter_none(freqs, delays, filter_type):
    delay_filter = utils.gen_delay_filter(freqs, 120, delay_filter_type=filter_type)
    assert np.all(delay_filter == 1)


@pytest.mark.parametrize("filter_type", [None, "tophat", "gauss", "trunc_gauss"])
@pytest.mark.parametrize("min_delay", [None, 50])
@pytest.mark.parametrize("max_delay", [None, 250])
def test_gen_delay_filter_bounded_delays(
    freqs, delays, min_delay, max_delay, filter_type
):
    delay_filter = utils.gen_delay_filter(
        freqs,
        100,
        min_delay=min_delay,
        max_delay=max_delay,
        delay_filter_type=filter_type,
    )
    min_delay = min_delay or 0
    max_delay = max_delay or np.inf
    region_of_interest = (np.abs(delays) >= min_delay) & (np.abs(delays) <= max_delay)
    assert np.all(delay_filter[~region_of_interest] == 0)


@pytest.mark.parametrize("normalize", [np.pi, 293])
def test_gen_delay_filter_normalize(freqs, normalize):
    delay_filter = utils.gen_delay_filter(
        freqs, 100, delay_filter_type=None, normalize=normalize
    )
    assert np.allclose(delay_filter, normalize, atol=1e-7)


@pytest.mark.parametrize("bl_len_ns", [50, 100])
@pytest.mark.parametrize("standoff", [0, 20])
@pytest.mark.parametrize("filter_type", ["gauss", "trunc_gauss"])
def test_gen_delay_filter_gauss(freqs, delays, bl_len_ns, standoff, filter_type):
    delay_filter = utils.gen_delay_filter(
        freqs, bl_len_ns, standoff=standoff, delay_filter_type=filter_type
    )
    one_sigma = (bl_len_ns + standoff) / 4
    expected_filter = np.exp(-0.5 * (delays / one_sigma) ** 2)
    if filter_type == "trunc_gauss":
        expected_filter[np.abs(delays) > bl_len_ns + standoff] = 0
    assert np.allclose(delay_filter, expected_filter, atol=1e-7)


@pytest.mark.parametrize("bl_len_ns", [[10, 0], [10, 0, 0]])
def test_gen_delay_filter_vector_bl_len_ns(freqs, delays, bl_len_ns):
    delay_filter = utils.gen_delay_filter(
        freqs, np.array(bl_len_ns), standoff=0, delay_filter_type="tophat"
    )
    expected_filter = np.ones_like(delays)
    expected_filter[np.abs(delays) > np.linalg.norm(bl_len_ns)] = 0
    assert np.all(delay_filter == expected_filter)


def test_gen_delay_filter_bad_filter_type(freqs):
    with pytest.raises(ValueError) as err:
        utils.gen_delay_filter(freqs, 0, delay_filter_type="bad filter")
    assert err.value.args[0] == "Didn't recognize filter_type bad filter"


@pytest.mark.parametrize("filter_type", ["delay", "fringe"])
def test_rough_filter_noisy_data(freqs, lsts, filter_type):
    Nrealizations = 1000
    mean_values = np.zeros((Nrealizations, 2))
    if filter_type == "delay":
        filt = utils.rough_delay_filter
        args = [freqs, 50]
        kwargs = {"standoff": 0, "delay_filter_type": "gauss"}
    else:
        filt = utils.rough_fringe_filter
        args = [lsts, freqs, 50]
        kwargs = {"fringe_filter_type": "gauss", "fr_width": 1e-4}
    rng = np.random.default_rng(0)
    for i in range(Nrealizations):
        data = utils.gen_white_noise((lsts.size, freqs.size), rng=rng)
        filtered_data = filt(data, *args, **kwargs)
        filtered_data_mean = np.mean(filtered_data)
        mean_values[i] = filtered_data_mean.real, filtered_data_mean.imag
    one_sigma = 1 / np.sqrt(freqs.size * lsts.size)
    # White noise is, on average, still white noise after a Gaussian filter.
    assert np.isclose(
        np.sum(np.abs(mean_values) < 3 * one_sigma) / mean_values.size,
        1,
        atol=0.01,
        rtol=0,
    )


@pytest.mark.parametrize("missing_param", ["freqs", "bl_len_ns"])
def test_rough_delay_filter_missing_param(freqs, lsts, missing_param):
    data = np.zeros((lsts.size, freqs.size))
    kwargs = {"freqs": freqs, "bl_len_ns": 100, "delay_filter": None}
    kwargs[missing_param] = None
    with pytest.raises(ValueError) as err:
        utils.rough_delay_filter(data, **kwargs)
    assert f"you must provide {missing_param}" in err.value.args[0]


# TODO: figure out why this test passes--it should just be a little math.
def test_delay_filter_norm(freqs):
    tsky = np.ones(freqs.size)

    rng = np.random.default_rng(0)  # set the seed for reproducibility.

    out = 0
    nreal = 5000
    for _ in range(nreal):
        _noise = tsky * utils.gen_white_noise(freqs.size, rng=rng)
        outnoise = utils.rough_delay_filter(_noise, freqs, 30, normalize=1)

        out += np.sum(np.abs(outnoise) ** 2)

    out /= nreal

    assert np.isclose(out, np.sum(tsky**2), atol=0, rtol=1e-2)


@pytest.mark.parametrize("bl_len_ns", [50, 150])
def test_fringe_filter_tophat_symmetry(freqs, lsts, bl_len_ns):
    fringe_filter = utils.gen_fringe_filter(
        lsts, freqs, bl_len_ns, fringe_filter_type="tophat"
    )
    assert all(
        np.allclose(fringe_filter[i], fringe_filter[-i])
        for i in range(1, lsts.size // 2)
    )


@pytest.mark.parametrize("bl_len_ns", [50, 150])
def test_fringe_filter_tophat_output(freqs, lsts, fringe_rates, bl_len_ns):
    fringe_filter = utils.gen_fringe_filter(
        lsts, freqs, bl_len_ns, fringe_filter_type="tophat"
    )
    max_fringe_rates = utils.calc_max_fringe_rate(freqs, bl_len_ns)
    # Check that we get a low-pass fringe-rate filter.
    assert all(
        np.all(fringe_filter[np.abs(fringe_rates) > max_fringe_rate, i] == 0)
        & np.all(fringe_filter[np.abs(fringe_rates) < max_fringe_rate, i] == 1)
        for i, max_fringe_rate in enumerate(max_fringe_rates)
    )


@pytest.mark.parametrize("bl_len_ns", [50, 150])
def test_fringe_filter_none(freqs, lsts, bl_len_ns):
    fringe_filter = utils.gen_fringe_filter(
        lsts, freqs, bl_len_ns, fringe_filter_type="none"
    )
    assert np.all(fringe_filter == 1)


# 75 ns baseline is about the upper bound for the measured fringe rates.
# If we want to use a longer baseline for testing, we need finer LST resolution.
@pytest.mark.parametrize("bl_len_ns", [50, 75])
def test_fringe_filter_gauss_center(freqs, lsts, fringe_rates, bl_len_ns):
    fringe_filter = utils.gen_fringe_filter(
        lsts, freqs, bl_len_ns, fringe_filter_type="gauss", fr_width=1e-4
    )
    max_fringe_rates = utils.calc_max_fringe_rate(freqs, bl_len_ns)
    # Check that the peak of the filter is at the maximum fringe rate.
    assert np.allclose(
        fringe_rates[np.argmax(fringe_filter, axis=0)],
        max_fringe_rates,
        rtol=0,
        atol=fringe_rates[1] - fringe_rates[0],
    )


@pytest.mark.parametrize("fr_width", [1e-4, 3e-4])
def test_fringe_filter_gauss_width(freqs, lsts, fringe_rates, fr_width):
    bl_len_ns = 50.0
    fringe_filter = utils.gen_fringe_filter(
        lsts, freqs, bl_len_ns, fringe_filter_type="gauss", fr_width=fr_width
    )
    half_max_fringe_rates = fringe_rates[np.argmin(np.abs(fringe_filter - 0.5), axis=0)]
    fringe_fwhm = 2 * fr_width * np.sqrt(2 * np.log(2))
    max_fringe_rates = utils.calc_max_fringe_rate(freqs, bl_len_ns)
    # Check that the half-max occurs at the correct locations.
    assert np.all(
        np.isclose(
            half_max_fringe_rates,
            max_fringe_rates - fringe_fwhm / 2,
            rtol=0,
            atol=fringe_rates[1] - fringe_rates[0],
        )
        | np.isclose(
            half_max_fringe_rates,
            max_fringe_rates + fringe_fwhm / 2,
            rtol=0,
            atol=fringe_rates[1] - fringe_rates[0],
        )
    )


def test_fringe_filter_custom(freqs, lsts, fringe_rates):
    fringe_filter_npz = np.load(DATA_PATH / "H37_FR_Filters_small.npz")
    model_filter = fringe_filter_npz["PB_rms"][0].T
    filter_freqs = fringe_filter_npz["freqs"] / 1e9
    filter_frates = fringe_filter_npz["frates"]
    bl_len_ns = 20  # This doesn't matter for this test.
    fringe_filter = utils.gen_fringe_filter(
        lsts,
        freqs,
        bl_len_ns,
        fringe_filter_type="custom",
        FR_filter=model_filter,
        FR_frates=filter_frates,
        FR_freqs=filter_freqs,
    )
    peak_frates_model = filter_frates[np.argmax(model_filter, axis=0)]
    peak_frates_interp = fringe_rates[np.argmax(fringe_filter, axis=0)]
    nearest_neighbors = np.array(
        [np.argmin(np.abs(filter_freqs - freq)) for freq in freqs]
    )
    # Check that the filters peak at roughly the same fringe rates.
    assert np.allclose(
        peak_frates_model[nearest_neighbors], peak_frates_interp, rtol=0.05, atol=0
    )


@pytest.mark.parametrize("bl_len_ns", [50, 150])
@pytest.mark.parametrize("fr_width", [1e-4, 3e-4])
def test_rough_fringe_filter_noisy_data(freqs, lsts, fringe_rates, bl_len_ns, fr_width):
    data = utils.gen_white_noise((lsts.size, freqs.size), rng=np.random.default_rng(0))
    max_fringe_rates = utils.calc_max_fringe_rate(freqs, bl_len_ns)
    filt_data = utils.rough_fringe_filter(
        data, lsts, freqs, bl_len_ns, fringe_filter_type="gauss", fr_width=fr_width
    )
    filt_data_fft = np.abs(np.fft.fft(filt_data, axis=0))
    # Check that there is no power at fringe-rates that have been filtered.
    # Note that not all power outside of the fringe filter is actually removed;
    # the use of a FFT-based filter causes power to leak. 5-sigma apparently isn't
    # enough to make it consistent with zero to machine precision, but 7-sigma
    # seems to work.
    assert all(
        np.allclose(
            filt_data_fft[np.abs(fringe_rates - max_fringe_rate) > 7 * fr_width, i],
            0,
            rtol=0,
            atol=1e-7,
        )
        for i, max_fringe_rate in enumerate(max_fringe_rates)
    )


def test_fringe_filter_bad_type(freqs, lsts):
    with pytest.raises(ValueError) as err:
        utils.gen_fringe_filter(lsts, freqs, 35, fringe_filter_type="bad type")
    assert err.value.args[0] == "filter_type bad type not recognized"


@pytest.mark.parametrize("missing_param", ["lsts", "freqs", "ew_bl_len_ns"])
def test_rough_fringe_filter_missing_param(lsts, freqs, missing_param):
    data = np.zeros((lsts.size, freqs.size))
    kwargs = {"lsts": lsts, "freqs": freqs, "ew_bl_len_ns": 10, "fringe_filter": None}
    kwargs[missing_param] = None
    with pytest.raises(ValueError) as err:
        utils.rough_fringe_filter(data, **kwargs)
    assert "Must provide 'lsts', 'freqs' and 'ew_bl_len_ns'" in err.value.args[0]


@pytest.mark.parametrize("filter_type", ["delay", "fringe"])
def test_use_pre_computed_filter(freqs, lsts, filter_type):
    data = np.outer(np.exp(2j * np.pi * lsts * 20), np.exp(2j * np.pi * freqs * 1000))
    filt = np.ones(data.shape)
    kwargs = {f"{filter_type}_filter": filt}
    filt_data = getattr(utils, f"rough_{filter_type}_filter")(data, **kwargs)
    assert np.allclose(data, filt_data)


@pytest.mark.parametrize("shape", [100, (100, 200)])
def test_gen_white_noise_shape(shape):
    noise = utils.gen_white_noise(shape)
    if isinstance(shape, int):
        shape = (shape,)
    assert noise.shape == shape


@pytest.mark.parametrize("shape", [100, (100, 200)])
def test_gen_white_noise_mean(shape):
    noise = utils.gen_white_noise(shape, rng=np.random.default_rng(0))
    assert np.allclose(
        [noise.mean().real, noise.mean().imag], 0, rtol=0, atol=5 / np.sqrt(noise.size)
    )


@pytest.mark.parametrize("shape", [100, (100, 200)])
def test_gen_white_noise_variance(shape):
    noise = utils.gen_white_noise(shape, rng=np.random.default_rng(0))
    assert np.isclose(np.std(noise), 1, rtol=0, atol=0.1)


@pytest.mark.parametrize("baseline", [1, (0, 1), [0, 1], np.array([0, 1, 2])])
def test_get_bl_len_vec(baseline):
    assert len(utils._get_bl_len_vec(baseline)) == 3


@pytest.mark.parametrize("baseline", [1, [1 / np.sqrt(2)] * 2, [1 / np.sqrt(3)] * 3])
def test_get_bl_len_magnitude(baseline):
    assert np.isclose(utils.get_bl_len_magnitude(baseline), 1)


@pytest.mark.parametrize("ra", [np.pi / 4, 3 * np.pi / 2])
def test_compute_ha_bounds(lsts, ra):
    hour_angles = utils.compute_ha(lsts, ra)
    assert np.all((hour_angles >= -np.pi) & (hour_angles <= np.pi))


@pytest.mark.parametrize("ra", [np.pi / 2, np.pi])
def test_compute_ha_accuracy(lsts, ra):
    hour_angles = utils.compute_ha(lsts, ra)
    ref_lst = 3 * np.pi / 2
    ref_ind = np.argmin(np.abs(lsts - ref_lst))
    dlst = np.mean(np.diff(lsts))
    assert np.isclose(hour_angles[ref_ind], ref_lst - ra, atol=0.5 * dlst)


@pytest.mark.parametrize("omega_p", ["array", "interp"])
def test_Jy2T(freqs, omega_p):
    if omega_p == "array":
        omega_p = np.ones_like(freqs)
    else:
        beamfile = DATA_PATH / "HERA_H1C_BEAM_POLY.npy"
        omega_p = Beam(beamfile)

    conversion_factors = utils.jansky_to_kelvin(freqs, omega_p)
    assert conversion_factors.shape == freqs.shape


@pytest.mark.parametrize("obj", [1, (1, 2), "abc", np.array([13])])
def test_listify(obj):
    assert isinstance(utils._listify(obj), list)


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize(
    "pols",
    [
        sorted(pols)[::-1]
        for i in range(1, 5)
        for pols in combinations(range(-8, -4), i)
    ],
)
def test_reshape_vis(jit, pols):
    # Mock up some data real quick
    defaults.set("debug")
    pols = [uvutils.polnum2str(pol) for pol in pols]
    sim = Simulator(polarization_array=np.array(pols))
    data_shape = sim.data_array.shape
    sim.data.data_array = np.random.normal(size=data_shape) + 1j * np.random.normal(
        size=data_shape
    )
    # Make the autos real otherwise we're going to have problems
    for ai, aj, pol in sim.get_antpairpols():
        if ai != aj:
            continue
        inds = sim.data.antpair2ind(ai, ai)
        p = list(sim.polarization_array).index(uvutils.polstr2num(pol))
        sim.data.data_array[inds, :, p] = np.random.normal(
            size=(sim.Ntimes, sim.Nfreqs)
        ).astype(complex)

    if "xy" in sim.get_pols() and "yx" in sim.get_pols():
        # Need to fix these autos as well
        for ai in sim.antenna_numbers:
            inds = sim.data.antpair2ind(ai, ai)
            p1 = list(sim.get_pols()).index("xy")
            p2 = list(sim.get_pols()).index("yx")
            sim.data.data_array[inds, :, p1] = sim.data_array[inds, :, p2].conj()

    reshape_args = [
        sim.data.data_array,
        sim.ant_1_array,
        sim.ant_2_array,
        sim.polarization_array,
        sim.antenna_numbers,
        sim.Ntimes,
        sim.Nfreqs,
        sim.Nants,
        sim.Npols,
    ]
    vis = utils.reshape_vis(
        utils.reshape_vis(*reshape_args, invert=False, use_numba=jit),
        *reshape_args[1:],
        invert=True,
        use_numba=jit,
    )
    assert np.all(sim.data_array == vis)


def test_tanh_window_warning():
    with pytest.warns(UserWarning, match="Insufficient information"):
        window = utils.tanh_window(np.linspace(0, 1, 100))
    assert np.all(window == 1)
