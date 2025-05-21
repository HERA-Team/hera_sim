"""
This tests the Simulator object and associated utilities. It does *not*
check for correctness of individual models, as they should be tested
elsewhere.
"""

import copy
import itertools
import os
import tempfile

import numpy as np
import pytest
import yaml
from deprecation import fail_if_not_removed
from pyuvdata import UVData

from hera_sim import CONFIG_PATH, DATA_PATH, Simulator, component
from hera_sim.antpos import hex_array
from hera_sim.defaults import defaults
from hera_sim.foregrounds import DiffuseForeground, diffuse_foreground
from hera_sim.interpolators import Beam
from hera_sim.noise import HERA_Tsky_mdl

beamfile = os.path.join(DATA_PATH, "HERA_H1C_BEAM_POLY.npy")
omega_p = Beam(beamfile)
Tsky_mdl = HERA_Tsky_mdl["xx"]

Nfreqs = 10
Ntimes = 20

base_config = dict(
    Nfreqs=Nfreqs,
    start_freq=1e8,
    channel_width=1e8 / 1024,
    Ntimes=Ntimes,
    start_time=2458115.9,
    integration_time=10.7,
    array_layout={0: (20.0, 20.0, 0), 1: (50.0, 50.0, 0)},
    no_autos=True,
)


def create_sim(autos=False, **kwargs):
    config = base_config.copy()
    config["no_autos"] = not autos
    config.update(kwargs)
    return Simulator(**config)


@pytest.fixture(scope="function")
def base_sim():
    return create_sim()


@pytest.fixture(scope="function")
def ref_sim(base_sim):
    base_sim.add("noiselike_eor", seed="redundant")
    return base_sim


def test_from_empty(base_sim):
    assert all(
        [
            base_sim.data.Ntimes == Ntimes,
            base_sim.data.Nfreqs == Nfreqs,
            np.all(base_sim.data.data_array == 0),
            base_sim.freqs.size == Nfreqs,
            base_sim.freqs.ndim == 1,
            base_sim.lsts.size == Ntimes,
            base_sim.lsts.ndim == 1,
        ]
    )


def test_initialize_from_defaults():
    with open(CONFIG_PATH / "H1C.yaml") as config:
        defaults = yaml.load(config.read(), Loader=yaml.FullLoader)
    setup = defaults["setup"]
    freq_params = setup["frequency_array"]
    time_params = setup["time_array"]
    array = defaults["telescope"]["array_layout"]
    sim = Simulator(defaults_config="h1c")
    assert all(
        [
            sim.freqs.size == freq_params["Nfreqs"],
            sim.freqs[0] == freq_params["start_freq"] / 1e9,
            sim.lsts.size == time_params["Ntimes"],
            sim.times[0] == time_params["start_time"],
            len(sim.antpos) == len(array),
        ]
    )


def test_phase_wrapped_lsts():
    sim = create_sim(start_time=2458120.15, Ntimes=100, integration_time=10.7)
    assert sim.lsts[0] > sim.lsts[-1]


def test_nondefault_blt_order_lsts():
    array_layout = hex_array(2, split_core=False, outriggers=0)
    sim = create_sim(
        Ntimes=100,
        integration_time=10.7,
        start_time=2458120.15,
        array_layout=array_layout,
    )
    sim.data.reorder_blts("baseline", minor_order="time")
    iswrapped = sim.lsts < sim.lsts[0]
    lsts = sim.lsts + np.where(iswrapped, 2 * np.pi, 0)
    assert np.all(lsts[1:] > lsts[:-1])


def test_add_with_str(base_sim):
    base_sim.add("noiselike_eor", rng=np.random.default_rng(0))
    assert not np.all(base_sim.data.data_array == 0)


def test_add_with_builtin_class(base_sim):
    base_sim.add(
        DiffuseForeground,
        Tsky_mdl=Tsky_mdl,
        omega_p=omega_p,
        rng=np.random.default_rng(0),
    )
    assert not np.all(np.isclose(base_sim.data.data_array, 0))


def test_add_with_class_instance(base_sim):
    base_sim.add(
        diffuse_foreground,
        Tsky_mdl=Tsky_mdl,
        omega_p=omega_p,
        rng=np.random.default_rng(0),
    )
    assert not np.all(np.isclose(base_sim.data.data_array, 0))


@pytest.mark.parametrize("multiplicative", [False, None])
def test_add_with_custom_class(base_sim, multiplicative):
    @component
    class TestBase:
        pass

    class Test(TestBase):
        is_multiplicative = multiplicative

        def __init__(self):
            pass

        def __call__(self, lsts, freqs):
            return np.ones((lsts.size, freqs.size), dtype=complex)

    if multiplicative is None:
        with pytest.warns(UserWarning, match="have not specified"):
            base_sim.add(Test)
    else:
        base_sim.add(Test)
        assert np.all(base_sim.data.data_array == 1)


def test_add_with_full_array_return(base_sim):
    @component
    class TestBase:
        pass

    class Test(TestBase):
        return_type = "full_array"
        attrs_to_pull = dict(pols="polarization_array")

        def __init__(self):
            pass

        def __call__(self, freqs, ant_1_array, pols):
            data_shape = (ant_1_array.size, freqs.size, pols.size)
            return np.ones(data_shape, dtype=complex)

    base_sim.add(Test)
    assert np.all(base_sim.data_array == 1)


def test_refresh(base_sim):
    base_sim.add("noiselike_eor")
    base_sim.refresh()

    assert np.all(base_sim.data.data_array == 0)


@pytest.mark.parametrize("make_sim_from", ["uvdata", "file"])
def test_io(base_sim, make_sim_from, tmp_path):
    # Simulate some data and write it to disk.
    filename = tmp_path / "tmp_data.uvh5"
    base_sim.add("pntsrc_foreground")
    base_sim.add("gains")
    base_sim.write(filename)

    if make_sim_from == "file":
        sim2 = Simulator(data=filename)
    elif make_sim_from == "uvdata":
        uvd = UVData()
        uvd.read_uvh5(filename)
        sim2 = Simulator(data=uvd)

    # Make sure that the data agree to numerical precision.
    assert np.allclose(
        base_sim.data.data_array, sim2.data.data_array, rtol=0, atol=1e-7
    )


def test_io_bad_format(base_sim, tmp_path):
    with pytest.raises(ValueError) as err:
        base_sim.write(tmp_path / "data.bad_extension", save_format="bad_type")
    assert "must correspond to a write method" in err.value.args[0]


@pytest.mark.parametrize("pol", [None, "xx"])
def test_get_full_data(ref_sim, pol):
    data = ref_sim.get("noiselike_eor", pol)
    if pol is None:
        assert np.allclose(data, ref_sim.data.data_array, rtol=0, atol=1e-7)
    else:
        assert np.allclose(data, ref_sim.data.data_array[..., 0], rtol=0, atol=1e-7)


@pytest.mark.parametrize("pol", [None, "xx"])
@pytest.mark.parametrize("conj", [True, False])
def test_get_with_one_seed(base_sim, pol, conj):
    # Set the seed mode to "once" even if that's not realistic.
    base_sim.add("noiselike_eor", seed="once")
    ant1, ant2 = (0, 1) if conj else (1, 0)
    key = (ant1, ant2, pol)
    data = base_sim.get("noiselike_eor", key)
    antpairpol = (ant1, ant2) if pol is None else (ant1, ant2, pol)
    true_data = base_sim.data.get_data(antpairpol)
    if pol:
        assert np.allclose(data, true_data, rtol=0, atol=1e-7)
    else:
        assert np.allclose(data[..., 0], true_data, rtol=0, atol=1e-7)


# TODO: this will need to be updated when full polarization support is added
@pytest.mark.parametrize("pol", [None, "xx"])
@pytest.mark.parametrize("conj", [True, False])
def test_get_with_initial_seed(pol, conj):
    # Simulate an effect where we would actually use this setting.
    sim = create_sim(autos=True)
    sim.add("thermal_noise", seed="initial")
    ant1, ant2 = (0, 1) if conj else (1, 0)
    vis = sim.get("thermal_noise", key=(ant1, ant2, pol))
    if pol:
        assert np.allclose(sim.data.get_data(ant1, ant2, pol), vis)
    else:
        assert np.allclose(sim.data.get_data(ant1, ant2), vis[..., 0])


def test_get_nonexistent_component(ref_sim):
    with pytest.raises(ValueError) as err:
        ref_sim.get("diffuse_foreground")
    assert "has not yet been simulated" in err.value.args[0]


def test_get_vis_only_one_antenna(ref_sim):
    with pytest.raises(ValueError) as err:
        ref_sim.get("noiselike_eor", 1)
    assert "a pair of antennas must be provided" in err.value.args[0]


@pytest.mark.parametrize("conj", [True, False])
@pytest.mark.parametrize("pol", [None, "xx"])
def test_get_redundant_data(pol, conj):
    antpos = {0: [0, 0, 0], 1: [10, 0, 0], 2: [0, 10, 0], 3: [10, 10, 0]}
    sim = create_sim(array_layout=antpos)
    defaults.set("h1c")
    sim.add("diffuse_foreground", seed="redundant")
    ant1, ant2 = (0, 1) if conj else (1, 0)
    ai, aj = (2, 3) if conj else (3, 2)
    vis = sim.get("diffuse_foreground", (ant1, ant2, pol))
    if pol:
        assert np.allclose(sim.data.get_data(ai, aj, pol), vis)
    else:
        assert np.allclose(sim.data.get_data(ai, aj), vis[..., 0])
    defaults.deactivate()


@pytest.mark.parametrize("pol", [None, "x"])
@pytest.mark.parametrize("ant1", [None, 1])
def test_get_multiplicative_effect(base_sim, pol, ant1):
    gains = base_sim.add("gains", seed="once", ret_vis=True)
    _gains = base_sim.get("gains", key=(ant1, pol))
    if pol is not None and ant1 is not None:
        assert np.allclose(gains[(ant1, pol)], _gains)
    elif pol is None and ant1 is not None:
        assert all(
            np.allclose(gains[(ant1, _pol)], _gains[(ant1, _pol)])
            for _pol in base_sim.data.get_feedpols()
        )
    elif pol is not None and ant1 is None:
        assert all(
            np.allclose(gains[(ant, pol)], _gains[(ant, pol)])
            for ant in base_sim.antpos
        )
    else:
        assert all(np.allclose(gains[antpol], _gains[antpol]) for antpol in gains)


def test_not_add_vis(base_sim):
    vis = base_sim.add(
        "noiselike_eor", add_vis=False, ret_vis=True, rng=np.random.default_rng(0)
    )

    assert np.all(base_sim.data.data_array == 0)

    assert not np.all(vis == 0)

    assert "noiselike_eor" not in base_sim.data.history
    assert "noiselike_eor" not in base_sim._components.keys()

    # make sure None is returned if neither adding nor returning
    assert base_sim.add("noiselike_eor", add_vis=False, ret_vis=False) is None


def test_adding_vis_but_also_returning(base_sim):
    vis = base_sim.add("noiselike_eor", ret_vis=True, rng=np.random.default_rng(0))

    assert not np.all(vis == 0)
    assert np.all(np.isclose(vis, base_sim.data.data_array))

    # use season defaults for simplicity
    defaults.set("h1c")
    vis += base_sim.add(
        "diffuse_foreground", ret_vis=True, rng=np.random.default_rng(90)
    )
    # deactivate defaults for good measure
    defaults.deactivate()
    assert np.all(np.isclose(vis, base_sim.data.data_array))


def test_filter():
    sim = create_sim(autos=True)

    # only add visibilities for the (0,1) baseline
    vis_filter = (0, 1, "xx")

    sim.add("noiselike_eor", vis_filter=vis_filter, rng=np.random.default_rng(10))
    assert np.all(sim.data.get_data(0, 0) == 0)
    assert np.all(sim.data.get_data(1, 1) == 0)
    assert np.all(sim.data.get_data(0, 1) != 0)
    assert np.all(sim.data.get_data(1, 0) != 0)
    assert np.all(sim.data.get_data(0, 1) == sim.data.get_data(1, 0).conj())


def test_consistent_across_reds():
    # initialize a sim with some redundant baselines
    # this is a 7-element hex array
    ants = hex_array(2, split_core=False, outriggers=0)
    sim = Simulator(
        Nfreqs=20,
        start_freq=1e8,
        channel_width=5e6,
        Ntimes=20,
        start_time=2458115.9,
        integration_time=10.7,
        array_layout=ants,
    )

    # activate season defaults for simplicity
    defaults.set("h1c")

    # add something that should be the same across a redundant group
    sim.add("diffuse_foreground", seed="redundant")

    # deactivate defaults for good measure
    defaults.deactivate()

    reds = sim.red_grps[1]  # choose non-autos
    # check that every pair in the redundant group agrees
    for i, _bl1 in enumerate(reds):
        for bl2 in reds[i + 1 :]:
            # get the antenna pairs from the baseline integers
            bl1 = sim.data.baseline_to_antnums(_bl1)
            bl2 = sim.data.baseline_to_antnums(bl2)
            vis1 = sim.data.get_data(bl1)
            vis2 = sim.data.get_data(bl2)
            assert np.all(np.isclose(vis1, vis2))

    # Check that seeds vary between redundant groups
    seeds = list(list(sim._seeds.values())[0].values())
    assert all(
        seed_pair[0] != seed_pair[1] for seed_pair in itertools.combinations(seeds, 2)
    )


def test_run_sim():
    # activate season defaults for simplicity
    defaults.set("h1c", refresh=True)

    sim_params = {
        "diffuse_foreground": {"Tsky_mdl": HERA_Tsky_mdl["xx"]},
        "pntsrc_foreground": {"nsrcs": 500, "Smin": 0.1},
        "noiselike_eor": {"eor_amp": 3e-2},
        "thermal_noise": {"Tsky_mdl": HERA_Tsky_mdl["xx"], "integration_time": 8.59},
        "rfi_scatter": {
            "scatter_chance": 0.99,
            "scatter_strength": 5.7,
            "scatter_std": 2.2,
        },
        "rfi_impulse": {"impulse_chance": 0.99, "impulse_strength": 17.22},
        "rfi_stations": {},
        "gains": {"gain_spread": 0.05},
        "sigchain_reflections": {
            "amp": [0.5, 0.5],
            "dly": [14, 7],
            "phs": [0.7723, 3.2243],
        },
        "whitenoise_xtalk": {"amplitude": 1.2345},
    }

    sim = create_sim(autos=True)

    sim.run_sim(**sim_params)

    assert not np.all(np.isclose(sim.data.data_array, 0))

    # instantiate a mock simulation file
    tmp_sim_file = tempfile.mkstemp()[1]
    # write something to it
    with open(tmp_sim_file, "w") as sim_file:
        sim_file.write(
            f"""
            diffuse_foreground:
                Tsky_mdl: !Tsky
                    datafile: {DATA_PATH}/HERA_Tsky_Reformatted.npz
                    pol: yy
            pntsrc_foreground:
                nsrcs: 500
                Smin: 0.1
            noiselike_eor:
                eor_amp: 0.03
            gains:
                gain_spread: 0.05
            cross_coupling_xtalk:
                amp: 0.225
                dly: 13.2
                phs: 2.1123
            thermal_noise:
                Tsky_mdl: !Tsky
                    datafile: {DATA_PATH}/HERA_Tsky_Reformatted.npz
                    pol: xx
                integration_time: 9.72
            rfi_scatter:
                scatter_chance: 0.99
                scatter_strength: 5.7
                scatter_std: 2.2
                """
        )
    sim = create_sim(autos=True)
    sim.run_sim(tmp_sim_file)
    assert not np.all(np.isclose(sim.data.data_array, 0))

    # deactivate season defaults for good measure
    defaults.deactivate()


def test_run_sim_both_args(base_sim, tmp_path):
    # make a temporary test file
    tmp_sim_file = tmp_path / "temp_config.yaml"
    with open(tmp_sim_file, "w") as sim_file:
        sim_file.write(
            """
            pntsrc_foreground:
                nsrcs: 5000
                """
        )
    sim_params = {"diffuse_foreground": {"Tsky_mdl": HERA_Tsky_mdl["xx"]}}
    with pytest.raises(ValueError) as err:
        base_sim.run_sim(tmp_sim_file, **sim_params)
    assert "Please only pass one of the two." in err.value.args[0]


@pytest.mark.parametrize("select_param", ["freq", "time", "ants", "pols"])
def test_params_ok_after_select(select_param):
    array_layout = {0: [0, 0, 0], 1: [10, 0, 0], 2: [0, 10, 0]}
    polarizations = np.array(["xx", "yy", "xy", "yx"])
    sim = create_sim(autos=True, array_layout=array_layout, polarizations=polarizations)
    if select_param == "freq":
        select_freqs = sim.freqs[:5]
        sim.data.select(freq_chans=np.arange(select_freqs.size))
        assert np.all(select_freqs == sim.freqs)
    elif select_param == "time":
        select_times = sim.times[:5]
        sim.data.select(times=select_times)
        assert np.all(select_times == sim.times)
    elif select_param == "ants":
        sim.data.select(antenna_nums=np.arange(2))
        assert (2 not in set(sim.ant_1_array).union(sim.ant_2_array)) and (
            2 not in sim.antpos
        )
    else:
        select_pols = sim.polarization_array[:2]
        sim.data.select(polarizations=select_pols)
        assert np.all(select_pols == sim.polarization_array)


def test_bad_yaml_config(base_sim, tmp_path):
    # make a bad config file
    tmp_sim_file = tmp_path / "bad_config.yaml"
    with open(tmp_sim_file, "w") as sim_file:
        sim_file.write(
            """
            this:
                is: a
                 bad: file
                 """
        )
    with pytest.raises(IOError) as err:
        base_sim.run_sim(tmp_sim_file)
    assert err.value.args[0] == "The configuration file was not able to be loaded."


def test_run_sim_bad_param_key(base_sim):
    bad_key = {"something": {"something else": "another different thing"}}
    with pytest.raises(ValueError) as err:
        base_sim.run_sim(**bad_key)
        assert "The component 'something' does not exist" in err.value.args[0]


def test_run_sim_bad_param_value(base_sim):
    bad_value = {"diffuse_foreground": 13}
    with pytest.raises(TypeError) as err:
        base_sim.run_sim(**bad_value)
    assert "The parameters for diffuse_foreground are not" in err.value.args[0]


@fail_if_not_removed
def test_add_eor(base_sim):
    base_sim.add_eor("noiselike_eor")


@fail_if_not_removed
def test_add_fg(base_sim):
    base_sim.add_foregrounds("pntsrc_foreground")


@fail_if_not_removed
def test_add_rfi(base_sim):
    base_sim.add_rfi("dtv")


def test_plot_array(base_sim):
    fig = base_sim.plot_array()
    ax = fig.axes[0]
    assert ax.get_xlabel() == "East Position [m]"
    assert ax.get_ylabel() == "North Position [m]"
    assert ax.get_title() == "Array Layout"


# Testing against using reference files is already done in test_io,
# so we only test for specifying integrations per file.
def test_chunker(base_sim, tmp_path):
    Nint_per_file = 5
    prefix = "zen"
    sky_cmp = "eor"
    state = "test"
    filetype = "uvh5"
    Nfiles = base_sim.Ntimes // Nint_per_file
    base_sim.add("noiselike_eor", seed="redundant")
    base_sim.chunk_sim_and_save(
        tmp_path,
        Nint_per_file=Nint_per_file,
        prefix=prefix,
        sky_cmp=sky_cmp,
        state=state,
        filetype=filetype,
    )
    assert (
        len(list(tmp_path.glob(f"{prefix}.*.{sky_cmp}.{state}.{filetype}"))) == Nfiles
    )


@pytest.mark.parametrize(
    "component",
    ["eor", "foregrounds", "noise", "rfi", "gains", "sigchain_reflections", "xtalk"],
)
def test_legacy_funcs(component):
    args = []
    sim = create_sim(autos=True)
    if component in ["eor", "foregrounds", "noise", "rfi", "xtalk"]:
        model = {
            "eor": "noiselike_eor",
            "foregrounds": "pntsrc_foreground",
            "noise": "thermal_noise",
            "rfi": "rfi_dtv",
            "xtalk": "cross_coupling_xtalk",
        }[component]
        args.append(model)
    if component == "sigchain_reflections":
        args.append([1])
    elif component == "xtalk":
        args.append([(0, 1)])
    getattr(sim, f"add_{component}")(*args)


def test_vis_filter_single_pol():
    sim = create_sim(polarization_array=["xx", "yy"])
    sim.add("noiselike_eor", vis_filter=["xx"], rng=np.random.default_rng(99))
    assert np.all(sim.get_data("xx")) and not np.any(sim.get_data("yy"))


def test_vis_filter_two_pol():
    sim = create_sim(polarization_array=["xx", "xy", "yx", "yy"])
    sim.add("noiselike_eor", vis_filter=["xx", "yy"], rng=np.random.default_rng(5))
    assert all(
        [
            np.all(sim.get_data("xx")),
            np.all(sim.get_data("yy")),
            not np.any(sim.get_data("xy")),
            not np.any(sim.get_data("yx")),
        ]
    )


def test_vis_filter_arbitrary_key():
    sim = create_sim(
        array_layout=hex_array(2, split_core=False, outriggers=0),
        polarization_array=["xx", "yy"],
    )
    sim.add("noiselike_eor", vis_filter=[1, 3, 5, "xx"], rng=np.random.default_rng(7))
    bls = sim.data.get_antpairs()
    assert not np.any(sim.get_data("yy"))
    assert all(
        np.all(sim.get_data((ai, aj, "xx")))
        for ai, aj in bls
        if ai in (1, 3, 5) or aj in (1, 3, 5)
    )


def test_bad_initialization_data():
    with pytest.raises(TypeError) as err:
        Simulator(data=123)
    assert "data type not understood." in err.value.args[0]


def test_integer_seed(base_sim):
    seed = 2**18
    d1 = base_sim.add("noiselike_eor", add_vis=False, ret_vis=True, seed=seed)
    d2 = base_sim.add("noiselike_eor", add_vis=False, ret_vis=True, seed=seed)
    assert np.allclose(d1, d2)


def test_none_seed(base_sim):
    d1 = base_sim.add("noiselike_eor", add_vis=False, ret_vis=True, seed=None)
    d2 = base_sim.add("noiselike_eor", add_vis=False, ret_vis=True, seed=None)
    assert not np.allclose(d1, d2)


def test_none_seed_state_recovery(base_sim):
    with pytest.warns(UserWarning, match="seed the random state"):
        base_sim.add("noiselike_eor", seed=None)
        vis = base_sim.get("noiselike_eor")
    assert not np.allclose(base_sim.data.data_array, vis)


@pytest.mark.parametrize("seed", [3.14, "redundant", "unsupported"])
def test_bad_seeds(base_sim, seed):
    err = TypeError if seed == 3.14 else ValueError
    match = {
        3.14: "seeding mode must be",
        "redundant": "baseline must be specified",
        "unsupported": "Seeding mode not supported.",
    }[seed]
    with pytest.raises(err, match=match):
        base_sim._seed_rng(seed, None, model_key="test")


def test_get_component_with_function():
    def func():
        pass

    with pytest.raises(TypeError, match="The input type for the component"):
        Simulator._get_component(func)


def test_get_component_bad_type():
    with pytest.raises(TypeError, match="Available component models are:"):
        Simulator._get_component(3)


def test_parse_key_with_baseline_number(base_sim):
    bl = base_sim.data.antnums_to_baseline(0, 1)
    assert base_sim._parse_key(bl) == (0, 1, None)


def test_cached_filters():
    defaults.set("debug")
    sim1 = create_sim(Ntimes=1000, Nfreqs=100)
    sim2 = copy.deepcopy(sim1)
    fringe_filter_kwargs = dict(fringe_filter_type="gauss", fr_width=0.005)
    delay_filter_kwargs = dict(delay_filter_type="gauss", standoff=15)
    kwargs = dict(
        delay_filter_kwargs=delay_filter_kwargs,
        fringe_filter_kwargs=fringe_filter_kwargs,
    )
    seed = 1420
    sim2.calculate_filters(**kwargs)
    sim1.add("diffuse_foreground", seed=seed, **kwargs)
    sim2.add("diffuse_foreground", seed=seed)
    defaults.deactivate()
    assert np.allclose(sim1.data.data_array, sim2.data.data_array)


def test_get_model_name():
    assert Simulator._get_model_name("noiselike_eor") == "noiselike_eor"
    assert Simulator._get_model_name("NOISELIKE_EOR") == "noiselike_eor"

    assert Simulator._get_model_name(DiffuseForeground) == "diffuseforeground"
    assert Simulator._get_model_name(diffuse_foreground) == "diffuseforeground"

    with pytest.raises(
        TypeError, match="You are trying to simulate an effect using a custom function"
    ):
        Simulator._get_model_name(lambda x: x)

    with pytest.raises(
        TypeError, match="You are trying to simulate an effect using a custom function"
    ):
        Simulator._get_model_name(3)


def test_parse_key(base_sim: Simulator):
    assert base_sim._parse_key(None) == (None, None, None)
    assert base_sim._parse_key(1) == (1, None, None)
    assert base_sim._parse_key(
        base_sim.data.baseline_array[-1]
    ) == base_sim.data.baseline_to_antnums(base_sim.data.baseline_array[-1]) + (None,)

    with pytest.raises(NotImplementedError, match="Functionality not yet supported"):
        base_sim._parse_key("auto")

    assert base_sim._parse_key("ee") == (None, None, "ee")

    for badkey in [3.14, [1, 2, 3], (1,)]:
        print(badkey)
        with pytest.raises(
            ValueError,
            match="Key must be an integer, string, antenna pair, or antenna pair with",
        ):
            base_sim._parse_key(badkey)

    with pytest.raises(ValueError, match="Invalid polarization string"):
        base_sim._parse_key("bad_pol")

    assert base_sim._parse_key((1, 2)) == (1, 2, None)
    assert base_sim._parse_key((1, "Jee")) == (1, None, "Jee")
    assert base_sim._parse_key((1, 2, "ee")) == (1, 2, "ee")
