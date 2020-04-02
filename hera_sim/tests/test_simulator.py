"""
This tests the Simulator object and associated utilities. It does *not*
check for correctness of individual models, as they should be tested
elsewhere.
"""

import shutil
import tempfile
import sys
from os import path

import numpy as np
from nose.tools import raises, assert_raises

from hera_sim.foregrounds import diffuse_foreground
from hera_sim.noise import thermal_noise, HERA_Tsky_mdl, resample_Tsky, jy2T, bm_poly_to_omega_p
from hera_sim.simulate import Simulator
from hera_sim.antpos import hex_array
from hera_sim.data import DATA_PATH
from pyuvdata import UVData


def create_sim(autos=False, **kwargs):
    return Simulator(
        n_freq=10,
        n_times=20,
        antennas={
            0: (20.0, 20.0, 0),
            1: (50.0, 50.0, 0)
        },
        no_autos=not autos,
        **kwargs
    )


# @raises(ValueError)
# def test_wrong_antpairs():
#     Simulator(
#         n_freq=10,
#         n_times=20,
#         antennas={
#             0: (20.0, 20.0, 0),
#             1: (50.0, 50.0, 0)
#         },
#     )
#
#
# @raises(KeyError)
# def test_bad_antpairs():
#     Simulator(
#         n_freq=10,
#         n_times=20,
#         antennas={
#             0: (20.0, 20.0, 0),
#             1: (50.0, 50.0, 0)
#         },
#         antpairs=[(2, 2)]
#     )


def test_from_empty():
    sim = create_sim()

    assert sim.data.data_array.shape == (20, 1, 10, 1)
    assert np.all(np.isclose(sim.data.data_array, 0))


def test_add_with_str():
    sim = create_sim()
    sim.add_eor("noiselike_eor")
    assert not np.all(np.isclose(sim.data.data_array, 0))


def test_add_with_builtin():
    sim = create_sim()
    sim.add_foregrounds(diffuse_foreground, Tsky_mdl=HERA_Tsky_mdl['xx'])
    assert not np.all(np.isclose(sim.data.data_array, 0))


def test_add_with_custom():
    sim = create_sim()

    def custom_noise(**kwargs):
        vis = thermal_noise(**kwargs)
        return 2 * vis

    sim.add_noise(custom_noise)
    assert not np.all(np.isclose(sim.data.data_array, 0))


def test_io():
    sim = create_sim()

    # Create a temporary directory to write stuff to (for python 3 this is much easier)
    direc = tempfile.mkdtemp()

    sim.add_foregrounds("pntsrc_foreground")
    sim.add_gains()

    sim.write_data(path.join(direc, 'tmp_data.uvh5'))

    sim2 = Simulator(
        data=path.join(direc, 'tmp_data.uvh5')
    )

    uvd = UVData()
    uvd.read_uvh5(path.join(direc, 'tmp_data.uvh5'))

    sim3 = Simulator(data=uvd)

    assert np.all(sim.data.data_array == sim2.data.data_array)
    assert np.all(sim.data.data_array == sim3.data.data_array)

    with assert_raises(ValueError):
        sim.write_data(path.join(direc, 'tmp_data.bad_extension'), file_type="bad_type")

    # delete the tmp
    shutil.rmtree(direc)


@raises(AttributeError)
def test_wrong_func():
    sim = create_sim()

    sim.add_eor("noiselike_EOR")  # wrong function name


@raises(TypeError)
def test_wrong_arguments():
    sim = create_sim()
    sim.add_foregrounds(diffuse_foreground, what=HERA_Tsky_mdl['xx'])


def test_other_components():
    sim = create_sim(autos=True)

    sim.add_xtalk('gen_whitenoise_xtalk', bls=[(0, 1, 'xx')])
    sim.add_xtalk('gen_cross_coupling_xtalk', bls=[(0, 1, 'xx')])
    sim.add_sigchain_reflections(ants=[0])

    assert not np.all(np.isclose(sim.data.data_array,  0))
    assert np.all(np.isclose(sim.data.get_data(0,0), 0))

    sim = create_sim()

    sim.add_rfi("rfi_stations")

    assert not np.all(np.isclose(sim.data.data_array,  0))


def test_not_add_vis():
    sim = create_sim()
    vis = sim.add_eor("noiselike_eor", add_vis=False)

    assert np.all(np.isclose(sim.data.data_array,  0))

    assert not np.all(np.isclose(vis, 0))

    assert "noiselike_eor" not in sim.data.history


def test_adding_vis_but_also_returning():
    sim = create_sim()
    vis = sim.add_eor("noiselike_eor", ret_vis=True)

    assert not np.all(np.isclose(vis, 0))
    np.testing.assert_array_almost_equal(vis, sim.data.data_array)

    vis = sim.add_foregrounds("diffuse_foreground", Tsky_mdl=HERA_Tsky_mdl['xx'], ret_vis=True)
    np.testing.assert_array_almost_equal(vis, sim.data.data_array, decimal=5)

def test_mult_pols():
    sim = create_sim(polarization_array=['xx','yy'])
    sim.add_foregrounds("diffuse_foreground", Tsky_mdl=HERA_Tsky_mdl['xx'], pol='xx')
    assert np.all(sim.data.get_data((0,1,'yy'))==0)
    assert not np.all(sim.data.get_data((0,1,'xx'))==0)

@raises(AssertionError)
def test_bad_pol():
    sim = create_sim(polarization_array=['xx','yy'])
    # not specifying polarization
    sim.add_foregrounds("diffuse_foreground", Tsky_mdl=HERA_Tsky_mdl['xx'])
    # specifying bad polarization
    sim.add_foregrounds("diffuse_foreground", Tsky_mdl=HERA_Tsky_mdl['xx'], pol='xy')

def test_consistent_across_reds():
    ants = hex_array(2,split_core=False,outriggers=0)
    sim = Simulator(n_freq=50, n_times=20, antennas=ants)
    sim.add_foregrounds('diffuse_foreground', Tsky_mdl=HERA_Tsky_mdl['xx'],
            seed_redundantly=True)
    sim.add_eor('noiselike_eor', seed_redundantly=True)
    reds = sim.data.get_baseline_redundancies()[0][1] # choose non-autos
    key1 = sim.data.baseline_to_antnums(reds[0]) + ('xx',)
    key2 = sim.data.baseline_to_antnums(reds[1]) + ('xx',)
    assert np.all(np.isclose(sim.data.get_data(key1),sim.data.get_data(key2)))

def test_return_and_save_seeds():
    ants = hex_array(2, split_core=False, outriggers=0)
    sim = Simulator(n_freq=50, n_times=20, antennas=ants)
    sim.add_foregrounds('diffuse_foreground', Tsky_mdl=HERA_Tsky_mdl['xx'],
            seed_redundantly=True)
    tempdir = tempfile.mkdtemp()
    vis_file = path.join(tempdir, "test.uvh5")
    seeds = sim.data.extra_keywords['seeds']
    alt_seeds = sim.write_data(vis_file, ret_seeds=True, save_seeds=True)
    saved_seeds = np.load(vis_file.replace(".uvh5", ".npy"), allow_pickle=True)
    saved_seeds = saved_seeds[None][0]
    assert seeds==alt_seeds
    assert seeds==sim.data.extra_keywords['seeds']
    assert np.all(seeds['diffuse_foreground']==saved_seeds['diffuse_foreground'])

if sys.version_info.major < 3 or \
   sys.version_info.major > 3 and sys.version_info.minor < 4:
    @raises(NotImplementedError)
    def test_run_sim():
        sim_params = {}
        sim = create_sim()
        sim.run_sim(**sim_params)
else:
    def test_run_sim():
        sim_params = {
                "diffuse_foreground": {"Tsky_mdl":HERA_Tsky_mdl['xx']},
                "pntsrc_foreground": {"nsrcs":500, "Smin":0.1},
                "noiselike_eor": {"eor_amp":3e-2},
                "thermal_noise": {"Tsky_mdl":HERA_Tsky_mdl['xx'], "inttime":8.59},
                "rfi_scatter": {"chance":0.99, "strength":5.7, "std":2.2},
                "rfi_impulse": {"chance":0.99, "strength":17.22},
                "rfi_stations": {},
                "gains": {"gain_spread":0.05},
                "sigchain_reflections": {"amp":[0.5,0.5],
                                         "dly":[14,7],
                                         "phs":[0.7723,3.2243]},
                "gen_whitenoise_xtalk": {"amplitude":1.2345} 
                }

        sim = create_sim()
    
        sim.run_sim(**sim_params)

        assert not np.all(np.isclose(sim.data.data_array, 0))

        # instantiate a mock simulation file
        tmp_sim_file = tempfile.mkstemp()[1]
        # write something to it
        with open(tmp_sim_file, 'w') as sim_file:
            sim_file.write("""
                diffuse_foreground: 
                    Tsky_mdl: !Tsky
                        datafile: {}/HERA_Tsky_Reformatted.npz
                        pol: yy
                pntsrc_foreground: 
                    nsrcs: 500
                    Smin: 0.1
                noiselike_eor: 
                    eor_amp: 0.03
                gains: 
                    gain_spread: 0.05
                gen_cross_coupling_xtalk: 
                    amp: 0.225
                    dly: 13.2
                    phs: 2.1123
                thermal_noise: 
                    Tsky_mdl: !Tsky 
                        datafile: {}/HERA_Tsky_Reformatted.npz
                        pol: xx
                    inttime: 9.72
                rfi_scatter: 
                    chance: 0.99
                    strength: 5.7
                    std: 2.2
                    """.format(DATA_PATH, DATA_PATH))
        sim = create_sim(autos=True)
        sim.run_sim(tmp_sim_file)
        assert not np.all(np.isclose(sim.data.data_array, 0))

    @raises(AssertionError)
    def test_run_sim_both_args():
        # make a temporary test file
        tmp_sim_file = tempfile.mkstemp()[1]
        with open(tmp_sim_file, 'w') as sim_file:
            sim_file.write("""
                pntsrc_foreground:
                    nsrcs: 5000
                    """)
        sim_params = {"diffuse_foreground": {"Tsky_mdl":HERA_Tsky_mdl['xx']} }
        sim = create_sim()
        sim.run_sim(tmp_sim_file, **sim_params)

    @raises(AssertionError)
    def test_run_sim_bad_param_key():
        bad_key = {"something": {"something else": "another different thing"} }
        sim = create_sim()
        sim.run_sim(**bad_key)

    @raises(AssertionError)
    def test_run_sim_bad_param_value():
        bad_value = {"diffuse_foreground": 13}
        sim = create_sim()
        sim.run_sim(**bad_value)

    @raises(SystemExit)
    def test_bad_yaml_config():
        # make a bad config file
        tmp_sim_file = tempfile.mkstemp()[1]
        with open(tmp_sim_file, 'w') as sim_file:
            sim_file.write("""
                this:
                    is: a
                     bad: file
                     """)
        sim = create_sim()
        sim.run_sim(tmp_sim_file)


def test_noise_from_autos():
    sim = Simulator(n_freq=100, n_times=10, antennas={0: (0, 0, 0), 1: (14, 0, 0)})

    freqs = sim.data.freq_array[0] / 1e9

    # Add foregrounds in a weird way
    sky_model = resample_Tsky(freqs, np.unique(sim.data.lst_array))

    # Convert to Jy.
    sky_model = sky_model / (jy2T(freqs, bm_poly_to_omega_p(freqs))/1000)

    # Add to autos.
    sim.data.data_array[sim.data.antpair2ind(0, 0), 0, :, 0] += sky_model
    sim.data.data_array[sim.data.antpair2ind(1, 1), 0, :, 0] += sky_model

    # Produce noise based on autos
    np.random.seed(1010)
    auto_vis = sim.add_noise('thermal_noise', use_autos=True, ret_vis=True, add_vis=False)

    # Produce noise based on default sky model.
    np.random.seed(1010)
    default_vis = sim.add_noise('thermal_noise', ret_vis=True, add_vis=False)

    assert np.allclose(auto_vis, default_vis)
