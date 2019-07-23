"""
This tests the Simulator object and associated utilities. It does *not*
check for correctness of individual models, as they should be tested
elsewhere.
"""

import shutil
import tempfile
from os import path

import numpy as np
from nose.tools import raises, assert_raises

from hera_sim.foregrounds import diffuse_foreground
from hera_sim.noise import thermal_noise, HERA_Tsky_mdl
from hera_sim.simulate import Simulator


def create_sim(autos=False):
    return Simulator(
        n_freq=10,
        n_times=20,
        antennas={
            0: (20.0, 20.0, 0),
            1: (50.0, 50.0, 0)
        },
        no_autos=not autos
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

    print(sim.data.antenna_names)
    sim.write_data(path.join(direc, 'tmp_data.uvh5'))

    sim2 = Simulator(
        data=path.join(direc, 'tmp_data.uvh5')
    )

    assert np.all(sim.data.data_array == sim2.data.data_array)

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

    assert np.all(np.isclose(sim.data.data_array,  0))

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

def test_run_sim():
    fgs_params = {
            "diffuse_foregrounds": {"Tsky_mdl":HERA_Tsky_mdl['xx']},
            "pntsrc_foregrounds": {"nsrcs":500, "Smin":0.1}
            }
    
    eor_params = {"noiselike_eor": {"eor_amp":3e-2}}

    noise_params = { "thermal_noise": {"Tsky_mdl":HERA_Tsky_mdl['xx'], "inttime":8.59} }

    rfi_params = { "rfi_scatter": {"chance":0.001, "strength":5.7, "std":2.2},
                   "rfi_dtv" : {"freq_min":0.172, "width":0.012, "chance":0.002},
                   "rfi_impulse": {"strength":17.22},
                   "rfi_stations": {} }

    sigchain_params = { "gains": {"gain_spread":0.05},
                        "sigchain_reflections":{} }

    xtalk_params = { "gen_whitenoise_xtalk": {"amplitude":1.2345} }

    param_list = [fgs_params, eor_params, noise_params, rfi_params, \
                  sigchain_params, xtalk_params]

    def check_update(sim, params):
        data = sim.data.data_array
        sim.run_sim(**params)
        assert not np.all(np.isclose(data, sim.data.data_array))

    sim = create_sim()

    for params in param_list:
        check_update(sim, params)

    sim_file = "test_sim_file.yaml"
    sim = create_sim()
    sim.run_sim(sim_file)
    assert not np.all(np.isclose(sim.data.data_array, 0))

@raises(AssertionError)
def test_run_sim_both_args():
    sim_file = "test_sim_file.yaml"
    sim_params = {"diffuse_foregrounds": {"Tsky_mdl":HERA_Tsky_mdl['xx']} }
    sim = create_sim()
    sim.run_sim(sim_file, **sim_params)

@raises(AssertionError)
def test_run_sim_bad_param_key():
    bad_key = {"something": {"something else": "another different thing"} }
    sim = create_sim()
    sim.run_sim(**bad_key)

@raises(AssertionError)
def test_run_sim_bad_param_value():
    bad_value = {"diffuse_foregrounds": 13}
    sim = create_sim
    sim.run_sim(**bad_value)
