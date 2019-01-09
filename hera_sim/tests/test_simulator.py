"""
This tests the Simulator object and associated utilities. It does *not*
check for correctness of individual models, as they should be tested
elsewhere.
"""

from hera_sim.simulate import Simulator
from hera_sim.foregrounds import diffuse_foreground
from hera_sim.noise import resample_Tsky, HERA_Tsky_mdl
import numpy as np
from os import path
import shutil
import tempfile

from nose.tools import raises

def create_sim():
    return Simulator(
        n_freq=10,
        n_times=20,
        antennas={
            0: (20.0, 20.0, 0),
            1: (50.0, 50.0, 0)
        },
        ant_pairs=[(0, 1)]
    )


def test_from_empty():
    sim = create_sim()

    assert sim.data.data_array.shape == (20, 1, 10, 1)
    assert np.all(sim.data.data_array == 0)


def test_add_with_str():
    sim = create_sim()
    sim.add_eor("noiselike_eor")
    assert not np.all(sim.data.data_array == 0)


def test_add_with_builtin():
    sim = create_sim()
    sim.add_foregrounds(diffuse_foreground, Tsky=HERA_Tsky_mdl['xx'])
    assert not np.all(sim.data.data_array == 0)


def test_add_with_custom():
    sim = create_sim()

    def custom_noise(**kwargs):
        vis = resample_Tsky(**kwargs)
        return 2*vis

    sim.add_noise(custom_noise)
    assert not np.all(sim.data.data_array == 0)


def test_io():
    sim = create_sim()

    # Create a temporary directory to write stuff to (for python 3 this is much easier)
    direc = tempfile.mkdtemp()

    sim.add_foregrounds("pntsrc_foreground")
    sim.add_gains()

    sim.write_data(path.join(direc, 'tmp_data.uvh5'))

    sim2 = Simulator(
        data_filename=path.join(direc, 'tmp_data.uvh5')
    )

    assert np.all(sim.data.data_array == sim2.data.data_array)

    # delete the tmp
    shutil.rmtree(direc)


@raises(AttributeError)
def test_wrong_func():
    sim = create_sim()

    sim.add_eor("noiselike_EOR") # wrong function name

@raises(TypeError)
def test_wrong_arguments():
    sim = create_sim()
    sim.add_foregrounds(diffuse_foreground, Tsky_mdl=HERA_Tsky_mdl['xx'])
    assert not np.all(sim.data.data_array == 0)
