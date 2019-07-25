# let's figure out some tests

import numpy as np
import os
import glob

from scipy.interpolate import RectBivariateSpline
from hera_sim.interpolators import Tsky
from hera_sim.data import DATA_PATH
from nose.tools import raises

def test_interpolator():
    dfile = os.path.join(DATA_PATH, 'HERA_Tsky_Reformatted.npz')
    tsky = Tsky(dfile)
    # make sure get_interpolator method works
    assert isinstance(tsky.get_interpolator(), RectBivariateSpline)
    # make sure the resampling method works
    freqs = np.linspace(0.125,0.175, 100)
    lsts = np.linspace(np.pi/4, 3*np.pi/4, 50)
    resampled_tsky = tsky.resample_Tsky(lsts, freqs)
    assert resampled_tsky.shape==(lsts.size, freqs.size)

@raises(AssertionError)
def test_bad_npz():
    # XXX this will need to be changed if more npz files are added to
    # XXX the data/tests directory
    bad_files = glob.glob('{}/tests/*.npz'.format(DATA_PATH))
    for bad_file in bad_files:
        tsky = Tsky(bad_file)
