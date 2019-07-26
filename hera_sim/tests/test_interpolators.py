# let's figure out some tests

import numpy as np
import os
import glob
import tempfile
import shutil

from scipy.interpolate import RectBivariateSpline
from hera_sim.interpolators import Tsky
from hera_sim.data import DATA_PATH
from nose.tools import raises

def test_interpolator():
    # make parameters for temporary npz file
    freqs = np.linspace(0.1,0.2,100)
    lsts = np.linspace(0,np.pi,75)
    pols = ('xx',)
    tsky_arr = np.array([np.ones((lsts.size,freqs.size))])
    meta = {'pols': pols}
    
    # make a path to the temporary file and save the file there
    dfile = os.path.join(DATA_PATH, 'test_file.npz')
    np.savez(dfile, tsky=tsky_arr, lsts=lsts, freqs=freqs, meta=meta)
    
    # instantiate a Tsky object
    tsky = Tsky(dfile)

    # make sure get_interpolator method works
    assert isinstance(tsky._interpolator, RectBivariateSpline)
    
    # make new frequencies and lsts for testing resampling
    freqs = np.linspace(0.125,0.175, 100)
    lsts = np.linspace(np.pi/4, 3*np.pi/4, 50)

    # check that calls to a Tsky object act like calls to an interpolator
    resampled_tsky = tsky(lsts, freqs)
    assert resampled_tsky.shape==(lsts.size, freqs.size)

    # delete the file
    os.remove(dfile)

@raises(AssertionError)
def test_bad_npz():
    # make a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # make arrays for npz files
    lsts = np.linspace(0,np.pi,10)
    freqs = np.linspace(0.1,0.2,50)
    ok_tsky = np.array([np.ones((lsts.size,freqs.size))])
    bad_tsky = np.array([np.arange(15).reshape(3,5)])
    
    # make metadata dict, just tracking polarizations here
    pols = ('xx',)
    meta = {'pols':pols}

    # start saving files
    np.savez(os.path.join(temp_dir, 'no_freqs'), lsts=lsts, tsky=ok_tsky, meta=meta)
    np.savez(os.path.join(temp_dir, 'no_lsts'), freqs=freqs, tsky=ok_tsky, meta=meta)
    np.savez(os.path.join(temp_dir, 'no_meta'), freqs=freqs, lsts=lsts, tsky=ok_tsky)
    np.savez(os.path.join(temp_dir, 'bad_key'), freqs=freqs, lsts=lsts, Tsky=ok_tsky, meta=meta)
    np.savez(os.path.join(temp_dir, 'bad_shape'), lsts=lsts, freqs=freqs, tsky=bad_tsky, meta=meta)

    # now make sure they all raise assertion errors
    for bad_file in os.listdir(temp_dir):
        tsky = Tsky(os.path.join(temp_dir,bad_file))
    
    # now check that it catches using a bad polarization
    np.savez(temp_dir+'/ok_file', lsts=lsts, freqs=freqs, tsky=ok_tsky, meta=meta)
    tsky = Tsky(temp_dir+'/ok_file.npz', pol='yy')
    
    # now delete the temporary directory
    shutil.rmtree(temp_dir)
