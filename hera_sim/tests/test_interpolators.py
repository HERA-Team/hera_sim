# let's figure out some tests

import numpy as np
import os
import glob
import tempfile

from scipy.interpolate import RectBivariateSpline, interp1d
from hera_sim.interpolators import Tsky, freq_interp1d
from nose.tools import raises

def test_interpolator():
    # make parameters for temporary npz file
    freqs = np.linspace(0.1,0.2,100)
    lsts = np.linspace(0,np.pi,75)
    pols = ('xx',)
    tsky_arr = np.array([np.ones((lsts.size,freqs.size))])
    meta = {'pols': pols}
    
    # make a path to the temporary file and save the file there
    tmpdir = tempfile.mkdtemp()
    dfile = os.path.join(tmpdir, 'test_file.npz')
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

def test_freq_interp1d():
    # make a temporary directory
    temp_dir = tempfile.mkdtemp()

    # make some mock data
    freqs = np.linspace(0.1,0.2,100)
    values = np.arange(freqs.size)

    # get a polyfit
    polyfit = np.polyfit(freqs, values, deg=1)

    # save some files
    np.save(os.path.join(temp_dir, 'polyfit'), polyfit)
    np.savez(os.path.join(temp_dir, 'beam'), freqs=freqs, beam=values)

    # check that things work as expected for a poly1d interpolator
    interp = freq_interp1d(os.path.join(temp_dir, 'polyfit.npy'),
                           interpolator='poly1d', obj='bandpass')
    assert isinstance(interp._interpolator, np.poly1d)
    assert interp(freqs).size == freqs.size
    
    # now do the same for a interp1d interpolator
    interp = freq_interp1d(os.path.join(temp_dir, 'beam.npz'),
                           interpolator='interp1d', obj='beam')
    assert isinstance(interp._interpolator, interp1d)
    assert interp(freqs).size == freqs.size

@raises(AssertionError)
def test_bad_params():
    # make a temporary directory
    temp_dir = tempfile.mkdtemp()

    # make some mock data
    freqs = np.linspace(0.1,0.2,100)
    values = np.arange(freqs.size)

    # make some bad files
    np.savez(os.path.join(temp_dir, 'no_freqs'), beam=values)
    np.savez(os.path.join(temp_dir, 'no_values'), freqs=freqs)
    np.save(os.path.join(temp_dir, 'some_npy'), values)

    # now try to make freq_interp1d objects with bad files/parameters
    # bad object type
    interp = freq_interp1d(os.path.join(temp_dir, 'some_npy.npy'), obj='something')
    # bad interpolator
    interp = freq_interp1d(os.path.join(temp_dir, 'some_npy.npy'), obj='beam',
                           interpolator='something')
    # bad datafile extension v1
    interp = freq_interp1d(os.path.join(temp_dir, 'some_npy.npy'), obj='beam',
                           interpolator='interp1d')
    # bad datafile extension v2
    interp = freq_interp1d(os.path.join(temp_dir, 'no_freqs.npz'), obj='beam',
                           interpolator='poly1d')
    # bad keys
    interp = freq_interp1d(os.path.join(temp_dir, 'no_freqs.npz'), obj='beam',
                           interpolator='interp1d')
    interp = freq_interp1d(os.path.join(temp_dir, 'no_values.npz'), obj='beam',
                           interpolator='interp1d')
    # nonexistent file
    interp = freq_interp1d(os.path.join(temp_dir, 'not_a_file.npz'), obj='beam',
                           interpolator='interp1d')

@raises(ValueError)
def none_obj_type():
    # make a temporary directory
    temp_dir = tempfile.mkdtemp()

    # make some mock data
    data = np.random.random(10)

    # save it
    np.save(os.path.join(temp_dir, 'data'), data)

    # try to make a freq_interp1d object
    interp = freq_interp1d(os.path.join(temp_dir, 'data.npy'))


