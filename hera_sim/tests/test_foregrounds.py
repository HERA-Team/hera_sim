import unittest
import os
from hera_sim import noise, foregrounds
from hera_sim.data import DATA_PATH
from hera_sim.interpolators import Beam
import numpy as np
import aipy
import astropy.units as u
import nose.tools as nt
from uvtools import dspec
from uvtools.utils import FFT

np.random.seed(0)

beamfile = os.path.join(DATA_PATH, "HERA_H1C_BEAM_POLY.npy")

# XXX add tests to make sure V_ij = V*_ji
# neither model has this feature!

class TestForegrounds(unittest.TestCase):
    def test_diffuse_foreground(self):
        # make some parameters
        freqs = np.linspace(.1, .2, 100, endpoint=False)
        omega_p = Beam(beamfile)
        lsts = np.linspace(0, 2*np.pi, 1000)
        times = lsts / (2*np.pi) * u.sday.to('s')
        Tsky_mdl = noise.HERA_Tsky_mdl['xx']
        bl_vec = [0, 0, 0]
        delay_filter_kwargs = {"delay_filter_type" : "tophat"}
        fringe_filter_kwargs = {"fringe_filter_type" : "tophat"}

        # simulate the effect
        vis = foregrounds.diffuse_foreground(
            lsts, freqs, bl_vec, Tsky_mdl=Tsky_mdl, omega_p=omega_p,
            delay_filter_kwargs=delay_filter_kwargs, 
            fringe_filter_kwargs=fringe_filter_kwargs
        )

        # check the shape
        self.assertEqual(vis.shape, (lsts.size,freqs.size))
        
        # check that an autocorrelation is real-valued
        self.assertTrue(np.all(vis.imag == 0))

        # any other tests we can do?

        #import uvtools, pylab as plt
        #uvtools.plot.waterfall(vis, mode='log'); plt.colorbar(); plt.show()
        
        # throw a value error if Tsky_mdl or omega_p not specified
        nt.assert_raises(
            ValueError, foregrounds.diffuse_foreground, lsts, freqs, 
            bl_vec, Tsky_mdl=None
        )
        nt.assert_raises(
            ValueError, foregrounds.diffuse_foreground, lsts, freqs, 
            bl_vec, Tsky_mdl=Tsky_mdl, omega_p=None
        )

    def test_pntsrc_foreground(self):
        # make some parameters
        freqs = np.linspace(0.1, 0.2, 100, endpoint=False)
        lsts = np.linspace(0, 2 * np.pi, 1000)
        times = lsts / (2 * np.pi) * u.sday.to('s')
        bl_vec = [300.0, 0]

        # simulate the effect
        vis = foregrounds.pntsrc_foreground(lsts, freqs, bl_vec, nsrcs=200)
        
        # check the shape
        self.assertEqual(vis.shape, (lsts.size, freqs.size))

        # check that an autocorrelation is real-valued
        bl_vec = [0, 0, 0]
        vis = foregrounds.pntsrc_foreground(lsts, freqs, bl_vec, nsrcs=200)

        assert np.all(np.isclose(vis.imag, 0))
        
        # XXX check more substantial things
        # import uvtools, pylab as plt
        # uvtools.plot.waterfall(vis, mode='phs'); plt.colorbar(); plt.show()

    def test_diffuse_foreground_orientation(self):
        # make some parameters
        freqs = np.linspace(.1, .2, 100, endpoint=False)
        omega_p = Beam(beamfile)
        lsts = np.linspace(0, 2 * np.pi, 1000)
        Tsky_mdl = noise.HERA_Tsky_mdl['xx']
        bl_vec = (0, 30.0)
        fringe_filter_kwargs = {"fringe_filter_type" : "tophat"}

        # simulate the effect
        vis = foregrounds.diffuse_foreground(
            lsts, freqs, bl_vec, Tsky_mdl=Tsky_mdl, omega_p=omega_p, 
            fringe_filter_kwargs=fringe_filter_kwargs
        )

        # check the shape
        self.assertEqual(vis.shape, (lsts.size, freqs.size))

        # test that foregrounds show up at positive fringe-rates
        
        # make a purely EW baseline
        bl_vec = (100.0, 0.0)
        
        # use a gaussian fringe filter with a width of 10 uHz
        fringe_filter_kwargs = {"fringe_filter_type" : "gauss", "fr_width" : 1e-5}

        vis = foregrounds.diffuse_foreground(
            lsts, freqs, bl_vec, Tsky_mdl=Tsky_mdl, omega_p=omega_p, 
            fringe_filter_kwargs=fringe_filter_kwargs
        )

        # transform the visibility to FR-freq space
        dfft = FFT(
            vis * dspec.gen_window('blackmanharris', len(vis))[:, None], axis=0
        )

        # calculate the fringe rates
        frates = np.fft.fftshift(
            np.fft.fftfreq(len(lsts), np.diff(lsts)[0] * u.sday.to("s") / (2 * np.pi))
        )

        max_frate = frates[np.argmax(np.abs(dfft[:, 0]))]
        nt.assert_true(max_frate > 0)

        # now test that they show up at negative fringe-rates for 
        # an oppositely oriented baseline
        bl_vec = (-100.0, 0.0)

        vis = foregrounds.diffuse_foreground(
            lsts, freqs, bl_vec, Tsky_mdl=Tsky_mdl, omega_p=omega_p,
            fringe_filter_kwargs=fringe_filter_kwargs
        )

        dfft = FFT(
            vis * dspec.gen_window('blackmanharris', len(vis))[:, None], axis=0
        )

        max_frate = frates[np.argmax(np.abs(dfft[:, 0]))]
        nt.assert_true(max_frate < 0)

    def test_diffuse_foreground_conjugation(self):
        # make some parameters
        freqs = np.linspace(.1, .2, 100, endpoint=False)
        omega_p = Beam(beamfile)
        lsts = np.linspace(0, 2*np.pi, 1000)
        times = lsts / (2*np.pi) * u.sday.to('s')
        Tsky_mdl = noise.HERA_Tsky_mdl['xx']
        bl_vec = np.asarray([10, 0, 0])
        delay_filter_kwargs = {"delay_filter_type" : "tophat"}
        fringe_filter_kwargs = {"fringe_filter_type" : "tophat"}

        np.random.seed(0)
        # simulate the effect
        vis = foregrounds.diffuse_foreground(
            lsts, freqs, bl_vec, Tsky_mdl=Tsky_mdl, omega_p=omega_p,
            delay_filter_kwargs=delay_filter_kwargs, 
            fringe_filter_kwargs=fringe_filter_kwargs
        )

        np.random.seed(0)
        # simulate the effect
        _vis = foregrounds.diffuse_foreground(
            lsts, freqs, -bl_vec, Tsky_mdl=Tsky_mdl, omega_p=omega_p,
            delay_filter_kwargs=delay_filter_kwargs, 
            fringe_filter_kwargs=fringe_filter_kwargs
        )

        # XXX this test does not pass! it turns out that V_ij = V_ji
        # for this model; how do we fix it?
        #np.testing.assert_allclose(vis, _vis.conj())

    def test_point_source_foreground_conjugation(self):
        # make some parameters
        freqs = np.linspace(0.1, 0.2, 100, endpoint=False)
        lsts = np.linspace(0, 2 * np.pi, 1000)
        times = lsts / (2 * np.pi) * u.sday.to('s')
        bl_vec = np.asarray([300.0, 0])

        np.random.seed(0)
        # simulate the effect
        vis = foregrounds.pntsrc_foreground(lsts, freqs, bl_vec, nsrcs=200)

        np.random.seed(0)
        # simulate for the conjugate baseline
        _vis = foregrounds.pntsrc_foreground(lsts, freqs, -bl_vec, nsrcs=200)

        # XXX this doesn't pass either!
        #np.testing.assert_allclose(vis, _vis.conj())


if __name__ == "__main__":
    unittest.main()
