import unittest
from hera_sim import noise, foregrounds
import numpy as np
import aipy
import nose.tools as nt
from uvtools import dspec

np.random.seed(0)


class TestForegrounds(unittest.TestCase):
    def test_diffuse_foreground(self):
        fqs = np.linspace(.1, .2, 100, endpoint=False)
        omega_p = noise.bm_poly_to_omega_p(fqs)
        lsts = np.linspace(0, 2*np.pi, 1000)
        times = lsts / (2*np.pi) * aipy.const.sidereal_day
        Tsky_mdl = noise.HERA_Tsky_mdl['xx']
        #Tsky = Tsky_mdl(lsts,fqs)
        bl_len_ns = 30.
        vis = foregrounds.diffuse_foreground(lsts, fqs, [bl_len_ns, 0, 0], Tsky_mdl=Tsky_mdl, delay_filter_type='tophat',
                                             fringe_filter_type='tophat', omega_p=omega_p)
        self.assertEqual(vis.shape, (lsts.size,fqs.size))
        # XXX check more substantial things
        #import uvtools, pylab as plt
        #uvtools.plot.waterfall(vis, mode='log'); plt.colorbar(); plt.show()
        nt.assert_raises(TypeError, foregrounds.diffuse_foreground, lsts, fqs, [bl_len_ns])

    def test_pntsrc_foreground(self):
        fqs = np.linspace(0.1, 0.2, 100, endpoint=False)
        lsts = np.linspace(0, 2 * np.pi, 1000)
        times = lsts / (2 * np.pi) * aipy.const.sidereal_day
        bl_vec = [300.0, 0]
        vis = foregrounds.pntsrc_foreground(lsts, fqs, bl_vec, nsrcs=200)
        self.assertEqual(vis.shape, (lsts.size, fqs.size))
        # XXX check more substantial things
        # import uvtools, pylab as plt
        # uvtools.plot.waterfall(vis, mode='phs'); plt.colorbar(); plt.show()

    def test_diffuse_foreground_orientation(self):
        fqs = np.linspace(.1, .2, 100, endpoint=False)
        omega_p = noise.bm_poly_to_omega_p(fqs)
        lsts = np.linspace(0, 2 * np.pi, 1000)
        Tsky_mdl = noise.HERA_Tsky_mdl['xx']

        bl_vec = (0, 30.0)
        vis = foregrounds.diffuse_foreground(lsts, fqs, bl_vec, Tsky_mdl=Tsky_mdl, fringe_filter_type='tophat', omega_p=omega_p)
        self.assertEqual(vis.shape, (lsts.size, fqs.size))

        # assert foregrounds show up at positive fringe-rates for FFT
        bl_vec = (100.0, 0.0)
        vis = foregrounds.diffuse_foreground(lsts, fqs, bl_vec, Tsky_mdl=Tsky_mdl, fringe_filter_type='gauss', fr_width=1e-5, omega_p=omega_p)
        dfft = np.fft.fftshift(np.fft.fft(vis * dspec.gen_window('blackmanharris', len(vis))[:, None], axis=0), axes=0)
        frates = np.fft.fftshift(np.fft.fftfreq(len(lsts), np.diff(lsts)[0]*12*3600/np.pi))
        max_frate = frates[np.argmax(np.abs(dfft[:, 0]))]
        nt.assert_true(max_frate > 0)

        bl_vec = (-100.0, 0.0)
        vis = foregrounds.diffuse_foreground(lsts, fqs, bl_vec, Tsky_mdl=Tsky_mdl, fringe_filter_type='gauss', fr_width=1e-5, omega_p=omega_p)
        dfft = np.fft.fftshift(np.fft.fft(vis * dspec.gen_window('blackmanharris', len(vis))[:, None], axis=0), axes=0)
        max_frate = frates[np.argmax(np.abs(dfft[:, 0]))]
        nt.assert_true(max_frate < 0)


if __name__ == "__main__":
    unittest.main()
