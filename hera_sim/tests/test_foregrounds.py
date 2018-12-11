import unittest
from hera_sim import noise, foregrounds, utils
import numpy as np
import aipy

np.random.seed(0)

class TestForegrounds(unittest.TestCase):
    def test_rough_delay_filter(self):
        fqs = np.linspace(.1,.2,100,endpoint=False)
        n1 = noise.white_noise((100,fqs.size))
        bl_len_ns = 30.
        dlys = np.fft.fftfreq(fqs.size, fqs[1]-fqs[0])
        n1_filt = utils.rough_delay_filter(n1, fqs, bl_len_ns)
        _n1_filt = np.fft.ifft(n1_filt, axis=-1)
        np.testing.assert_array_less(np.mean(np.abs(_n1_filt), axis=0), 
            np.exp(-dlys**2 / (2*bl_len_ns**2)).clip(1e-15,1))
        #import uvtools, pylab as plt
        #uvtools.plot.waterfall(_n1_filt, mode='log', drng=3); plt.show()
        #plt.plot(np.mean(np.abs(_n1_filt), axis=0))
        #plt.plot( np.exp(-dlys**2 / (2*bl_len_ns**2)))
        #plt.show()
    def test_rough_fringe_filter(self):
        fqs = np.linspace(.1,.2,100,endpoint=False)
        lsts = np.linspace(0,2*np.pi,1000)
        times = lsts / (2*np.pi) * aipy.const.sidereal_day
        n1 = noise.white_noise((lsts.size,fqs.size))
        bl_len_ns = 30.
        n1_filt, ff, fr = utils.rough_fringe_filter(n1, lsts, fqs, bl_len_ns)
        _n1_filt = np.fft.fft(n1_filt, axis=-2)
        fr_max1 = utils.calc_max_fringe_rate(.1, bl_len_ns)
        fr_max2 = utils.calc_max_fringe_rate(.2, bl_len_ns)
        fringe_rates = np.fft.fftfreq(times.size, times[1]-times[0])
        inside_filter = np.where(np.abs(fringe_rates) < fr_max1)[0]
        outside_filter = np.where(np.abs(fringe_rates) > fr_max2, 1, 0)
        outside_filter.shape = (-1,1)
        np.testing.assert_allclose(np.sqrt(np.mean(np.abs(_n1_filt[inside_filter,:])**2,axis=-1)), np.sqrt(lsts.size), rtol=.15)
        np.testing.assert_array_less(np.abs(_n1_filt*outside_filter), 1e-10)
        #import uvtools, pylab as plt
        #uvtools.plot.waterfall(_n1_filt * outside_filter, mode='log', drng=3); plt.show()
        #plt.plot(np.mean(np.abs(_n1_filt), axis=0))
        #plt.plot( np.exp(-dlys**2 / (2*bl_len_ns**2)))
        #plt.show()
    def test_diffuse_foreground(self):
        fqs = np.linspace(.1,.2,100,endpoint=False)
        lsts = np.linspace(0,2*np.pi,1000)
        times = lsts / (2*np.pi) * aipy.const.sidereal_day
        Tsky_mdl = noise.HERA_Tsky_mdl['xx']
        #Tsky = Tsky_mdl(lsts,fqs)
        bl_len_ns = 30.
        vis = foregrounds.diffuse_foreground(Tsky_mdl, lsts, fqs, bl_len_ns)
        self.assertEqual(vis.shape, (lsts.size,fqs.size))
        # XXX check more substantial things
        #import uvtools, pylab as plt
        #uvtools.plot.waterfall(vis, mode='log'); plt.colorbar(); plt.show()
    def test_pntsrc_foreground(self):
        fqs = np.linspace(.1,.2,100,endpoint=False)
        lsts = np.linspace(0,2*np.pi,1000)
        times = lsts / (2*np.pi) * aipy.const.sidereal_day
        bl_len_ns = 300.
        vis = foregrounds.pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=200)
        self.assertEqual(vis.shape, (lsts.size,fqs.size))
        # XXX check more substantial things
        #import uvtools, pylab as plt
        #uvtools.plot.waterfall(vis, mode='phs'); plt.colorbar(); plt.show()
    

        
        

if __name__ == '__main__':
    unittest.main()
