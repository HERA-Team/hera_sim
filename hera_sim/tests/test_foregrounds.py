import unittest
from hera_sim import noise, foregrounds, utils
import numpy as np
import aipy
import nose.tools as nt

np.random.seed(0)

class TestForegrounds(unittest.TestCase):
    def test_diffuse_foreground(self):
        fqs = np.linspace(.1, .2, 100, endpoint=False)
        lsts = np.linspace(0, 2*np.pi, 1000)
        times = lsts / (2*np.pi) * aipy.const.sidereal_day
        Tsky_mdl = noise.HERA_Tsky_mdl['xx']
        #Tsky = Tsky_mdl(lsts,fqs)
        bl_len_ns = 30.
        vis, ff, df = foregrounds.diffuse_foreground(Tsky_mdl, lsts, fqs, bl_len_ns, delay_filter_type='tophat', fringe_filter_type='tophat')
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
