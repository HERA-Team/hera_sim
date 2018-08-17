import unittest
from hera_sim import rfi
import numpy as np

np.random.seed(0)

class TestRFI(unittest.TestCase):
    def test_RfiStation(self):
        s = rfi.RfiStation(.15, std=0.)
        fqs = np.array([.146,.147,.148,.149,.150,.151,.152,.153])
        lsts = np.linspace(0,2*np.pi,100)
        r = s.gen_rfi(fqs,lsts)
        self.assertEqual(r.shape, (lsts.size, fqs.size))
        np.testing.assert_allclose(r[:,4], s.strength, 4)
        np.testing.assert_allclose(r[:,3], 0, 4)
        np.testing.assert_allclose(r[:,5], 0, 4)
        s = rfi.RfiStation(.1505, std=0.)
        fqs = np.array([.146,.147,.148,.149,.150,.151,.152,.153])
        lsts = np.linspace(0,2*np.pi,100)
        r = s.gen_rfi(fqs,lsts)
        self.assertEqual(r.shape, (lsts.size, fqs.size))
        np.testing.assert_allclose(r[:,4], s.strength/2, 4)
        np.testing.assert_allclose(r[:,3], 0, 4)
        np.testing.assert_allclose(r[:,5], s.strength/2, 4)
    def test_rfi_stations(self):
        #fqs = np.linspace(.1,.2,1024)
        fqs = np.linspace(.1,.2,100)
        lsts = np.linspace(0,2*np.pi,1000)
        r = rfi.rfi_stations(fqs, lsts)
        self.assertEqual(r.shape, (lsts.size, fqs.size))
        #import uvtools, pylab as plt
        #uvtools.plot.waterfall(r); plt.show()
    def test_rfi_impulse(self):
        fqs = np.linspace(.1,.2,100)
        lsts = np.linspace(0,2*np.pi,200)
        r = rfi.rfi_impulse(fqs, lsts, chance=.5)
        self.assertEqual(r.shape, (lsts.size, fqs.size))
        self.assertAlmostEqual(np.where(np.abs(r) > 0)[0].size, r.size/2, -3)
        #import uvtools, pylab as plt
        #uvtools.plot.waterfall(r); plt.show()
    def test_rfi_scatter(self):
        fqs = np.linspace(.1,.2,100)
        lsts = np.linspace(0,2*np.pi,200)
        r = rfi.rfi_scatter(fqs, lsts, chance=.5)
        self.assertEqual(r.shape, (lsts.size, fqs.size))
        self.assertAlmostEqual(np.where(np.abs(r) > 0)[0].size, r.size/2, -3)
        #import uvtools, pylab as plt
        #uvtools.plot.waterfall(r); plt.show()
        
        

if __name__ == '__main__':
    unittest.main()
