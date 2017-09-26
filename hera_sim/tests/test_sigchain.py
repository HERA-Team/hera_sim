import unittest
import sigchain
import numpy as np
import aipy

np.random.seed(0)

class TestSigchain(unittest.TestCase):
    def test_gen_bandpass(self):
        fqs = np.linspace(.1,.2,1024,endpoint=False)
        g = sigchain.gen_bandpass(fqs, [1,2], gain_spread=0)
        self.assertTrue(g.has_key(1))
        self.assertTrue(g.has_key(2))
        self.assertEqual(g[1].size, fqs.size)
        np.testing.assert_array_equal(g[1],g[2])
        g = sigchain.gen_bandpass(fqs, range(10), .2)
        self.assertFalse(np.all(g[1] == g[2]))
    def test_gen_delay_phs(self):
        fqs = np.linspace(.12,.18,1024,endpoint=False)
        phs = sigchain.gen_delay_phs(fqs, [1,2], dly_rng=(0,20))
        self.assertEqual(len(phs), 2)
        self.assertTrue(phs.has_key(1))
        self.assertTrue(phs.has_key(2))
        np.testing.assert_almost_equal(np.abs(phs[1]), 1.)
        p = np.polyfit(fqs, np.unwrap(np.angle(phs[1])), deg=1)
        self.assertAlmostEqual(p[-1] % (2*np.pi), 0., -2)
        self.assertLessEqual(p[0], 20*2*np.pi)
        self.assertGreaterEqual(p[0], 0*2*np.pi)
    def test_gen_gains(self):
        fqs = np.linspace(.12,.18,1024,endpoint=False)
        g = sigchain.gen_gains(fqs, [1,2], gain_spread=0, dly_rng=(10,20))
        np.testing.assert_allclose(np.abs(g[1]), np.abs(g[2]), 1e-5)
        for i in g:
            p = np.polyfit(fqs, np.unwrap(np.angle(g[i])), deg=1)
            self.assertAlmostEqual(p[-1] % (2*np.pi), 0., -2)
            self.assertLessEqual(p[0], 20*2*np.pi)
            self.assertGreaterEqual(p[0], 10*2*np.pi)
    def test_apply_gains(self):
        fqs = np.linspace(.12,.18,1024,endpoint=False)
        vis = np.ones((100,fqs.size), dtype=np.complex)
        g = sigchain.gen_gains(fqs, [1,2], gain_spread=0, dly_rng=(10,10))
        gvis = sigchain.apply_gains(vis, g, (1,2))
        np.testing.assert_allclose(np.angle(gvis), 0, 1e-5)
    

        
        

if __name__ == '__main__':
    unittest.main()
