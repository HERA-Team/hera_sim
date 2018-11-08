import unittest
from hera_sim import vis
import numpy as np

np.random.seed(0)
NANT = 16
NTIMES = 10
BM_PIX = 31
NPIX = 12 * 16**2

class TestVisCpu(unittest.TestCase):
    def test_shapes(self):
        antpos = np.zeros((NANT, 3))
        eq2tops = np.zeros((NTIMES,3,3))
        crd_eq = np.zeros((3,NPIX))
        I_sky = np.zeros(NPIX)
        bm_cube = np.zeros((NANT,BM_PIX, BM_PIX))
        v = vis.vis_cpu(antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube)
        self.assertEqual(v.shape, (NTIMES,NANT,NANT))
        self.assertRaises(AssertionError, vis.vis_cpu, antpos.T, 0.15, eq2tops, crd_eq, I_sky, bm_cube)
        self.assertRaises(AssertionError, vis.vis_cpu, antpos, 0.15, eq2tops.T, crd_eq, I_sky, bm_cube)
        self.assertRaises(AssertionError, vis.vis_cpu, antpos, 0.15, eq2tops, crd_eq.T, I_sky, bm_cube)
        self.assertRaises(AssertionError, vis.vis_cpu, antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube.T)
    def test_dtypes(self):
        for dtype in (np.float32, np.float64):
            antpos = np.zeros((NANT, 3), dtype=dtype)
            eq2tops = np.zeros((NTIMES,3,3), dtype=dtype)
            crd_eq = np.zeros((3,NPIX), dtype=dtype)
            I_sky = np.zeros(NPIX, dtype=dtype)
            bm_cube = np.zeros((NANT,BM_PIX,BM_PIX), dtype=dtype)
            v = vis.vis_cpu(antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube)
            self.assertEqual(v.dtype, np.complex64)
            v = vis.vis_cpu(antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
            self.assertEqual(v.dtype, np.complex128)
    def test_values(self):
        antpos = np.ones((NANT, 3))
        eq2tops = np.array([np.identity(3)] * NTIMES)
        crd_eq = np.zeros((3,NPIX)); crd_eq[2] = 1
        I_sky = np.ones(NPIX)
        bm_cube = np.ones((NANT,BM_PIX,BM_PIX))
        # Make sure that a zero in sky or beam gives zero output
        v = vis.vis_cpu(antpos, 1., eq2tops, crd_eq, I_sky*0, bm_cube)
        np.testing.assert_equal(v, 0)
        v = vis.vis_cpu(antpos, 1., eq2tops, crd_eq, I_sky, bm_cube*0)
        np.testing.assert_equal(v, 0)
        # For co-located ants & sources on sky, answer should be sum of pixels
        v = vis.vis_cpu(antpos, 1., eq2tops, crd_eq, I_sky, bm_cube)
        np.testing.assert_almost_equal(v, NPIX, 2)
        v = vis.vis_cpu(antpos, 1., eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float64, complex_dtype=np.complex128)
        np.testing.assert_almost_equal(v, NPIX, 10)
        # For co-located ants & two sources separated on sky, answer should still be sum
        crd_eq = np.zeros((3,2))
        crd_eq[2,0] = 1
        crd_eq[1,1] = np.sqrt(.5)
        crd_eq[2,1] = np.sqrt(.5)
        I_sky = np.ones(2)
        v = vis.vis_cpu(antpos, 1., eq2tops, crd_eq, I_sky, bm_cube)
        np.testing.assert_almost_equal(v, 2, 2)
        # For ant[0] at (0,0,1), ant[1] at (1,1,1), src[0] at (0,0,1) and src[1] at (0,.707,.707)
        antpos[0,0] = 0
        antpos[0,1] = 0
        v = vis.vis_cpu(antpos, 1., eq2tops, crd_eq, I_sky, bm_cube)
        np.testing.assert_almost_equal(v[:,0,1], 1 + np.exp(-2j * np.pi * np.sqrt(.5)), 7)
        

if __name__ == '__main__':
    unittest.main()
