import unittest
from hera_sim.visibilities import vis_cpu
import numpy as np

np.random.seed(0)
NANT = 16
NTIMES = 10
BM_PIX = 31
NPIX = 12 * 16 ** 2


class TestVisCpu(unittest.TestCase):
    def test_shapes(self):
        antpos = np.zeros((NANT, 3))
        eq2tops = np.zeros((NTIMES, 3, 3))
        crd_eq = np.zeros((3, NPIX))
        I_sky = np.zeros(NPIX)
        bm_cube = np.zeros((NANT, BM_PIX, BM_PIX))
        v = vis_cpu.vis_cpu(antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube)
        self.assertEqual(v.shape, (NTIMES, NANT, NANT))
        self.assertRaises(
            AssertionError, vis_cpu.vis_cpu, antpos.T, 0.15, eq2tops, crd_eq, I_sky, bm_cube
        )
        self.assertRaises(
            AssertionError, vis_cpu.vis_cpu, antpos, 0.15, eq2tops.T, crd_eq, I_sky, bm_cube
        )
        self.assertRaises(
            AssertionError, vis_cpu.vis_cpu, antpos, 0.15, eq2tops, crd_eq.T, I_sky, bm_cube
        )
        self.assertRaises(
            AssertionError, vis_cpu.vis_cpu, antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube.T
        )

    def test_dtypes(self):
        for dtype in (np.float32, np.float64):
            antpos = np.zeros((NANT, 3), dtype=dtype)
            eq2tops = np.zeros((NTIMES, 3, 3), dtype=dtype)
            crd_eq = np.zeros((3, NPIX), dtype=dtype)
            I_sky = np.zeros(NPIX, dtype=dtype)
            bm_cube = np.zeros((NANT, BM_PIX, BM_PIX), dtype=dtype)
            v = vis_cpu.vis_cpu(antpos, 0.15, eq2tops, crd_eq, I_sky, bm_cube)
            self.assertEqual(v.dtype, np.complex64)
            v = vis_cpu.vis_cpu(
                antpos,
                0.15,
                eq2tops,
                crd_eq,
                I_sky,
                bm_cube,
                real_dtype=np.float64,
                complex_dtype=np.complex128,
            )
            self.assertEqual(v.dtype, np.complex128)

    def test_values(self):
        antpos = np.ones((NANT, 3))
        eq2tops = np.array([np.identity(3)] * NTIMES)
        crd_eq = np.zeros((3, NPIX))
        crd_eq[2] = 1
        I_sky = np.ones(NPIX)
        bm_cube = np.ones((NANT, BM_PIX, BM_PIX))
        # Make sure that a zero in sky or beam gives zero output
        v = vis_cpu.vis_cpu(antpos, 1.0, eq2tops, crd_eq, I_sky * 0, bm_cube)
        np.testing.assert_equal(v, 0)
        v = vis_cpu.vis_cpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube * 0)
        np.testing.assert_equal(v, 0)
        # For co-located ants & sources on sky, answer should be sum of pixels
        v = vis_cpu.vis_cpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube)
        np.testing.assert_almost_equal(v, NPIX, 2)
        v = vis_cpu.vis_cpu(
            antpos,
            1.0,
            eq2tops,
            crd_eq,
            I_sky,
            bm_cube,
            real_dtype=np.float64,
            complex_dtype=np.complex128,
        )
        np.testing.assert_almost_equal(v, NPIX, 10)
        # For co-located ants & two sources separated on sky, answer should still be sum
        crd_eq = np.zeros((3, 2))
        crd_eq[2, 0] = 1
        crd_eq[1, 1] = np.sqrt(0.5)
        crd_eq[2, 1] = np.sqrt(0.5)
        I_sky = np.ones(2)
        v = vis_cpu.vis_cpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube)
        np.testing.assert_almost_equal(v, 2, 2)
        # For ant[0] at (0,0,1), ant[1] at (1,1,1), src[0] at (0,0,1) and src[1] at (0,.707,.707)
        antpos[0, 0] = 0
        antpos[0, 1] = 0
        v = vis_cpu.vis_cpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube)
        np.testing.assert_almost_equal(
            v[:, 0, 1], 1 + np.exp(-2j * np.pi * np.sqrt(0.5)), 7
        )

class TestSimRedData(unittest.TestCase):

    def test_sim_red_data(self):
        # Test that redundant baselines are redundant up to the gains in single pol mode
        #antpos = build_linear_array(5)
        #reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        # Hard code redundancies to eliminate dependence on hera_cal
        reds = [[(0, 1, 'xx'), (1, 2, 'xx'), (2, 3, 'xx'), (3, 4, 'xx')], 
                [(0, 2, 'xx'), (1, 3, 'xx'), (2, 4, 'xx')],
                [(0, 3, 'xx'), (1, 4, 'xx')],
                [(0, 4, 'xx')]]
        gains, true_vis, data = vis.sim_red_data(reds)
        assert len(gains) == 5
        assert len(data) == 10
        for bls in reds:
            bl0 = bls[0]
            ai, aj, pol = bl0
            ans0 = data[bl0] / (gains[(ai, 'Jxx')] * gains[(aj, 'Jxx')].conj())
            for bl in bls[1:]:
                ai, aj, pol = bl
                ans = data[bl] / (gains[(ai, 'Jxx')] * gains[(aj, 'Jxx')].conj())
                # compare calibrated visibilities knowing the input gains
                np.testing.assert_almost_equal(ans0, ans, decimal=7)

        # Test that redundant baselines are redundant up to the gains in 4-pol mode
        #reds = om.get_reds(antpos, pols=['xx', 'yy', 'xy', 'yx'], pol_mode='4pol')
        # Hard code redundancies to eliminate dependence on hera_cal
        reds = [[(0, 1, 'xx'), (1, 2, 'xx'), (2, 3, 'xx'), (3, 4, 'xx')],
                [(0, 2, 'xx'), (1, 3, 'xx'), (2, 4, 'xx')],
                [(0, 3, 'xx'), (1, 4, 'xx')],
                [(0, 4, 'xx')],
                [(0, 1, 'yy'), (1, 2, 'yy'), (2, 3, 'yy'), (3, 4, 'yy')],
                [(0, 2, 'yy'), (1, 3, 'yy'), (2, 4, 'yy')],
                [(0, 3, 'yy'), (1, 4, 'yy')],
                [(0, 4, 'yy')],
                [(0, 1, 'xy'), (1, 2, 'xy'), (2, 3, 'xy'), (3, 4, 'xy')],
                [(0, 2, 'xy'), (1, 3, 'xy'), (2, 4, 'xy')],
                [(0, 3, 'xy'), (1, 4, 'xy')],
                [(0, 4, 'xy')],
                [(0, 1, 'yx'), (1, 2, 'yx'), (2, 3, 'yx'), (3, 4, 'yx')],
                [(0, 2, 'yx'), (1, 3, 'yx'), (2, 4, 'yx')],
                [(0, 3, 'yx'), (1, 4, 'yx')],
                [(0, 4, 'yx')]]
        gains, true_vis, data = vis.sim_red_data(reds)
        assert len(gains) == 2 * (5)
        assert len(data) == 4 * (10)
        for bls in reds:
            bl0 = bls[0]
            ai, aj, pol = bl0
            ans0xx = data[(ai, aj, 'xx',)] / (gains[(ai, 'Jxx')] * gains[(aj, 'Jxx')].conj())
            ans0xy = data[(ai, aj, 'xy',)] / (gains[(ai, 'Jxx')] * gains[(aj, 'Jyy')].conj())
            ans0yx = data[(ai, aj, 'yx',)] / (gains[(ai, 'Jyy')] * gains[(aj, 'Jxx')].conj())
            ans0yy = data[(ai, aj, 'yy',)] / (gains[(ai, 'Jyy')] * gains[(aj, 'Jyy')].conj())
            for bl in bls[1:]:
                ai, aj, pol = bl
                ans_xx = data[(ai, aj, 'xx',)] / (gains[(ai, 'Jxx')] * gains[(aj, 'Jxx')].conj())
                ans_xy = data[(ai, aj, 'xy',)] / (gains[(ai, 'Jxx')] * gains[(aj, 'Jyy')].conj())
                ans_yx = data[(ai, aj, 'yx',)] / (gains[(ai, 'Jyy')] * gains[(aj, 'Jxx')].conj())
                ans_yy = data[(ai, aj, 'yy',)] / (gains[(ai, 'Jyy')] * gains[(aj, 'Jyy')].conj())
                # compare calibrated visibilities knowing the input gains
                np.testing.assert_almost_equal(ans0xx, ans_xx, decimal=7)
                np.testing.assert_almost_equal(ans0xy, ans_xy, decimal=7)
                np.testing.assert_almost_equal(ans0yx, ans_yx, decimal=7)
                np.testing.assert_almost_equal(ans0yy, ans_yy, decimal=7)

        # Test that redundant baselines are redundant up to the gains in 4-pol minV mode (where Vxy = Vyx)
        #reds = om.get_reds(antpos, pols=['xx', 'yy', 'xy', 'yX'], pol_mode='4pol_minV')
        # Hard code redundancies to eliminate dependence on hera_cal
        reds = [[(0, 1, 'xx'), (1, 2, 'xx'), (2, 3, 'xx'), (3, 4, 'xx')],
                [(0, 2, 'xx'), (1, 3, 'xx'), (2, 4, 'xx')],
                [(0, 3, 'xx'), (1, 4, 'xx')],
                [(0, 4, 'xx')],
                [(0, 1, 'yy'), (1, 2, 'yy'), (2, 3, 'yy'), (3, 4, 'yy')],
                [(0, 2, 'yy'), (1, 3, 'yy'), (2, 4, 'yy')],
                [(0, 3, 'yy'), (1, 4, 'yy')],
                [(0, 4, 'yy')],
                [(0, 1, 'xy'),
                 (1, 2, 'xy'),
                 (2, 3, 'xy'),
                 (3, 4, 'xy'),
                 (0, 1, 'yx'),
                 (1, 2, 'yx'),
                 (2, 3, 'yx'),
                 (3, 4, 'yx')],
                [(0, 2, 'xy'),
                 (1, 3, 'xy'),
                 (2, 4, 'xy'),
                 (0, 2, 'yx'),
                 (1, 3, 'yx'),
                 (2, 4, 'yx')],
                [(0, 3, 'xy'), (1, 4, 'xy'), (0, 3, 'yx'), (1, 4, 'yx')],
                [(0, 4, 'xy'), (0, 4, 'yx')]]
        gains, true_vis, data = vis.sim_red_data(reds)
        assert len(gains) == 2 * (5)
        assert len(data) == 4 * (10)
        for bls in reds:
            bl0 = bls[0]
            ai, aj, pol = bl0
            ans0xx = data[(ai, aj, 'xx',)] / (gains[(ai, 'Jxx')] * gains[(aj, 'Jxx')].conj())
            ans0xy = data[(ai, aj, 'xy',)] / (gains[(ai, 'Jxx')] * gains[(aj, 'Jyy')].conj())
            ans0yx = data[(ai, aj, 'yx',)] / (gains[(ai, 'Jyy')] * gains[(aj, 'Jxx')].conj())
            ans0yy = data[(ai, aj, 'yy',)] / (gains[(ai, 'Jyy')] * gains[(aj, 'Jyy')].conj())
            np.testing.assert_almost_equal(ans0xy, ans0yx, decimal=7)
            for bl in bls[1:]:
                ai, aj, pol = bl
                ans_xx = data[(ai, aj, 'xx',)] / (gains[(ai, 'Jxx')] * gains[(aj, 'Jxx')].conj())
                ans_xy = data[(ai, aj, 'xy',)] / (gains[(ai, 'Jxx')] * gains[(aj, 'Jyy')].conj())
                ans_yx = data[(ai, aj, 'yx',)] / (gains[(ai, 'Jyy')] * gains[(aj, 'Jxx')].conj())
                ans_yy = data[(ai, aj, 'yy',)] / (gains[(ai, 'Jyy')] * gains[(aj, 'Jyy')].conj())
                # compare calibrated visibilities knowing the input gains
                np.testing.assert_almost_equal(ans0xx, ans_xx, decimal=7)
                np.testing.assert_almost_equal(ans0xy, ans_xy, decimal=7)
                np.testing.assert_almost_equal(ans0yx, ans_yx, decimal=7)
                np.testing.assert_almost_equal(ans0yy, ans_yy, decimal=7)

if __name__ == "__main__":
    unittest.main()
