import unittest

import numpy as np

from hera_sim import visibilities as vis
import healpy

np.random.seed(0)
NANT = 16
NTIMES = 10
BM_PIX = 31
NPIX = 12 * 16 ** 2


class TestVisCpu(unittest.TestCase):
    def test_shapes(self):
        antpos = np.zeros((NANT, 3))
        I_sky = np.zeros(NPIX)
        lsts = np.linspace(0, 2 * np.pi, NTIMES)

        v = vis.VisCPU(
            freq=0.15,
            antpos=antpos,
            latitude=0,
            lsts=lsts,
            sky_intensity=I_sky
        )

        self.assertEqual(v.simulate().shape, (NTIMES, NANT, NANT))

        # TODO: not sure how to do this in unittest properly
        # self.assertRaises(
        #     ValueError, vis.VisCPU, 0.15, antpos.T, 0, np.linspace(0, 2 * np.pi, NTIMES), I_sky
        # )

    def test_dtypes(self):
        for dtype in (np.float32, np.float64):
            for cdtype in (np.complex64, np.complex128):
                antpos = np.zeros((NANT, 3), dtype=dtype)
                I_sky = np.zeros(NPIX, dtype=dtype)
                lsts = np.linspace(0, 2 * np.pi, NTIMES, dtype=dtype)

                sim = vis.VisCPU(freq=0.15, antpos=antpos, latitude=0, lsts=lsts, sky_intensity=I_sky,
                                 real_dtype=dtype, complex_dtype=cdtype)

                v = sim.simulate()
                self.assertEqual(v.dtype, cdtype)

    def test_zero_sky(self):
        antpos = np.ones((NANT, 3))
        lsts = np.linspace(0, 2 * np.pi, NTIMES)
        crd_eq = np.zeros((3, NPIX))
        crd_eq[2] = 1
        I_sky = np.ones(NPIX)

        v = vis.VisCPU(
            freq=1,
            antpos=antpos,
            latitude=0,
            lsts=lsts,
            sky_intensity=0 * I_sky
        ).simulate()
        np.testing.assert_equal(v, 0)

    def test_zero_beam(self):
        antpos = np.ones((NANT, 3))
        lsts = np.linspace(0, 2 * np.pi, NTIMES)
        I_sky = np.ones(NPIX)

        v = vis.VisCPU(
            freq=1,
            antpos=antpos,
            latitude=0,
            lsts=lsts,
            sky_intensity=I_sky,
            beams=np.array([np.zeros(NPIX)]),
            beam_ids=np.zeros(NANT, dtype=np.int)
        ).simulate()

        np.testing.assert_equal(v, 0)

    def test_colocation(self):
        # For co-located ants & sources on sky, answer should be sum of pixels
        # (over half the sky)
        antpos = np.ones((NANT, 3))
        lsts = np.linspace(0, 2 * np.pi, NTIMES)
        I_sky = np.ones(NPIX)

        for i, (dtype, ctype) in enumerate([(np.float32, np.complex64),
                                            (np.float64, np.complex128)]):
            v = vis.VisCPU(
                freq=1,
                antpos=antpos,
                latitude=0,
                lsts=lsts,
                sky_intensity=I_sky,
                real_dtype=dtype,
                complex_dtype=ctype
            ).simulate()

            np.testing.assert_almost_equal(v, NPIX / 2, [2, 10][i])

    def test_two_sources(self):
        # For co-located ants & two sources separated on sky, answer should still be sum

        antpos = np.ones((NANT, 3))

        point_sources = np.array([[0, 0, 1], [0, 0.1, 1]])

        v = vis.VisCPU(
            freq=1,
            antpos=antpos,
            latitude=0,
            lsts=np.array([0]),
            point_sources=point_sources
        ).simulate()

        np.testing.assert_almost_equal(v, 2/healpy.nside2pixarea(16), 2)

    # def test_exact_value_two_sources(self):
    #
    #     # For ant[0] at (0,0,1), ant[1] at (1,1,1), src[0] at (0,0,1) and src[1] at (0,.707,.707)
    #     antpos[0, 0] = 0
    #     antpos[0, 1] = 0
    #     v = simulators.vis_cpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube)
    #     np.testing.assert_almost_equal(
    #         v[:, 0, 1], 1 + np.exp(-2j * np.pi * np.sqrt(0.5)), 7
    #     )

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
