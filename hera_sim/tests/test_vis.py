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


if __name__ == "__main__":
    unittest.main()
