import unittest
from hera_sim import noise
import numpy as np
import aipy

np.random.seed(0)


class TestNoise(unittest.TestCase):
    def test_bm_poly_to_omega_p(self):
        fqs = np.linspace(0.1,0.2,100)
        omega_p = noise.bm_poly_to_omega_p(fqs)
        self.assertEqual(fqs.size, omega_p.size)
    
    def test_white_noise(self):
        n1 = noise.white_noise(100)
        self.assertEqual(n1.size, 100)
        self.assertEqual(n1.shape, (100,))
        n2 = noise.white_noise((100, 100))
        self.assertEqual(n2.shape, (100, 100))
        n3 = noise.white_noise(100000)
        self.assertAlmostEqual(np.average(n3), 0, 1)
        self.assertAlmostEqual(np.std(n3), 1, 2)

    def test_resample_Tsky(self):
        fqs = np.linspace(0.1, 0.2, 100)
        lsts = np.linspace(0, 2 * np.pi, 200)
        tsky = noise.resample_Tsky(fqs, lsts)
        self.assertEqual(tsky.shape, (200, 100))
        self.assertTrue(np.all(tsky[0] == tsky[1]))
        self.assertFalse(np.all(tsky[:, 0] == tsky[:, 1]))
        # import uvtools, pylab as plt
        # uvtools.plot.waterfall(tsky); plt.show()
        tsky = noise.resample_Tsky(fqs, lsts, Tsky_mdl=noise.HERA_Tsky_mdl["xx"])
        self.assertEqual(tsky.shape, (200, 100))
        self.assertFalse(np.all(tsky[0] == tsky[1]))
        self.assertFalse(np.all(tsky[:, 0] == tsky[:, 1]))
        # import uvtools, pylab as plt
        # uvtools.plot.waterfall(tsky); plt.show()

    def test_sky_noise_jy(self):
        fqs = np.linspace(0.1, 0.2, 100)
        lsts = np.linspace(0, 2 * np.pi, 200)
        omp = noise.bm_poly_to_omega_p(fqs)
        tsky = noise.resample_Tsky(fqs, lsts)
        jy2T = noise.jy2T(fqs, omega_p=omp) / 1e3
        jy2T.shape = (1, -1)
        nos_jy = noise.sky_noise_jy(tsky, fqs, lsts, inttime=10.7, omega_p=omp)
        self.assertEqual(nos_jy.shape, (200, 100))
        np.testing.assert_allclose(np.average(nos_jy, axis=0), 0, atol=0.7)
        scaling = np.average(tsky, axis=0) / jy2T
        np.testing.assert_allclose(
            np.std(nos_jy, axis=0) / scaling * np.sqrt(1e6 * 10.7), 1.0, atol=0.1
        )
        np.random.seed(0)
        nos_jy = noise.sky_noise_jy(tsky, fqs, lsts, inttime=None, omega_p=omp)
        np.testing.assert_allclose(
            np.std(nos_jy, axis=0) / scaling * np.sqrt(1e6 * aipy.const.sidereal_day/200), 1.0, atol=0.1
        )
        np.random.seed(0)
        nos_jy = noise.sky_noise_jy(tsky, fqs, lsts, B=.1, inttime=10.7, omega_p=omp)
        np.testing.assert_allclose(
            np.std(nos_jy, axis=0) / scaling * np.sqrt(1e8 * 10.7), 1.0, atol=0.1
        )
        # tsky = noise.resample_Tsky(fqs,lsts,noise.HERA_Tsky_mdl['xx'])
        # nos_jy = noise.sky_noise_jy(tsky, fqs, lsts)
        # import uvtools, pylab as plt
        # uvtools.plot.waterfall(nos_jy, mode='real'); plt.show()


if __name__ == "__main__":
    unittest.main()
