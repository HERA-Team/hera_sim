import unittest
from hera_sim import rfi
import numpy as np

np.random.seed(0)


class TestRFI(unittest.TestCase):
    def test_RfiStation(self):
        s = rfi.RfiStation(0.15, std=0.0)
        fqs = np.array([0.146, 0.147, 0.148, 0.149, 0.150, 0.151, 0.152, 0.153])
        lsts = np.linspace(0, 2 * np.pi, 100)
        r = s.gen_rfi(fqs, lsts)
        self.assertEqual(r.shape, (lsts.size, fqs.size))
        np.testing.assert_allclose(r[:, 4], s.strength, 4)
        np.testing.assert_allclose(r[:, 3], 0, 4)
        np.testing.assert_allclose(r[:, 5], 0, 4)
        s = rfi.RfiStation(0.1505, std=0.0)
        fqs = np.array([0.146, 0.147, 0.148, 0.149, 0.150, 0.151, 0.152, 0.153])
        lsts = np.linspace(0, 2 * np.pi, 100)
        r = s.gen_rfi(fqs, lsts)
        self.assertEqual(r.shape, (lsts.size, fqs.size))
        np.testing.assert_allclose(r[:, 4], s.strength / 2, 4)
        np.testing.assert_allclose(r[:, 3], 0, 4)
        np.testing.assert_allclose(r[:, 5], s.strength / 2, 4)

    def test_rfi_stations(self):
        # fqs = np.linspace(.1,.2,1024)
        fqs = np.linspace(0.1, 0.2, 100)
        lsts = np.linspace(0, 2 * np.pi, 1000)
        r = rfi.rfi_stations(fqs, lsts)
        self.assertEqual(r.shape, (lsts.size, fqs.size))
        # import uvtools, pylab as plt
        # uvtools.plot.waterfall(r); plt.show()

    def test_rfi_impulse(self):
        fqs = np.linspace(0.1, 0.2, 100)
        lsts = np.linspace(0, 2 * np.pi, 200)
        r = rfi.rfi_impulse(fqs, lsts, chance=0.5)
        self.assertEqual(r.shape, (lsts.size, fqs.size))
        self.assertAlmostEqual(np.where(np.abs(r) > 0)[0].size / float(r.size), 0.5, -1)
        # import uvtools, pylab as plt
        # uvtools.plot.waterfall(r); plt.show()

    def test_rfi_scatter(self):
        fqs = np.linspace(0.1, 0.2, 100)
        lsts = np.linspace(0, 2 * np.pi, 200)
        r = rfi.rfi_scatter(fqs, lsts, chance=0.5)
        self.assertEqual(r.shape, (lsts.size, fqs.size))
        self.assertAlmostEqual(np.where(np.abs(r) > 0)[0].size, r.size / 2, -3)
        # import uvtools, pylab as plt
        # uvtools.plot.waterfall(r); plt.show()

    def test_rfi_dtv(self):
        fqs = np.linspace(0.1, 0.2, 100)
        lsts = np.linspace(0, 2 * np.pi, 200)

        # Scalar chance.
        r = rfi.rfi_dtv(fqs, lsts, freq_min=0.15, freq_max=0.20, width=0.01,
                        chance=0.5, strength=10, strength_std=1)
        self.assertEqual(r.shape, (lsts.size, fqs.size))
        self.assertAlmostEqual(np.where(np.abs(r) > 0)[0].size / float(r.size), 0.5,  -1)

        # Ensure that a sub-band is the same across frequencies.
        np.testing.assert_allclose(np.mean(r[:, np.logical_and(fqs>=0.15, fqs<0.16)], axis=1), r[:, 50])
        np.testing.assert_allclose(np.mean(r[:, np.logical_and(fqs>=0.19, fqs<0.2)], axis=1), r[:, -1])

        # List of chances.
        r = rfi.rfi_dtv(fqs, lsts, freq_min=0.15, freq_max=0.20, width=0.01,
                        chance=[0.5, 0.6, 0.7, 0.8, 1.0], strength=10, strength_std=1e-5)

        # Ensure that the correct chance gets applied to each band
        self.assertAlmostEqual(np.sum(np.abs(r[:, -1])), lsts.size * 10, -2)
        self.assertAlmostEqual(np.sum(np.abs(r[:, 0])), 0, -5)


if __name__ == "__main__":
    unittest.main()
