import os
import unittest
from hera_sim import rfi
from hera_sim import DATA_PATH
import numpy as np

np.random.seed(0)


class TestRFI(unittest.TestCase):
    def test_RfiStation(self):
        # make a station at 0.15 GHz exactly
        s = rfi.RfiStation(0.15, std=0.0)

        # choose some frequencies and lsts
        freqs = np.array([0.146, 0.147, 0.148, 0.149, 0.150, 0.151, 0.152, 0.153])
        lsts = np.linspace(0, 2 * np.pi, 100)

        # generate the rfi
        r = s(lsts, freqs)
        self.assertEqual(r.shape, (lsts.size, freqs.size))
        np.testing.assert_allclose(r[:, 4], s.strength, 4)
        np.testing.assert_allclose(r[:, 3], 0, 4)
        np.testing.assert_allclose(r[:, 5], 0, 4)

        # now make a station at 150.5 MHz exactly
        s = rfi.RfiStation(0.1505, std=0.0)

        # choose parameters
        freqs = np.array([0.146, 0.147, 0.148, 0.149, 0.150, 0.151, 0.152, 0.153])
        lsts = np.linspace(0, 2 * np.pi, 100)

        # generate rfi
        r = s(lsts, freqs)
        self.assertEqual(r.shape, (lsts.size, freqs.size))
        np.testing.assert_allclose(r[:, 4], s.strength / 2, 4)
        np.testing.assert_allclose(r[:, 3], 0, 4)
        np.testing.assert_allclose(r[:, 5], s.strength / 2, 4)

    def test_rfi_stations(self):
        # choose observing parameters
        freqs = np.linspace(0.1, 0.2, 100)
        lsts = np.linspace(0, 2 * np.pi, 1000)

        # choose RFI station parameters
        stations = DATA_PATH / "HERA_H1C_RFI_STATIONS.npy"

        # generate rfi
        r = rfi.rfi_stations(lsts, freqs, stations=stations)
        self.assertEqual(r.shape, (lsts.size, freqs.size))
        # import uvtools, pylab as plt
        # uvtools.plot.waterfall(r); plt.show()

    def test_rfi_impulse(self):
        # choose observing parameters
        freqs = np.linspace(0.1, 0.2, 100)
        lsts = np.linspace(0, 2 * np.pi, 200)

        # generate rfi
        r = rfi.rfi_impulse(lsts, freqs, impulse_chance=0.5)
        self.assertEqual(r.shape, (lsts.size, freqs.size))
        self.assertAlmostEqual(np.where(np.abs(r) > 0)[0].size / float(r.size), 0.5, -1)
        # import uvtools, pylab as plt
        # uvtools.plot.waterfall(r); plt.show()

    def test_rfi_scatter(self):
        # choose observing parameters
        freqs = np.linspace(0.1, 0.2, 100)
        lsts = np.linspace(0, 2 * np.pi, 200)

        # generate rfi
        r = rfi.rfi_scatter(lsts, freqs, scatter_chance=0.5)
        self.assertEqual(r.shape, (lsts.size, freqs.size))
        self.assertAlmostEqual(np.where(np.abs(r) > 0)[0].size, r.size / 2, -3)
        # import uvtools, pylab as plt
        # uvtools.plot.waterfall(r); plt.show()

    def test_rfi_dtv(self):
        freqs = np.linspace(0.1, 0.2, 100)
        lsts = np.linspace(0, 2 * np.pi, 200)

        # Scalar chance.
        r = rfi.rfi_dtv(
            lsts,
            freqs,
            dtv_band=(0.15, 0.20),
            dtv_channel_width=0.01,
            dtv_chance=0.5,
            dtv_strength=10,
            dtv_std=1,
        )
        self.assertEqual(r.shape, (lsts.size, freqs.size))
        self.assertAlmostEqual(np.where(np.abs(r) > 0)[0].size / float(r.size), 0.5, -1)

        # Ensure that a sub-band is the same across frequencies.
        np.testing.assert_allclose(
            np.mean(r[:, np.logical_and(freqs >= 0.15, freqs < 0.16)], axis=1), r[:, 50]
        )

        np.testing.assert_allclose(
            np.mean(r[:, np.logical_and(freqs >= 0.19, freqs < 0.2)], axis=1), r[:, -1]
        )

        # List of chances.
        r = rfi.rfi_dtv(
            lsts,
            freqs,
            dtv_band=(0.15, 0.20),
            dtv_channel_width=0.01,
            dtv_chance=[0.5, 0.6, 0.7, 0.8, 1.0],
            dtv_strength=10,
            dtv_std=1e-5,
        )

        # Ensure that the correct chance gets applied to each band
        self.assertAlmostEqual(np.sum(np.abs(r[:, -1])), lsts.size * 10, -2)
        self.assertAlmostEqual(np.sum(np.abs(r[:, 0])), 0, -5)


if __name__ == "__main__":
    unittest.main()
