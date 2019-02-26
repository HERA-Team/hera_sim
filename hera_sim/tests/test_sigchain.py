import unittest
from hera_sim import sigchain, noise, foregrounds
import numpy as np
import aipy
import nose.tools as nt
from scipy.signal import windows

np.random.seed(0)


class TestSigchain(unittest.TestCase):
    def test_gen_bandpass(self):
        fqs = np.linspace(0.1, 0.2, 1024, endpoint=False)
        g = sigchain.gen_bandpass(fqs, [1, 2], gain_spread=0)
        self.assertTrue(1 in g)
        self.assertTrue(2 in g)
        self.assertEqual(g[1].size, fqs.size)
        np.testing.assert_array_equal(g[1], g[2])
        g = sigchain.gen_bandpass(fqs, list(range(10)), 0.2)
        self.assertFalse(np.all(g[1] == g[2]))

    def test_gen_delay_phs(self):
        fqs = np.linspace(0.12, 0.18, 1024, endpoint=False)
        phs = sigchain.gen_delay_phs(fqs, [1, 2], dly_rng=(0, 20))
        self.assertEqual(len(phs), 2)
        self.assertTrue(1 in phs)
        self.assertTrue(2 in phs)
        np.testing.assert_almost_equal(np.abs(phs[1]), 1.0)
        p = np.polyfit(fqs, np.unwrap(np.angle(phs[1])), deg=1)
        self.assertAlmostEqual(p[-1] % (2 * np.pi), 0.0, -2)
        self.assertLessEqual(p[0], 20 * 2 * np.pi)
        self.assertGreaterEqual(p[0], 0 * 2 * np.pi)

    def test_gen_gains(self):
        fqs = np.linspace(0.12, 0.18, 1024, endpoint=False)
        g = sigchain.gen_gains(fqs, [1, 2], gain_spread=0, dly_rng=(10, 20))
        np.testing.assert_allclose(np.abs(g[1]), np.abs(g[2]), 1e-5)
        for i in g:
            p = np.polyfit(fqs, np.unwrap(np.angle(g[i])), deg=1)
            self.assertAlmostEqual(p[-1] % (2 * np.pi), 0.0, -2)
            self.assertLessEqual(p[0], 20 * 2 * np.pi)
            self.assertGreaterEqual(p[0], 10 * 2 * np.pi)

    def test_apply_gains(self):
        fqs = np.linspace(0.12, 0.18, 1024, endpoint=False)
        vis = np.ones((100, fqs.size), dtype=np.complex)
        g = sigchain.gen_gains(fqs, [1, 2], gain_spread=0, dly_rng=(10, 10))
        gvis = sigchain.apply_gains(vis, g, (1, 2))
        np.testing.assert_allclose(np.angle(gvis), 0, 1e-5)


class TestSigchainReflections(unittest.TestCase):
    def setUp(self):
        # setup simulation parameters
        np.random.seed(0)
        fqs = np.linspace(0.1, 0.2, 100, endpoint=False)
        lsts = np.linspace(0, 2 * np.pi, 200)
        times = lsts / (2 * np.pi) * aipy.const.sidereal_day
        Tsky_mdl = noise.HERA_Tsky_mdl["xx"]
        Tsky = Tsky_mdl(lsts, fqs)
        bl_len_ns = 50.0
        # + 20 is to boost k=0 mode
        vis = foregrounds.diffuse_foreground(Tsky_mdl, lsts, fqs, bl_len_ns) + 20

        self.freqs = fqs
        self.lsts = lsts
        self.Tsky = Tsky
        self.bl_len_ns = bl_len_ns
        self.vis = vis
        self.vfft = np.fft.fft(vis, axis=1)
        self.dlys = np.fft.fftfreq(len(fqs), d=np.median(np.diff(fqs)))

    def test_reflection_gains(self):
        # introduce a cable reflection into the autocorrelation
        gains = sigchain.gen_reflection_gains(self.freqs, [0], amp=[1e-1], dly=[300], phs=[1])
        outvis = sigchain.apply_gains(self.vis, gains, [0, 0])
        ovfft = np.fft.fft(
            outvis * windows.blackmanharris(len(self.freqs))[None, :], axis=1
        )

        # assert reflection is at +300 ns and check its amplitude
        select = self.dlys > 200
        nt.assert_almost_equal(
            self.dlys[select][np.argmax(np.mean(np.abs(ovfft), axis=0)[select])], 300
        )
        select = np.argmin(np.abs(self.dlys - 300))
        m = np.mean(np.abs(ovfft), axis=0)
        nt.assert_true(np.isclose(m[select] / m[0], 1e-1, atol=1e-2))

        # assert also reflection at -300 ns
        select = self.dlys < -200
        nt.assert_almost_equal(
            self.dlys[select][np.argmax(np.mean(np.abs(ovfft), axis=0)[select])], -300
        )
        select = np.argmin(np.abs(self.dlys - -300))
        m = np.mean(np.abs(ovfft), axis=0)
        nt.assert_true(np.isclose(m[select] / m[0], 1e-1, atol=1e-2))

    def test_cross_coupling_xtalk(self):
        # introduce a cross reflection at a single delay
        outvis = sigchain.gen_cross_coupling_xtalk(self.freqs, self.Tsky, amp=1e-2, dly=300, phs=1)
        ovfft = np.fft.fft(
            outvis * windows.blackmanharris(len(self.freqs))[None, :], axis=1
        )

        # take covariance across time and assert delay 300 is highly covariant
        # compared to neighbors
        cov = np.cov(ovfft.T)
        mcov = np.mean(np.abs(cov), axis=0)
        select = np.argsort(np.abs(self.dlys - 300))[:10]
        nt.assert_almost_equal(self.dlys[select][np.argmax(mcov[select])], 300.0)
        # inspect for yourself: plt.matshow(np.log10(np.abs(cov)))

        # conjugate it and assert it shows up at -300
        outvis = sigchain.gen_cross_coupling_xtalk(self.freqs, self.Tsky, amp=1e-2, dly=300, phs=1, conj=True)
        ovfft = np.fft.fft(
            outvis * windows.blackmanharris(len(self.freqs))[None, :], axis=1
        )
        cov = np.cov(ovfft.T)
        mcov = np.mean(np.abs(cov), axis=0)
        select = np.argsort(np.abs(self.dlys - -300))[:10]
        nt.assert_almost_equal(self.dlys[select][np.argmax(mcov[select])], -300.0)

        # assert its phase stable across time
        select = np.argmin(np.abs(self.dlys - -300))
        nt.assert_true(
            np.isclose(np.angle(ovfft[:, select]), -1, atol=1e-4, rtol=1e-4).all()
        )


if __name__ == "__main__":
    unittest.main()
