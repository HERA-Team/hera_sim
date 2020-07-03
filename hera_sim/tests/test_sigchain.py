import os
import unittest
from hera_sim import sigchain, noise, foregrounds
from hera_sim.interpolators import Bandpass, Beam
from hera_sim import DATA_PATH
import numpy as np
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

        # extra parameters needed
        Tsky_mdl = noise.HERA_Tsky_mdl["xx"]
        Tsky = Tsky_mdl(lsts, fqs)
        bl_vec = np.array([50.0, 0, 0])
        beamfile = DATA_PATH / "HERA_H1C_BEAM_POLY.npy"
        omega_p = Beam(beamfile)

        # mock up visibilities
        vis = foregrounds.diffuse_foreground(
            lsts,
            fqs,
            bl_vec,
            Tsky_mdl=Tsky_mdl,
            omega_p=omega_p,
            delay_filter_kwargs={"delay_filter_type": "gauss"},
        )

        # add a constant offset to boost k=0 mode
        vis += 20

        self.freqs = fqs
        self.lsts = lsts
        self.Tsky = Tsky
        self.bl_vec = bl_vec
        self.vis = vis
        self.vfft = np.fft.fft(vis, axis=1)
        self.dlys = np.fft.fftfreq(len(fqs), d=np.median(np.diff(fqs)))

    def test_reflection_gains(self):
        # introduce a cable reflection into the autocorrelation
        gains = sigchain.gen_reflection_gains(
            self.freqs, [0], amp=[1e-1], dly=[300], phs=[1]
        )
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

        # test reshaping into Ntimes
        amp = np.linspace(1e-2, 1e-3, 3)
        gains = sigchain.gen_reflection_gains(
            self.freqs, [0], amp=[amp], dly=[300], phs=[1]
        )
        nt.assert_equal(gains[0].shape, (3, 100))

        # test frequency evolution with one time
        amp = np.linspace(1e-2, 1e-3, 100).reshape(1, -1)
        gains = sigchain.gen_reflection_gains(
            self.freqs, [0], amp=[amp], dly=[300], phs=[1]
        )
        nt.assert_equal(gains[0].shape, (1, 100))
        # now test with multiple times
        amp = np.repeat(np.linspace(1e-2, 1e-3, 100).reshape(1, -1), 10, axis=0)
        gains = sigchain.gen_reflection_gains(
            self.freqs, [0], amp=[amp], dly=[300], phs=[1]
        )
        nt.assert_equal(gains[0].shape, (10, 100))

        # exception
        amp = np.linspace(1e-2, 1e-3, 2).reshape(1, -1)
        nt.assert_raises(
            AssertionError,
            sigchain.gen_reflection_gains,
            self.freqs,
            [0],
            amp=[amp],
            dly=[300],
            phs=[1],
        )

    def test_cross_coupling_xtalk(self):
        # introduce a cross reflection at a single delay
        outvis = sigchain.gen_cross_coupling_xtalk(
            self.freqs, self.Tsky, amp=1e-2, dly=300, phs=1
        )
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
        outvis = sigchain.gen_cross_coupling_xtalk(
            self.freqs, self.Tsky, amp=1e-2, dly=300, phs=1, conj=True
        )
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

class TestTimeVariation(unittest.TestCase):
    def setUp(self):
        # Mock up some gains.
        freqs = np.linspace(0.1, 0.2, 1024)
        times = np.linspace(0, 1, 500)
        dly = 20  # ns
        delays = {0: dly}
        bp_poly = Bandpass(datafile="HERA_H1C_BANDPASS.npy")
        gains = sigchain.gen_gains(
            freqs, ants=delays, dly_rng=(dly, dly), bp_poly=bp_poly
        )
        self.gains = gains
        self.freqs = freqs
        self.times = times
        self.delays = delays

    def test_vary_gain_amp(self):
        varied_gains = sigchain.vary_gains_in_time(
            gains=self.gains,
            times=self.times,
            parameter="amp",
            variation_ref_times=(self.times.mean(),),
            variation_timescales=(2 * (self.times[-1] - self.times[0]),),
            variation_amps=(0.1,),
            variation_modes=("linear",),
        )

        # Check that the gains are their original value at the center time.
        varied_gain = varied_gains[0]
        original_gain = self.gains[0]
        assert np.allclose(
            varied_gain[np.argmin(np.abs(self.times - self.times.mean())),:],
            original_gain,
            rtol=0.001,
        )

if __name__ == "__main__":
    unittest.main()
