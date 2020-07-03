import os
import unittest
from hera_sim import sigchain, noise, foregrounds
from hera_sim.interpolators import Bandpass, Beam
from hera_sim import DATA_PATH
import uvtools
import numpy as np
import nose.tools as nt
from astropy import units
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
        times = np.linspace(0, 1, 500)  # hours
        dly = 20  # ns
        delays = {0: dly}
        bp_poly = Bandpass(datafile="HERA_H1C_BANDPASS.npy")
        gains = sigchain.gen_gains(
            freqs, ants=delays, gain_spread=0, dly_rng=(dly, dly), bp_poly=bp_poly
        )
        # Throw in a random phase, since gains have zero phase offset.
        gains = {
            ant: gain * np.exp(1j * np.random.uniform(-np.pi, np.pi))
            for ant, gain in gains.items()
        }
        self.delay_phases = (2 * np.pi * freqs * dly) % (2 * np.pi) - np.pi
        self.phases = np.angle(gains[0])
        phase_offsets = (self.phases - self.delay_phases) % (2 * np.pi)
        self.phase_offsets = phase_offsets - np.pi  # Map phase offsets to [-pi, pi)
        self.gains = gains
        self.freqs = freqs
        self.times = times
        self.delays = delays

        # Calculate fringe rates for inspecting the fringe-rate transforms.
        fringe_rates = uvtools.utils.fourier_freqs(self.times * units.h.to("s"))
        pos_fringe_key = np.argwhere(
            fringe_rates > 5 * np.mean(np.diff(fringe_rates))
        ).flatten()
        neg_fringe_key = np.argwhere(
            fringe_rates < -5 * np.mean(np.diff(fringe_rates))
        ).flatten()
        self.fringe_rates = fringe_rates
        self.fringe_keys = {"pos": pos_fringe_key, "neg": neg_fringe_key}

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
        assert np.allclose(
            varied_gain[np.argmin(np.abs(self.times - self.times.mean())), :],
            self.gains[0],
            rtol=0.001,
        )

        # Check that the amount of variation in the amplitudes is as expected.
        assert np.allclose(varied_gain[-1, :] / self.gains[0], 1.1)
        assert np.allclose(varied_gain[0, :] / self.gains[0], 0.9)

        # Now add sinusoidal variation, check that it shows up where it's expected.
        vary_timescale = 30 * units.s.to("hour")  # Fast variations
        vary_freq = 1 / (vary_timescale * units.h.to("s"))  # In case numerical issues
        varied_gains = sigchain.vary_gains_in_time(
            gains=self.gains,
            times=self.times,
            parameter="amp",
            variation_timescales=(vary_timescale,),
            variation_amps=(0.5,),
            variation_modes=("sinusoidal",),
        )

        varied_gain = varied_gains[0]
        varied_gain_fft = uvtools.utils.FFT(varied_gain, axis=0, taper="bh7")
        for fringe_key in self.fringe_keys.values():
            peak_index = np.argmax(np.abs(varied_gain_fft[fringe_key, 150]))
            assert np.isclose(
                vary_freq, np.abs(self.fringe_rates[fringe_key][peak_index]), rtol=0.01
            )

        # Finally, check noiselike variations.
        vary_amp = 0.1
        varied_gains = sigchain.vary_gains_in_time(
            gains=self.gains,
            times=self.times,
            parameter="amp",
            variation_modes="noiselike",
            variation_amps=vary_amp,
        )

        varied_gain = varied_gains[0]
        standard_deviations = np.std(np.abs(varied_gain), axis=0)
        gain_avg = np.mean(np.abs(varied_gain), axis=0)
        assert np.allclose(gain_avg, np.abs(self.gains[0]), rtol=0.05)
        assert np.allclose(
            standard_deviations, vary_amp * np.abs(self.gains[0]), rtol=0.05
        )

    def test_vary_gain_phase(self):
        # Vary phase offsets linearly.
        vary_amp = 0.1  # Vary by 0.1 radians over half the variation time.
        varied_gains = sigchain.vary_gains_in_time(
            gains=self.gains,
            times=self.times,
            parameter="phs",
            variation_modes="linear",
            variation_amps=vary_amp,
            variation_ref_times=self.times.mean(),
            variation_timescales=2 * (self.times[-1] - self.times[0]),
        )

        varied_gain = varied_gains[0]
        varied_phases = np.angle(varied_gain)
        varied_phase_offsets = (varied_phases - self.delay_phases) % (2 * np.pi) - np.pi
        assert np.allclose(
            self.phase_offsets,
            varied_phase_offsets[np.argmin(np.abs(self.times - self.times.mean()))],
            rtol=0.01,
        )
        assert np.allclose(
            varied_phase_offsets[-1, :] - self.phase_offsets, vary_amp, rtol=0.01
        )
        assert np.allclose(
            varied_phase_offsets[0, :] - self.phase_offsets, -vary_amp, rtol=0.01
        )

        # Vary phase offsets sinusoidally.
        timescale = 1 * units.min.to("h")
        vary_freq = 1 / (timescale * units.h.to("s"))
        varied_gains = sigchain.vary_gains_in_time(
            gains=self.gains,
            times=self.times,
            parameter="phs",
            variation_modes="sinusoidal",
            variation_amps=vary_amp,
            variation_timescales=timescale,
        )

        varied_gain = varied_gains[0]
        varied_phases = np.angle(varied_gain)
        varied_phase_offsets = (varied_phases - self.delay_phases) % (2 * np.pi) - np.pi
        varied_phs_offset_fft = uvtools.utils.FFT(
            varied_phase_offsets, axis=0, taper="bh7"
        )

        # Check that there's variation at the expected modes.
        for fringe_key in self.fringe_keys.values():
            peak_index = np.argmax(np.abs(varied_phs_offset_fft[fringe_key, 150]))
            assert np.isclose(
                vary_freq, np.abs(self.fringe_rates[fringe_key][peak_index]), rtol=0.01
            )

        # Finally, check noiselike variation.
        varied_gains = sigchain.vary_gains_in_time(
            gains=self.gains,
            times=self.times,
            parameter="phs",
            variation_modes="noiselike",
            variation_amps=vary_amp,
        )

        varied_gain = varied_gains[0]
        varied_phases = np.angle(varied_gain)
        varied_phase_offsets = (varied_phases - self.delay_phases) % (2 * np.pi) - np.pi
        mean_offset = np.mean(varied_phase_offsets, axis=0)
        offset_std = np.std(varied_phase_offsets, axis=0)
        assert np.allclose(mean_offset, self.phase_offsets, rtol=0.01)
        # In the amplitude test, this tolerance is a bit lower (5%). I think that the
        # complex exponentiation, among the other operations to extract the phase
        # offsets, introduces numerical artifacts that makes this sampled variance
        # a bit larger than the true variance. It may be worthwhile to do a test
        # somewhere that increasing the number of times improves the agreement
        # between the sampled variance and the true variance.
        assert np.allclose(offset_std, vary_amp, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
