import os
import unittest
import pytest
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

@pytest.fixture(scope="function")
def freqs():
    return np.linspace(0.1, 0.2, 1024)
    
@pytest.fixture(scope="function")
def times():
    return np.linspace(0, 1, 500)

@pytest.fixture(scope="function")
def delays():
    return {0: 20}  # ns

@pytest.fixture(scope="function")
def bp_poly():
    return Bandpass(datafile="HERA_H1C_BANDPASS.npy")

@pytest.fixture(scope="function")
def gains(freqs, delays, bp_poly):
    dly = delays[0]
    gain_dict = sigchain.gen_gains(
        freqs, ants=delays, gain_spread=0, dly_rng=(dly, dly), bp_poly=bp_poly
    )
    # Add a phase offset to the gains
    gain_dict[0] *= np.exp(1j * np.pi / 4)
    return gain_dict

@pytest.fixture(scope="function")
def delay_phases(freqs, delays):
    dly = delays[0]
    return (2 * np.pi * freqs * dly) % (2 * np.pi) - np.pi

@pytest.fixture(scope="function")
def phase_offsets(gains, delay_phases):
    phases = np.angle(gains[0])
    return (phases - delay_phases) % (2 * np.pi) - np.pi

@pytest.fixture(scope="function")
def fringe_rates(times):
    return uvtools.utils.fourier_freqs(times * units.h.to("s"))

@pytest.fixture(scope="function")
def fringe_keys(fringe_rates):
    pos_fringe_key = np.argwhere(
        fringe_rates > 5 * np.mean(np.diff(fringe_rates))
    ).flatten()
    neg_fringe_key = np.argwhere(
        fringe_rates < -5 * np.mean(np.diff(fringe_rates))
    ).flatten()
    return (pos_fringe_key, neg_fringe_key)

def varies_as_expected(quantity, vary_freq, fringe_key, fringe_rates):
    quantity_fft = uvtools.utils.FFT(quantity, axis=0, taper="bh7")
    peak_index = np.argmax(np.abs(quantity_fft[fringe_key, 150]))
    return np.isclose(vary_freq, np.abs(fringe_rates[fringe_key][peak_index]), rtol=0.01)

def test_vary_gain_amp_linear(gains, times):
    varied_gains = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        parameter="amp",
        variation_ref_time=times.mean(),
        variation_timescale=2 * (times[-1] - times[0]),
        variation_amp=0.1,
        variation_mode="linear",
    )[0]

    # Check that the original value is at the center time.
    assert np.allclose(
        varied_gain[np.argmin(np.abs(times - times.mean())), :],
        gains[0],
        rtol=0.001,
    )

    # Check that the variation amount is as expected.
    assert np.allclose(varied_gains[-1, :] / gains[0], 1.1)
    assert np.allclose(varied_gains[0, :] / gains[0], 0.9)

def test_vary_gain_amp_sinusoidal(gains, times, fringe_rates, fringe_keys):
    vary_timescale = 30 * units.s.to("hour")
    vary_freq = 1 / (vary_timescale * units.h.to("s"))
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        parameter="amp",
        variation_timescale=vary_timescale,
        variation_amp=0.5,
        variation_mode="sinusoidal",
    )[0]

    # Check that there's variation at the expected timescale.
    for fringe_key in fringe_keys:
        assert varies_as_expected(varied_gain, vary_freq, fringe_key, fringe_rates)

def test_vary_gain_amp_noiselike(gains, times):
    vary_amp = 0.1
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        parameter="amp",
        variation_mode="noiselike",
        variation_amp=vary_amp,
    )[0]

    # Check that the mean gain amplitude is the original gain amplitude.
    gain_avg = np.mean(np.abs(varied_gain), axis=0)
    assert np.allclose(gain_avg, np.abs(gains[0]), rtol=0.05)

    # Check that the spread in gain amplitudes is as expected.
    standard_deviations = np.std(np.abs(varied_gain), axis=0)
    assert np.allclose(standard_deviations, vary_amp * np.abs(gains[0]), rtol=0.05)

def test_vary_gain_phase_linear(gains, times, phase_offsets, delay_phases):
    vary_amp = 0.1
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        parameter="phs",
        variation_mode="linear",
        variation_amp=vary_amp,
        variation_ref_time=times.mean(),
        variation_timescale=2 * (times[-1] - times[0]),
    )[0]

    varied_phases = np.angle(varied_gain)
    varied_phase_offsets = (varied_phases - delay_phases) % (2 * np.pi) - np.pi

    # Check that the original phase offset occurs at the central time.
    assert np.allclose(
        phase_offsets,
        varied_phase_offsets[np.argmin(np.abs(times - times.mean()))],
        rtol=0.01,
    )

    # Check that the variation amount is as expected.
    assert np.allclose(varied_phase_offsets[-1] - phase_offsets, vary_amp, rtol=0.01)
    assert np.allclose(varied_phase_offsets[0] - phase_offsets, -vary_amp, rtol=0.01)

def test_vary_gain_phase_sinusoidal(gains, times, delay_phases, fringe_rates, fringe_keys):
    timescale = 1 * units.min.to("h")
    vary_freq = 1 / (timescale * units.h.to("s"))
    vary_amp = 0.1
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        parameter="phs",
        variation_mode="sinusoidal",
        variation_amp=vary_amp,
        variation_timescale=timescale,
    )[0]

    varied_phases = np.angle(varied_gain)
    varied_phase_offsets = (varied_phases - delay_phases) % (2 * np.pi) - np.pi

    # Check that there's variation at the expected timescale.
    for fringe_key in fringe_keys:
        assert varies_as_expected(
            varied_phase_offsets, vary_freq, fringe_key, fringe_rates
        )

def test_vary_gain_phase_noiselike(gains, times, delay_phases, phase_offsets):
    vary_amp = 0.1
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        parameter="phs",
        variation_mode="noiselike",
        variation_amp=vary_amp,
    )[0]

    varied_phases = np.angle(varied_gain)
    varied_phase_offsets = (varied_phases - delay_phases) % (2 * np.pi) - np.pi

    # Check that the mean phase offset is close to the original phase offset.
    mean_offset = np.mean(varied_phase_offsets, axis=0)
    assert np.allclose(mean_offset, phase_offsets, rtol=0.05)

    # Check that the spread in phase offsets is as expected.
    offset_std = np.std(varied_phase_offsets, axis=0)
    assert np.allclose(offset_std, vary_amp, rtol=0.1)

def test_vary_gain_delay_linear(gains, times, freqs, delays):
    vary_amp = 0.5
    dly = delays[0]
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        freqs=freqs,
        delays=delays,
        parameter="dly",
        variation_amp=vary_amp,
        variation_mode="linear",
    )[0]

    varied_gain_fft = uvtools.utils.FFT(varied_gain, axis=1, taper="bh7")
    dlys = uvtools.utils.fourier_freqs(freqs)
    min_dly = dlys[np.argmax(np.abs(varied_gain_fft[0, :]))]
    max_dly = dlys[np.argmax(np.abs(varied_gain_fft[-1, :]))]
    center_index = np.argmin(np.abs(times - times.mean()))
    mid_dly = delays[np.argmax(np.abs(varied_gain_fft[center_index, :]))]
    
    # Check that the delays vary as expected.
    assert np.isclose(min_dly, (1 - vary_amp) * dly, rtol=0.01)
    assert np.isclose(mid_dly, dly, rtol=0.01)
    assert np.isclose(max_dly, (1 + vary_amp) * dly, rtol=0.01)

def test_vary_gain_delay_sinusoidal(
    gains, times, freqs, delays, fringe_rates, fringe_keys
):
    vary_amp = 0.5
    timescale = 30 * units.s.to("h")
    vary_freq = 1 / (timescale * units.h.to("s"))
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        freqs=freqs,
        delays=delays,
        parameter="dly",
        variation_amp=vary_amp,
        variation_mode="sinusoidal",
        variation_timescale=timescale,
    )[0]

    # Determine the bandpass delay at each time.
    varied_gain_fft = uvtools.utils.FFT(varied_gain, axis=1, taper="bh7")
    dlys = uvtools.utils.fourier_freqs(freqs)
    gain_delays = np.array(
        [dlys[np.argmax(np.abs(gain))] for gain in varied_gain_fft]
    )

    # Check that delays vary at the expected timescale.
    for fringe_key in fringe_keys:
        assert varies_as_expected(gain_delays, vary_freq, fringe_key, fringe_rates)

def test_vary_gain_delay_noiselike(gains, times, freqs, delays):
    vary_amp = 0.5
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        freqs=freqs,
        delays=delays,
        parameter="dly",
        variation_amp=vary_amp,
        variation_mode="noiselike",
    )[0]

    # Determine the bandpass delay at each time.
    dlys = uvtools.utils.fourier_freqs(freqs)
    varied_gain_fft = uvtools.utils.FFT(varied_gain, axis=1, taper="bh7")
    gain_delays = np.array(
        [dlys[np.argmax(np.abs(gain))] for gain in varied_gain_fft]
    )

    # Check that the delays vary as expected.
    assert np.isclose(gain_delays.mean(), delays[0], rtol=0.05)
    assert np.isclose(gain_delays.std(), vary_amp * delays[0], rtol=0.1)

def test_vary_gains_exception_bad_times():
    with pytest.raises(TypeError) as err:
        sigchain.vary_gains_in_time(gains={}, times=42)
    assert err.value.args[0] == "times must be an array of real numbers."

def test_vary_gains_exception_complex_times():
    with pytest.raises(TypeError) as err:
        sigchain.vary_gains_in_time(gains={}, times=np.ones(10, dtype=np.complex))
    assert err.value.args[0] == "times must be an array of real numbers."

def test_vary_gains_exception_bad_gains():
    with pytest.raises(TypeError) as err:
        sigchain.vary_gains_in_time(gains=[], times=[1, 2, 3])
    assert err.value.args[0] == "gains must be provided as a dictionary."

def test_vary_gains_exception_bad_param():
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(gains={}, times=[1, 2], parameter="bad choice")
    assert "parameter must be one of" in err.value.args[0]

def test_vary_gains_exception_bad_gain_shapes():
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains={0: np.ones(10), 1: np.ones(15)}, times=[1, 2, 3]
        )
    assert err.value.args[0] == "Gains must all have the same shape."

def test_vary_gains_exception_insufficient_delay_info():
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains=gains, times=times, parameter="dly", freqs=freqs
        )
    assert "you must provide both" in err.value.args[0]

def test_vary_gains_exception_insufficient_freq_info():
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains=gains, times=times, parameter="dly", delays=delays
        )
    assert "you must provide both" in err.value.args[0]

def test_vary_gains_exception_mismatched_keys(times, freqs):
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains={0: 0, 1: 1},
            times=times,
            parameter="dly",
            freqs=freqs,
            delays={1: 1, 2: 2},
        )
    assert err.value.args[0] == "Delays and gains must have the same keys."

def test_vary_gains_exception_bad_gain_waterfall_shape(times, freqs):
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains={0: np.ones((10, 15))},
            times=times,
            freqs=freqs,
            delays={0: 20},
            parameter="dly",
        )
    assert err.value.args[0] == "Gain waterfalls must have shape (Ntimes, Nfreqs)."

def test_vary_gains_exception_bad_gain_spectrum_shape(times, freqs):
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains={0: np.ones(37)},
            times=times,
            freqs=freqs,
            delays={0: 20},
            parameter="dly",
        )
    assert "Gain spectra must be" in err.value.args[0]

def test_vary_gains_exception_too_many_dimensions(times, freqs, delays):
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains={0: np.ones((10, 10, 10))},
            times=times,
            freqs=freqs,
            delays=delays,
            parameter="dly",
        )
    assert "must be at most 2-dimensional." in err.value.args[0]

def test_vary_gains_exception_not_enough_parameters(gains, times):
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains=gains,
            times=times,
            parameter="amp",
            variation_mode=("linear", "sinusoidal"),
            variation_amp=1,
        )
    assert "does not have the same number of entries" in err.value.args[0]

def test_vary_gains_exception_bad_variation_mode(gains, times):
    with pytest.raises(NotImplementedError) as err:
        sigchain.vary_gains_in_time(
            gains=gains,
            times=times,
            parameter="amp",
            variation_mode="foobar",
        )
    assert err.value.args[0] == "Variation mode 'foobar' not supported."


if __name__ == "__main__":
    unittest.main()
