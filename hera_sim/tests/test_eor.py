import unittest
from hera_sim import eor
import numpy as np
import aipy
import nose.tools as nt

np.random.seed(0)

class TestForegrounds(unittest.TestCase):
    def test_noiselike_eor(self):
        # setup simulation parameters
        fqs = np.linspace(.1, .2, 201, endpoint=False)
        lsts = np.linspace(0, 1, 100)
        times = lsts / (2*np.pi) * aipy.const.sidereal_day
        bl_len_ns = 50.

        # Simulate vanilla eor
        vis = eor.noiselike_eor(lsts, fqs, bl_len_ns, eor_amp=1e-5, spec_tilt=0, min_delay=0, max_delay=1e5)

        # assert covariance across freq is close to diagonal (i.e. frequency covariance is essentially noise-like)
        cov = np.cov(vis.T)
        mean_diag = np.mean(cov.diagonal())
        mean_offdiag = np.mean(cov - np.eye(len(cov)) * cov.diagonal())
        nt.assert_true(np.abs(mean_diag / mean_offdiag) > 1e3)
        # Look at it manually to check: plt.matshow(np.abs(cov))

        # assert covariance across time has some non-neglible off diagonal (i.e. sky locked)
        cov = np.cov(vis)
        mean_diag = np.mean(cov.diagonal())
        mean_offdiag = np.mean(cov - np.eye(len(cov)) * cov.diagonal())
        nt.assert_true(np.abs(mean_diag / mean_offdiag) < 20)
        # Look at it manually to check: plt.matshow(np.abs(cov))

        # Introduce a spectral tilt: generally EoR is flat-ish in Delta^2 and therefore
        # follows a negative power-law in P(k)
        lsts = np.linspace(0, 2*np.pi, 1000)
        vis = eor.noiselike_eor(lsts, fqs, bl_len_ns, eor_amp=1e-5, spec_tilt=-2, min_delay=200, max_delay=500)

        # take FFT and incoherently average over time to check spectral tilt
        vfft = np.mean(np.abs(np.fft.fft(vis, axis=1)), axis=0)
        delays = np.fft.fftfreq(len(fqs), d=np.median(np.diff(fqs)))
        select = (delays > 200) & (delays < 500)
        fit = np.polyfit(np.log10(delays[select]), np.log10(vfft[select]), 1)
        nt.assert_almost_equal(fit[0], -2, places=1)

        # test amplitude scaling is correct
        vis1 = eor.noiselike_eor(lsts, fqs, bl_len_ns, eor_amp=1e-5, spec_tilt=0, min_delay=200, max_delay=500)
        vis2 = eor.noiselike_eor(lsts, fqs, bl_len_ns, eor_amp=1e-3, spec_tilt=0, min_delay=200, max_delay=500)
        nt.assert_almost_equal(np.mean(np.abs(vis1 / vis2)) / np.sqrt(2), 1e-2, places=2)


if __name__ == '__main__':
    unittest.main()
