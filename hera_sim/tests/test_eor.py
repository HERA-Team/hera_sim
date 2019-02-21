import unittest
from hera_sim import eor
import numpy as np
import aipy
import nose.tools as nt

np.random.seed(0)

class TestEoR(unittest.TestCase):
    def test_noiselike_eor(self):
        # setup simulation parameters
        fqs = np.linspace(.1, .2, 201, endpoint=False)
        lsts = np.linspace(0, 1, 500)
        times = lsts / (2*np.pi) * aipy.const.sidereal_day
        bl_len_ns = 50.

        # Simulate vanilla eor
        vis, ff = eor.noiselike_eor(lsts, fqs, bl_len_ns, eor_amp=1e-5, fringe_filter_type='tophat')

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

        # test amplitude scaling is correct
        vis1, ff = eor.noiselike_eor(lsts, fqs, bl_len_ns, eor_amp=1e-5, spec_tilt=0, min_delay=0, max_delay=1e5, fringe_filter_type='tophat')
        vis2, ff = eor.noiselike_eor(lsts, fqs, bl_len_ns, eor_amp=1e-3, spec_tilt=0, min_delay=0, max_delay=1e5, fringe_filter_type='tophat')
        nt.assert_almost_equal(np.mean(np.abs(vis1 / vis2)) / np.sqrt(2), 1e-2, places=2)


if __name__ == "__main__":
    unittest.main()
