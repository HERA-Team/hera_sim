import unittest
from hera_sim import reflections, foregrounds, noise
import numpy as np
import aipy
import nose.tools as nt
from scipy.signal import windows

np.random.seed(0)

class TestReflections(unittest.TestCase):

    def setUp(self):
        # setup simulation parameters
        fqs = np.linspace(.1, .2, 100, endpoint=False)
        lsts = np.linspace(0, 2*np.pi, 200)
        times = lsts / (2*np.pi) * aipy.const.sidereal_day
        Tsky_mdl = noise.HERA_Tsky_mdl['xx']
        Tsky = Tsky_mdl(lsts, fqs)
        bl_len_ns = 50.
        vis = foregrounds.diffuse_foreground(Tsky_mdl, lsts, fqs, bl_len_ns)

        self.freqs = fqs
        self.lsts = lsts
        self.Tsky = Tsky
        self.bl_len_ns = bl_len_ns
        self.vis = vis
        self.vfft = np.fft.fft(vis, axis=1)
        self.dlys = np.fft.fftfreq(len(fqs), d=np.median(np.diff(fqs)))

    def test_auto_reflection(self):
        # introduce a cable reflection into the autocorrelation
        outvis = reflections.auto_reflection(self.Tsky, self.freqs, 1e-1, 300, 1)
        ovfft = np.fft.fft(outvis * windows.blackmanharris(len(self.freqs))[None, :], axis=1)

        # assert reflection is at +300 ns and check its amplitude
        select = self.dlys > 100
        nt.assert_almost_equal(self.dlys[select][np.argmax(np.mean(np.abs(ovfft), axis=0)[select])], 300)
        select = np.argmin(np.abs(self.dlys - 300))
        nt.assert_true(np.mean(np.abs(ovfft), axis=0)[select] > 900)

        # assert no reflection at -300 ns
        select = np.argmin(np.abs(self.dlys - -300))
        nt.assert_true(np.mean(np.abs(ovfft), axis=0)[select] < 1.0)

        # conjugate reflection, and assert it now shows up at -300 ns
        outvis = reflections.auto_reflection(self.Tsky, self.freqs, 1e-1, 300, 1, conj=True)
        ovfft = np.fft.fft(outvis * windows.blackmanharris(len(self.freqs))[None, :], axis=1)

        select = self.dlys < -100
        nt.assert_almost_equal(self.dlys[select][np.argmax(np.mean(np.abs(ovfft), axis=0)[select])], -300)
        select = np.argmin(np.abs(self.dlys - -300))
        nt.assert_true(np.mean(np.abs(ovfft), axis=0)[select] > 900)

    def test_cross_reflection(self):
        # introduce a cross reflection at a single delay
        outvis = reflections.cross_reflection(self.vis, self.freqs, self.Tsky, 1e-2, 300, 0)
        ovfft = np.fft.fft(outvis * windows.blackmanharris(len(self.freqs))[None, :], axis=1)

        # take covariance across time and assert delay 300 is highly covariant
        # compared to neighbors
        cov = np.cov(ovfft.T)
        mcov = np.mean(np.abs(cov), axis=0)
        select = np.argsort(np.abs(self.dlys - 300))[:10]
        nt.assert_almost_equal(self.dlys[select][np.argmax(mcov[select])], 300.0)
        # inspect for yourself: plt.matshow(np.log10(np.abs(cov)))

        # conjugate it and assert it shows up at -300
        outvis = reflections.cross_reflection(self.vis, self.freqs, self.Tsky, 1e-2, 300, 1, conj=True)
        ovfft = np.fft.fft(outvis * windows.blackmanharris(len(self.freqs))[None, :], axis=1)
        cov = np.cov(ovfft.T)
        mcov = np.mean(np.abs(cov), axis=0)
        select = np.argsort(np.abs(self.dlys - -300))[:10]
        nt.assert_almost_equal(self.dlys[select][np.argmax(mcov[select])], -300.0)

        # assert its phase stable across time
        select = np.argmin(np.abs(self.dlys - -300))
        nt.assert_true(np.isclose(np.angle(ovfft[:, select]), -1, atol=1e-4, rtol=1e-4).all())

