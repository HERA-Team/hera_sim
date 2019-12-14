import os
import unittest
from hera_sim import noise, utils
from hera_sim.data import DATA_PATH
from hera_sim.interpolators import Beam
import numpy as np
import aipy

np.random.seed(0)

beamfile = os.path.join(DATA_PATH, "HERA_H1C_BEAM_POLY.npy")

class TestNoise(unittest.TestCase):
    pass # until I figure out how to get things working
    # move this to utils test
#    def test_white_noise(self):
#        n1 = noise.white_noise(100)
#        self.assertEqual(n1.size, 100)
#        self.assertEqual(n1.shape, (100,))
#        n2 = noise.white_noise((100, 100))
#        self.assertEqual(n2.shape, (100, 100))
#        n3 = noise.white_noise(100000)
#        self.assertAlmostEqual(np.average(n3), 0, 1)
#        self.assertAlmostEqual(np.std(n3), 1, 2)

    # move this to interpolators test
#    def test_resample_Tsky(self):
#        fqs = np.linspace(0.1, 0.2, 100)
#        lsts = np.linspace(0, 2 * np.pi, 200)
#        tsky = noise.resample_Tsky(fqs, lsts)
#        self.assertEqual(tsky.shape, (200, 100))
#        self.assertTrue(np.all(tsky[0] == tsky[1]))
#        self.assertFalse(np.all(tsky[:, 0] == tsky[:, 1]))
#        # import uvtools, pylab as plt
#        # uvtools.plot.waterfall(tsky); plt.show()
#        tsky = noise.resample_Tsky(fqs, lsts, Tsky_mdl=noise.HERA_Tsky_mdl["xx"])
#        self.assertEqual(tsky.shape, (200, 100))
#        self.assertFalse(np.all(tsky[0] == tsky[1]))
#        self.assertFalse(np.all(tsky[:, 0] == tsky[:, 1]))
#        # import uvtools, pylab as plt
#        # uvtools.plot.waterfall(tsky); plt.show()

    # adapt this to test thermal noise

    # note that this test currently fails when checking that the
    # dimensionless standard deviation of the noise in time is 
    # close to 1 for all frequencies (this fails in every freq channel)
#    def test_sky_noise_jy(self):
#        # make some parameters
#        freqs = np.linspace(0.1, 0.2, 100)
#        lsts = np.linspace(0, 2 * np.pi, 500)
#        omega_p = Beam(beamfile)
#        Tsky_mdl = noise.HERA_Tsky_mdl['xx']
#        B = 1e6 # channel width in Hz
#
#        # resample the sky temperature model
#        Tsky = Tsky_mdl(lsts, freqs)
#
#        # get the conversion from Jy -> K
#        Jy2T = utils.Jy2T(freqs, omega_p)
#        Jy2T.shape = (1, -1)
#
#        # simulate the noise
#        nos_jy = noise.sky_noise_jy(
#            lsts, freqs, Tsky_mdl=Tsky_mdl, 
#            omega_p=omega_p, integration_time=10.7
#        )
#
#        # check the shape
#        self.assertEqual(nos_jy.shape, (lsts.size, freqs.size))
#
#        # check that it's actually noiselike
#        # why is the tolerance set to what it is?
#        np.testing.assert_allclose(np.average(nos_jy, axis=0), 0, atol=0.7)
#        
#        # figure out a better comment for this part
#        scaling = np.average(Tsky, axis=0) / Jy2T
#        dt = 10.7
#
#        np.testing.assert_allclose(
#            np.std(nos_jy, axis=0) / scaling * np.sqrt(B * dt), 1.0, atol=0.1
#        )
#
#        # now do it again but with the integration time set by lsts
#        np.random.seed(0)
#        nos_jy = noise.sky_noise_jy(
#            lsts, freqs, Tsky_mdl=Tsky_mdl, 
#            omega_p=omega_p, integration_time=None
#        )
#        
#        dt = u.sday.to('s') / lsts.size
#        np.testing.assert_allclose(
#            np.std(nos_jy, axis=0) / scaling * np.sqrt(B * dt), 1.0, atol=0.1
#        )
#
#        # once more with a manually set channel width
#        np.random.seed(0)
#        B = 1e8
#        dt = 10.7
#        nos_jy = noise.sky_noise_jy(
#            lsts, freqs, Tsky_mdl=Tsky_mdl,  
#            channel_width=B, integration_time=dt, 
#            omega_p=omega_p
#        )
#        np.testing.assert_allclose(
#            np.std(nos_jy, axis=0) / scaling * np.sqrt(B * dt), 1.0, atol=0.1
#        )
#
#        # tsky = noise.resample_Tsky(fqs,lsts,noise.HERA_Tsky_mdl['xx'])
#        # nos_jy = noise.sky_noise_jy(tsky, fqs, lsts)
#        # import uvtools, pylab as plt
#        # uvtools.plot.waterfall(nos_jy, mode='real'); plt.show()


if __name__ == "__main__":
    unittest.main()
