import os
import unittest
from hera_sim import noise, utils
from hera_sim import DATA_PATH
from hera_sim.interpolators import Beam
import numpy as np
import astropy.units as u
from hera_sim.defaults import defaults

np.random.seed(0)

beamfile = DATA_PATH / "HERA_H1C_BEAM_POLY.npy"


class TestNoise(unittest.TestCase):
    def test_resample_Tsky(self):
        # make the necessary parameters
        freqs = np.linspace(0.1, 0.2, 100)
        lsts = np.linspace(0, 2 * np.pi, 200)

        # make a power-law sky temperature
        tsky = noise.resample_Tsky(lsts, freqs)

        # check that the shape is ok
        self.assertEqual(tsky.shape, (lsts.size, freqs.size))

        # check that it's constant in time
        self.assertTrue(np.all(tsky[0] == tsky[1]))

        # check that it varies in frequency
        self.assertFalse(np.all(tsky[:, 0] == tsky[:, 1]))

        # now test it using an interpolation object as a sky model
        tsky = noise.resample_Tsky(lsts, freqs, Tsky_mdl=noise.HERA_Tsky_mdl["xx"])

        # check that the shape is ok
        self.assertEqual(tsky.shape, (lsts.size, freqs.size))

        # now check that it's constant in neither time nor frequency
        self.assertFalse(np.all(tsky[0] == tsky[1]))
        self.assertFalse(np.all(tsky[:, 0] == tsky[:, 1]))

    def test_sky_noise_jy(self):
        defaults.deactivate()

        # make some parameters
        freqs = np.linspace(0.1, 0.2, 100)
        lsts = np.linspace(0, 2 * np.pi, 500)
        omega_p = Beam(beamfile)(freqs)
        Tsky_mdl = None
        B = 1e6  # channel width in Hz

        # resample the sky temperature model
        Tsky = noise.resample_Tsky(lsts, freqs, Tsky_mdl=Tsky_mdl)

        # get the conversion from Jy -> K
        Jy2T = utils.jansky_to_kelvin(freqs, omega_p)
        Jy2T.shape = (1, -1)

        # simulate the noise
        np.random.seed(0)
        nos_jy = noise.sky_noise_jy(
            lsts, freqs, Tsky_mdl=Tsky_mdl, omega_p=omega_p, integration_time=10.7
        )

        # check the shape
        self.assertEqual(nos_jy.shape, (lsts.size, freqs.size))

        # check that it's actually noiselike
        # why is the tolerance set to what it is?
        np.testing.assert_allclose(np.average(nos_jy, axis=0), 0, atol=0.7)

        # figure out a better comment for this part
        scaling = np.average(Tsky, axis=0) / Jy2T
        dt = 10.7

        # XXX this test works just fine, but it breaks when testing the
        # entire repo... can't figure out why
        np.testing.assert_allclose(
            np.std(nos_jy, axis=0) / scaling * np.sqrt(B * dt), 1.0, atol=0.1
        )

        # now do it again but with the integration time set by lsts
        np.random.seed(0)
        nos_jy = noise.sky_noise_jy(
            lsts, freqs, Tsky_mdl=Tsky_mdl, omega_p=omega_p, integration_time=None
        )

        dt = u.sday.to("s") / lsts.size
        np.testing.assert_allclose(
            np.std(nos_jy, axis=0) / scaling * np.sqrt(B * dt), 1.0, atol=0.1
        )

        # once more with a manually set channel width
        np.random.seed(0)
        B = 1e8
        dt = 10.7
        nos_jy = noise.sky_noise_jy(
            lsts,
            freqs,
            Tsky_mdl=Tsky_mdl,
            channel_width=B,
            integration_time=dt,
            omega_p=omega_p,
        )
        np.testing.assert_allclose(
            np.std(nos_jy, axis=0) / scaling * np.sqrt(B * dt), 1.0, atol=0.1
        )

        # tsky = noise.resample_Tsky(freqs,lsts,noise.HERA_Tsky_mdl['xx'])
        # nos_jy = noise.sky_noise_jy(tsky, freqs, lsts)
        # import uvtools, pylab as plt
        # uvtools.plot.waterfall(nos_jy, mode='real'); plt.show()


if __name__ == "__main__":
    unittest.main()
