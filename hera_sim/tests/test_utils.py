import unittest
from hera_sim import utils, noise
from hera_sim.data import DATA_PATH
import numpy as np
import aipy
import nose.tools as nt
import os

np.random.seed(0)

class TestUtils(unittest.TestCase):
    def test_gen_delay_filter(self):
        np.random.seed(0)
        fqs = np.linspace(.1, .2, 100, endpoint=False)
        bl_len_ns = 50.0
        standoff = 0.0

        df = utils.gen_delay_filter(fqs, bl_len_ns, standoff=standoff, filter_type='tophat')
        nt.assert_almost_equal(np.sum(df), 11)

        df = utils.gen_delay_filter(fqs, bl_len_ns, standoff=standoff, filter_type='gauss')
        nt.assert_almost_equal(np.sum(df), 3.133285343289006)

        df = utils.gen_delay_filter(fqs, bl_len_ns, standoff=standoff, filter_type='trunc_gauss')
        nt.assert_almost_equal(np.sum(df), 3.1332651717678575)

        df = utils.gen_delay_filter(fqs, bl_len_ns, standoff=standoff, filter_type='none')
        nt.assert_almost_equal(np.sum(df), 100)

        df = utils.gen_delay_filter(fqs, bl_len_ns, standoff=standoff, filter_type='tophat',
                                    min_delay=100.0)
        nt.assert_almost_equal(np.sum(df), 0)

        df = utils.gen_delay_filter(fqs, bl_len_ns, standoff=standoff, filter_type='tophat',
                                    max_delay=50.0)
        nt.assert_almost_equal(np.sum(df), 11)


    def test_rough_delay_filter(self):
        np.random.seed(0)
        lsts = np.linspace(0, 2*np.pi, 200)
        fqs = np.linspace(.1, .2, 100, endpoint=False)
        bl_len_ns = 50.0
        standoff = 0.0

        data = utils.gen_white_noise((len(lsts), len(fqs)))
        dfilt = utils.rough_delay_filter(
            data, fqs, bl_len_ns, standoff=standoff, 
            delay_filter_type='gauss'
        )
        dfft = np.mean(np.abs(np.fft.ifft(dfilt, axis=1)), axis=0)
        nt.assert_true(np.isclose(dfft[20:-20], 0.0).all())

    def test_gen_fringe_filter(self):
        np.random.seed(0)
        lsts = np.linspace(0, 2*np.pi, 200)
        fqs = np.linspace(.1, .2, 100, endpoint=False)
        bl_len_ns = 50.0
        FRF = np.load(os.path.join(DATA_PATH, "H37_FR_Filters_small.npz"))
        fr_filt = FRF['PB_rms'][0].T
        fr_frates = FRF['frates']
        fr_freqs = FRF['freqs'] / 1e9

        ff = utils.gen_fringe_filter(lsts, fqs, bl_len_ns, filter_type='none')
        nt.assert_true(np.isclose(ff, 1.0).all())

        ff = utils.gen_fringe_filter(lsts, fqs, bl_len_ns, filter_type='tophat')
        nt.assert_almost_equal(np.sum(ff[50]), np.sum(ff[-50]), 41)

        # for some reason this fails, but no changes have been made
        ff = utils.gen_fringe_filter(lsts, fqs, bl_len_ns, filter_type='gauss', fr_width=1e-4)
        nt.assert_almost_equal(np.sum(ff[50]), 63.06179083841268)

        ff = utils.gen_fringe_filter(
            lsts, fqs, bl_len_ns, filter_type='custom', 
            FR_filter=fr_filt, FR_frates=fr_frates, FR_freqs=fr_freqs
        )
        nt.assert_almost_equal(np.sum(ff[50]), 14.66591593210259, places=3)

    def test_rough_fringe_filter(self):
        np.random.seed(0)
        lsts = np.linspace(0, 2*np.pi, 400)
        fqs = np.linspace(.1, .2, 100, endpoint=False)
        bl_len_ns = 50.0
        FRF = np.load(os.path.join(DATA_PATH, "H37_FR_Filters_small.npz"))
        fr_filt = FRF['PB_rms'][0].T
        fr_frates = FRF['frates']
        fr_freqs = FRF['freqs'] / 1e9

        data = utils.gen_white_noise((len(lsts), len(fqs)))
        dfilt = utils.rough_fringe_filter(
            data, lsts, fqs, bl_len_ns, 
            fringe_filter_type='gauss', fr_width=1e-4
        )
        dfft = np.mean(np.abs(np.fft.ifft(dfilt, axis=0)), axis=1)
        nt.assert_true(np.isclose(dfft[50:150], 0.0).all())

    def test_gen_white_noise(self):
        # this test is just ported from the old noise testing module
        n1 = utils.gen_white_noise(100)
        self.assertEqual(n1.size, 100)
        self.assertEqual(n1.shape, (100,))
        n2 = utils.gen_white_noise((100, 100))
        self.assertEqual(n2.shape, (100, 100))
        n3 = utils.gen_white_noise(100000)
        self.assertAlmostEqual(np.average(n3), 0, 1)
        self.assertAlmostEqual(np.std(n3), 1, 2)


def test_bl_vec():
    bl = 1

    assert len(utils._get_bl_len_vec(bl)) == 3

    bl = (0, 1)

    assert len(utils._get_bl_len_vec(bl)) == 3

    bl = [0, 1]

    assert len(utils._get_bl_len_vec(bl)) == 3

    bl = np.array([0, 1, 2])

    assert len(utils._get_bl_len_vec(bl)) == 3


def test_delay_filter_norm():
    N = 50
    fqs = np.linspace(0.1, 0.2, N)

    tsky = np.ones(N)

    np.random.seed(1234) # set the seed for reproducibility.

    out = 0
    nreal = 5000
    for i in range(nreal):
        _noise = tsky * utils.gen_white_noise(N)
        outnoise = utils.rough_delay_filter(_noise, fqs, 30, normalize=1)

        out += np.sum(np.abs(outnoise)**2)

    out /= nreal

    print(out, np.sum(tsky**2))
    assert np.isclose(out, np.sum(tsky**2), atol=0, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
