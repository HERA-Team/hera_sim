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
        nt.assert_almost_equal(np.sum(df), 6.266570686577507)

        df = utils.gen_delay_filter(fqs, bl_len_ns, standoff=standoff, filter_type='trunc_gauss')
        nt.assert_almost_equal(np.sum(df), 6.09878044622021)

        df = utils.gen_delay_filter(fqs, bl_len_ns, standoff=standoff, filter_type='none')
        nt.assert_almost_equal(np.sum(df), 100)

    def test_rough_delay_filter(self):
        np.random.seed(0)
        lsts = np.linspace(0, 2*np.pi, 200)
        fqs = np.linspace(.1, .2, 100, endpoint=False)
        bl_len_ns = 50.0
        standoff = 0.0

        data = noise.white_noise((len(lsts), len(fqs)))
        dfilt, df = utils.rough_delay_filter(data, fqs, bl_len_ns, standoff=standoff, filter_type='gauss')
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

        ff = utils.gen_fringe_filter(lsts, fqs, bl_len_ns, filter_type='gauss', fr_width=1e-4)
        nt.assert_almost_equal(np.sum(ff[50]), 63.06179083841268)

        ff = utils.gen_fringe_filter(lsts, fqs, bl_len_ns, filter_type='custom', FR_filter=fr_filt, FR_frates=fr_frates, FR_freqs=fr_freqs)
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

        data = noise.white_noise((len(lsts), len(fqs)))
        dfilt, ff = utils.rough_fringe_filter(data, lsts, fqs, bl_len_ns, filter_type='gauss', fr_width=1e-4)
        dfft = np.mean(np.abs(np.fft.ifft(dfilt, axis=0)), axis=1)
        nt.assert_true(np.isclose(dfft[150:-50], 0.0).all())


if __name__ == '__main__':
    unittest.main()
