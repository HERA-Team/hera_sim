import unittest
import numpy as np

from hera_sim import io


class TestIO(unittest.TestCase):
    def test_empty_uvdata(self):
        # Make sure that empty_uvdata() can produce a UVData object
        nfreqs = 150
        ntimes = 20
        ants = {0: (1.0, 2.0, 3.0), 1: (3.0, 4.0, 5.0)}
        antpairs1 = [(0, 1), (1, 0), (1, 1)]
        antpairs2 = [(0, 1), (1, 0), (1, 1), (0, 1)]  # duplicate baseline

        # Build object and check that data_array has correct dimensions
        uvd = io.empty_uvdata(nfreqs, ntimes, ants=ants, antpairs=antpairs1)
        self.assertEqual(uvd.data_array.shape, (len(antpairs1) * ntimes, 1, nfreqs, 1))

        # Check that duplicate baselines get filtered out
        uvd = io.empty_uvdata(nfreqs, ntimes, ants=ants, antpairs=antpairs2)
        self.assertEqual(uvd.data_array.shape,
                         (len(antpairs1) * ntimes, 1, nfreqs, 1))

    def test_antpair_order(self):
        nfreqs = 10
        ntimes = 10
        ants = {j: tuple(np.random.rand(3)) for j in range(10)}
        uvd = io.empty_uvdata(nfreqs, ntimes, ants=ants)
        for ant1, ant2 in uvd.get_antpairs():
            self.assertLessEqual(ant1,ant2)

if __name__ == '__main__':
    unittest.main()
