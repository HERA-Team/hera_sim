import unittest
from hera_sim import io
import numpy as np


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
        self.assertEqual( uvd.data_array.shape, 
                          (len(antpairs1)*ntimes, 1, nfreqs, 1) )

    def test_cross_antpairs(self):
        ants = {
            0: (1., 2., 3.),
            1: (3., 4., 5.),
            2: (5. ,6. ,7.)
        }

        antpairs, ant1, ant2 = io._get_antpairs(ants, "cross")

        self.assertEqual(antpairs, [(0, 1), (0, 2), (1, 2)])
        self.assertEqual(ant1, (0, 0, 1))
        self.assertEqual(ant2, (1, 2, 2))

    def test_auto_antpairs(self):
        ants = {
            0: (1., 2., 3.),
            1: (3., 4., 5.),
            2: (5. ,6. ,7.)
        }

        antpairs, ant1, ant2 = io._get_antpairs(ants, "autos")

        self.assertEqual(antpairs, [(0,0), (1,1), (2, 2)])
        self.assertEqual(ant1, (0, 1, 2))
        self.assertEqual(ant2, (0, 1, 2))

    def test_EW_antpairs(self):
        ants = {
            0: (1., 2., 3.),
            1: (3., 2., 3.),
            2: (5. ,6. ,7.)
        }

        antpairs, ant1, ant2 = io._get_antpairs(ants, "EW")

        self.assertEqual(set(antpairs), {(0, 0), (1,1), (2,2), (0,1)})

    def test_redundant_antpairs(self):
        ants = {
            0: (1., 2., 3.),
            1: (2., 3., 3.),
            2: (3., 4., 3.)
        }

        antpairs, ant1, ant2 = io._get_antpairs(ants, "redundant")

        self.assertEqual(set(antpairs), {(0, 0), (1,1), (2,2 ), (0,1), (0, 2)})

    def test_bad_antpairs(self):
        # Make sure that empty_uvdata() can produce a UVData object
        nfreqs = 150
        ntimes = 20
        ants = {
            0: (1., 2., 3.),
            1: (3., 4., 5.)
        }
        antpairs1 = [(0, 1), (1, 0), (1, 1), 1]

        with self.assertRaises(TypeError):
            uvd = io.empty_uvdata(nfreqs, ntimes, ants=ants, antpairs=antpairs1)


if __name__ == '__main__':
    unittest.main()
