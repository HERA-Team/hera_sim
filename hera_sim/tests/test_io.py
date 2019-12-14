import unittest
import numpy as np

from hera_sim import io


class TestIO(unittest.TestCase):
    # define some extra parameters, since these all default to None
    extra_init_params = {
        "start_time" : 2458119.5, "integration_time" : 10.7,
        "start_freq" : 1e8, "channel_width" : 1e6,
        "polarization_array" : ['xx']
    }

    def test_empty_uvdata(self):
        # Make sure that empty_uvdata() can produce a UVData object
        Nfreqs = 150
        Ntimes = 20
        ants = {0: (1.0, 2.0, 3.0), 1: (3.0, 4.0, 5.0)}
        antpairs1 = [(0, 1), (1, 0), (1, 1)]
        antpairs2 = [(0, 1), (1, 0), (1, 1), (0, 1)]  # duplicate baseline

        # Build object and check that data_array has correct dimensions
        uvd = io.empty_uvdata(
            Nfreqs=Nfreqs, Ntimes=Ntimes, array_layout=ants, 
            antpairs=antpairs1, **self.extra_init_params
        )
        
        self.assertEqual(
            uvd.data_array.shape, (len(antpairs1) * Ntimes, 1, Nfreqs, 1)
        )

        # Check that duplicate baselines get filtered out
        uvd = io.empty_uvdata(
            Nfreqs=Nfreqs, Ntimes=Ntimes, array_layout=ants, 
            antpairs=antpairs2, **self.extra_init_params
        )
        
        self.assertEqual(
            uvd.data_array.shape, (len(antpairs1) * Ntimes, 1, Nfreqs, 1)
        )

    def test_antpair_order(self):
        ants = {j: tuple(np.random.rand(3)) for j in range(10)}
        uvd = io.empty_uvdata(
            Ntimes=10, Nfreqs=10, array_layout=ants,
            **self.extra_init_params
        )
        for ant1, ant2 in uvd.get_antpairs():
            self.assertLessEqual(ant1,ant2)

if __name__ == '__main__':
    unittest.main()
