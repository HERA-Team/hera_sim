import unittest
from hera_sim.h1c import sigchain
import numpy as np

class TestSigchain(unittest.TestCase):
    def test_get_bandpass(self):
        fqs = np.linspace(0.1,0.2,1000)
        bandpass = sigchain.get_bandpass(fqs)
        self.assertEqual(fqs.size, bandpass.size)

if __name__=='__main__':
    unittest.main()
