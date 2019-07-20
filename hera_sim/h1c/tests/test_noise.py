import unittest
from hera_sim.h1c import noise
import numpy as np

class TestNoise(unittest.TestCase):
    def test_get_omega_p(self):
        fqs = np.linspace(0.1,0.2,1000)
        omega_p = noise.get_omega_p(fqs)
        self.assertEqual(fqs.size, omega_p.size)

if __name__=='__main__':
    unittest.main()
