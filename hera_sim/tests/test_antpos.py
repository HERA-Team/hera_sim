import unittest
from hera_sim import antpos
import numpy as np

np.random.seed(0)

class TestLinearArray(unittest.TestCase):
    def test_nant(self):
        for N in range(1, 200, 20):
            xyz = antpos.linear_array(N)
            self.assertEqual(len(xyz), N)
        xyz = antpos.linear_array(300)
        self.assertEqual(range(300), sorted(xyz.keys()))
    def test_positions(self):
        xyz = antpos.linear_array(20, sep=1.)
        for i in range(20):
            bl = xyz[i] - xyz[0]
            bl_len = np.sqrt(np.dot(bl, bl))
            self.assertEqual(bl_len, i)
        xyz = antpos.linear_array(20, sep=2.)
        for i in range(20):
            bl = xyz[i] - xyz[0]
            bl_len = np.sqrt(np.dot(bl, bl))
            self.assertEqual(bl_len, 2*i)

class TestHexArray(unittest.TestCase):
    def test_nant(self):
        # Test core scaling
        for N in range(1, 15):
            xyz = antpos.hex_array(N, split_core=False, outriggers=0)
            self.assertEqual(len(xyz), 3*N**2 - 3*N + 1)
        xyz = antpos.hex_array(15, split_core=False, outriggers=0)
        self.assertEqual(range(3*15**2 - 3*15 + 1), sorted(xyz.keys()))
        # Test outrigger scaling
        N = 10
        Ncore = 3*N**2 - 3*N + 1
        for R in range(0, 3):
            xyz = antpos.hex_array(N, split_core=False, outriggers=R)
            self.assertEqual(len(xyz), Ncore + 3*R**2 + 9*R)
        # Test core split
        for N in range(2, 15):
            xyz = antpos.hex_array(N, split_core=True, outriggers=0)
            self.assertEqual(len(xyz), 3*N**2 - 4*N + 1)
        xyz = antpos.hex_array(15, split_core=True, outriggers=0)
        self.assertEqual(range(3*15**2 - 4*15 + 1), sorted(xyz.keys()))
    def test_positions(self):
        # Test one single answer
        xyz = antpos.hex_array(2, split_core=False, outriggers=0, sep=1.)
        for i in range(7):
            if i == 3: continue
            bl = xyz[i] - xyz[3]
            bl_len = np.sqrt(np.dot(bl, bl))
            self.assertAlmostEqual(bl_len, 1, 10)
        xyz = antpos.hex_array(2, split_core=False, outriggers=0, sep=2.)
        # Test scaling with sep
        for i in range(7):
            if i == 3: continue
            bl = xyz[i] - xyz[3]
            bl_len = np.sqrt(np.dot(bl, bl))
            self.assertAlmostEqual(bl_len, 2, 10)
        # Test top corner of a lot of configs
        for N in range(2, 15):
            xyz = antpos.hex_array(N, split_core=False, outriggers=0, sep=1.)
            bl1 = xyz[1] - xyz[0]
            bl2 = xyz[N] - xyz[0]
            np.testing.assert_allclose(bl1, [1, 0, 0], atol=1e-10)
            np.testing.assert_allclose(bl2, [-.5, -3**.5/2, 0], atol=1e-10)
        # Test split
        # XXX todo
        # Test outriggers
        # XXX todo
        

if __name__ == '__main__':
    unittest.main()
