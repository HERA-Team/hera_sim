import unittest
from hera_sim import hera_season

class TestSeason(unittest.TestCase):
    def test_set_default(self):
        init_default = hera_season.DEFAULT_SEASON
        if init_default=='h2c':
            hera_season.set_default('h1c')
        else:
            hera_season.set_default('h2c')
        new_default = hera_season.DEFAULT_SEASON
        self.assertNotEqual(init_default, new_default)

    def test_get_season(self):
        h1c = hera_season.h1c
        h2c = hera_season.h2c
        self.assertIs(h1c, hera_season.get_season('h1c'))
        self.assertIs(h2c, hera_season.get_season('h2c'))

if __name__=='__main__':
    unittest.main()

