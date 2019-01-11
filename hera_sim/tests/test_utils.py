import numpy as np

from hera_sim import utils


def test_bl_vec():
    bl = 1

    assert len(utils._get_bl_len_vec(bl)) == 3

    bl = (0, 1)

    assert len(utils._get_bl_len_vec(bl)) == 3

    bl = [0, 1]

    assert len(utils._get_bl_len_vec(bl)) == 3

    bl = np.array([0, 1, 2])

    assert len(utils._get_bl_len_vec(bl)) == 3
