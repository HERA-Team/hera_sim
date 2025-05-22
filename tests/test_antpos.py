import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hera_sim import antpos


def check_ant_labels(Nants_expected, ants):
    return set(range(Nants_expected)) == set(ants)


def test_linear_array_nants():
    for Nants in range(1, 202, 20):
        ants = antpos.linear_array(Nants)
        assert check_ant_labels(Nants_expected=Nants, ants=ants)


def test_linear_array_positions():
    for sep in np.linspace(1, 10, 10):
        for Nants in range(5, 21, 5):
            ants = antpos.linear_array(Nants, sep=sep)
            for i in range(Nants):
                bl = ants[i] - ants[0]
                bl_len = np.linalg.norm(bl)
                assert bl_len == i * sep


def test_hex_array_core_nants():
    for hex_num in range(1, 15):
        Nants = 3 * hex_num * (hex_num - 1) + 1
        ants = antpos.hex_array(hex_num, split_core=False, outriggers=0)
        assert check_ant_labels(Nants_expected=Nants, ants=ants)


def test_hex_array_outrigger_nants_scaling():
    for hex_num in range(4, 15):
        Nants_core = 3 * hex_num * (hex_num - 1) + 1
        for Nrings in range(3):
            Noutriggers = 3 * Nrings * (Nrings + 3)
            Nants = Nants_core + Noutriggers
            ants = antpos.hex_array(hex_num, split_core=False, outriggers=Nrings)
            assert check_ant_labels(Nants_expected=Nants, ants=ants)


def test_hex_array_split_core_nants_scaling():
    for hex_num in range(2, 15):
        Nants_core = 3 * hex_num * (hex_num - 1) + 1
        Nants = Nants_core - hex_num
        ants = antpos.hex_array(hex_num, split_core=True, outriggers=0)
        assert check_ant_labels(Nants_expected=Nants, ants=ants)


def test_hex_array_7_element_positions():
    for sep in np.linspace(1, 10, 10):
        ants = antpos.hex_array(2, split_core=False, outriggers=0, sep=sep)
        # Central antenna is number 3; all should be equidistant from 3.
        for i in range(7):
            if i == 3:
                continue
            bl = ants[i] - ants[3]
            bl_len = np.linalg.norm(bl)
            assert np.isclose(bl_len, sep)


def test_hex_array_core_corner_positions():
    for sep in np.linspace(1, 10, 10):
        for hex_num in range(2, 15):
            ants = antpos.hex_array(hex_num, split_core=False, outriggers=0, sep=sep)
            bl1 = ants[1] - ants[0]
            bl2 = ants[hex_num] - ants[0]
            expected_bl1 = sep * np.array([1, 0, 0])
            expected_bl2 = sep * np.array([-0.5, -np.sqrt(3) / 2, 0])
            assert np.allclose(bl1, expected_bl1)
            assert np.allclose(bl2, expected_bl2)


def test_hex_array_split_core_positions():
    # This test is rather brittle and could be improved.
    ants = antpos.hex_array(3, sep=14.6, split_core=True, outriggers=0)
    bl1 = ants[3] - ants[7]
    bl2 = ants[5] - ants[10]
    expected_bl1 = 14.6 * np.array([0, 2 / np.sqrt(3), 0])
    expected_bl2 = 14.6 * np.array([-1, 1 / np.sqrt(3), 0])
    assert np.allclose(bl1, expected_bl1)
    assert np.allclose(bl2, expected_bl2)


def test_hex_array_HERA350_positions():
    # This test is rather brittle and could be improved.
    ants = antpos.hex_array(11, sep=14.6, split_core=True, outriggers=2)
    bl1 = ants[348] - ants[347]
    bl2 = ants[347] - ants[346]
    bl3 = ants[325] - ants[324]
    expected_bl1 = 14.6 * np.array([10, 0, 0])
    expected_bl2 = 14.6 * np.array([10, 1 / np.sqrt(3), 0])
    expected_bl3 = 14.6 * np.array([10, -1 / np.sqrt(3), 0])
    assert np.allclose(bl1, expected_bl1)
    assert np.allclose(bl2, expected_bl2)
    assert np.allclose(bl3, expected_bl3)


@pytest.mark.parametrize(
    "rot", [Rotation.from_euler("x", [0]), Rotation.from_euler("xyz", [0.1, 0.2, 0.3])]
)
def test_idealize_antpos_identity(rot):
    """Test that idealizing a perfectly redundant array doesn't change it."""
    ants = antpos.hex_array(5, sep=14.6, split_core=False, outriggers=False)
    ants = {k: rot.apply(v).ravel() for k, v in ants.items()}

    ideal = antpos.idealize_antpos(ants, bl_error_tol=0.1)
    assert all(np.allclose(ants[a], ideal[a]) for a in ants)


@pytest.mark.parametrize(
    "rot",
    [
        Rotation.from_euler("x", [0]),
        Rotation.from_euler("xyz", [0.1, 0.2, 0.3]),
        Rotation.from_euler("x", [np.pi / 3]),
    ],
)
def test_idealize_antpos_perturb(rot):
    """Test that idealizing a perfectly redundant array doesn't change it."""
    ants = antpos.hex_array(5, sep=14.6, split_core=False, outriggers=False)
    ants = {k: rot.apply(v).ravel() for k, v in ants.items()}
    rng = np.random.default_rng(12)
    pants = {k: v + rng.normal(scale=0.001, size=3) for k, v in ants.items()}

    ideal = antpos.idealize_antpos(pants, bl_error_tol=0.1)

    ants = np.array(list(ants.values()))
    ideal = np.array(list(ideal.values()))

    np.testing.assert_allclose(ants, ideal, atol=0.001)
