"""Test the various simulation adjustment tools."""

import os
import pytest

from astropy import units
import numpy as np

from hera_sim import adjustment
from hera_sim import antpos
from hera_sim import Simulator

def antpos_equal(antpos_1, antpos_2, tol=0.1):
    return all(
        any(np.allclose(pos_1, pos_2, atol=tol) for pos_1 in antpos_1.values())
        for pos_2 in antpos_2.values()
    )

def test_antenna_matching():
    # Simple test: reflected right-triangles should match exactly
    array_1 = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [1, 1, 0]
    }

    array_2 = {
        0: [0, 0, 0],
        1: [0, 1, 0],
        2: [1, 1, 0]
    }

    array_intersection = adjustment._get_array_intersection(array_1, array_2, tol=0)
    assert antpos_equal(array_intersection, array_2, 0)
    
    # In general, rotated right-triangles should have at least two antennas in their
    # intersection.
    array_3 = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [0, 1, 0]
    }
    array_intersection = adjustment._get_array_intersection(array_1, array_3, tol=0)
    assert len(array_intersection) == 2

    # A simple translation should just be undone
    translation = np.random.uniform(-1, 1, 3)
    array_4 = {ant: np.array(pos) - translation for ant, pos in array_1.items()}
    array_intersection = adjustment._get_array_intersection(array_1, array_4, tol=0.1)
    assert antpos_equal(array_intersection, array_4, 0.1)

    # A small hex array should be a subset of a larger hex array
    hex_array = antpos.HexArray(split_core=False, outriggers=0)
    hex_array_1 = hex_array(4)
    hex_array_2 = hex_array(3)
    array_intersection = adjustment._get_array_intersection(hex_array_1, hex_array_2)
    assert antpos_equal(array_intersection, hex_array_2)
    assert len(array_intersection) == len(hex_array_2)
