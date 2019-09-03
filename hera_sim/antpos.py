"""
A module defining routines for creating antenna array configurations. Input arguments for each function are
arbitrary, but the return value is always a dictionary with keys representing antenna numbers, and values
giving the 3D position of each antenna.
"""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import *
from past.utils import old_div
import numpy as np
from builtins import range


def linear_array(nants, sep=14.6):
    """
    Build a linear (east-west) array configuration.

    Args:
        nants (int): the number of antennas in the configuration.
        sep (float): the separation between linearly spaced antennas (meters).

    Returns:
        dict: A dictionary of antenna numbers and positions.  Positions are x,y,z
            in topocentric coordinates, in meters.
    """
    antpos = {i: np.array([sep * i, 0, 0]) for i in range(nants)}
    return antpos


def hex_array(hex_num, sep=14.6, split_core=True, outriggers=2):
    """
    Build a hexagonal array configuration, nominally matching HERA's ideal configuration.

    Args:
        hex_num (int): the hexagon (radial) number of the core configuration.
            Number of core antennas returned is 3N^2 - 3N + 1.
        sep (float): the separation between hexagonal grid points (meters).
        split_core (bool): fractures the hexagonal core into tridrents that subdivide
            a hexagonal grid. Loses N antennas, so the number of core antennas returned
            is 3N^2 - 4N + 1.
        outriggers (int): adds R extra rings of outriggers around the core that tile
            with the core to produce a fully-sampled UV plane.  The first ring
            corresponds to the exterior of a hexNum=3 hexagon. Adds 3R^2 + 9R antennas.
    Returns:
        dict: a dictionary of antenna numbers and positions.
            Positions are x,y,z in topocentric coordinates, in meters.
    """
    # Main Hex
    positions = []
    for row in range(
            hex_num - 1, -hex_num + split_core, -1
    ):  # the + split_core deletes a row
        for col in range(0, 2 * hex_num - abs(row) - 1):
            x_pos = sep * ((-(2 * hex_num - abs(row)) + 2) / 2.0 + col)
            y_pos = old_div(row * sep * 3 ** 0.5, 2)
            positions.append([x_pos, y_pos, 0])

    # unit vectors
    up_right = sep * np.asarray([0.5, old_div(3 ** 0.5, 2), 0])
    up_left = sep * np.asarray([-0.5, old_div(3 ** 0.5, 2), 0])

    # Split the core into 3 pieces
    if split_core:
        new_pos = []
        for i, pos in enumerate(positions):
            theta = np.arctan2(pos[1], pos[0])
            if pos[0] == 0 and pos[1] == 0:
                new_pos.append(pos)
            elif old_div(-np.pi, 3) < theta < old_div(np.pi, 3):
                new_pos.append(np.asarray(pos) + old_div((up_right + up_left), 3))
            elif old_div(np.pi, 3) <= theta < np.pi:
                new_pos.append(np.asarray(pos) + up_left - old_div((up_right + up_left), 3))
            else:
                new_pos.append(pos)
        positions = new_pos

    # Add outriggers
    if outriggers:
        exterior_hex_num = outriggers + 2
        for row in range(exterior_hex_num - 1, -exterior_hex_num, -1):
            for col in range(2 * exterior_hex_num - abs(row) - 1):
                x_pos = (
                        ((-(2 * exterior_hex_num - abs(row)) + 2) / 2.0 + col)
                        * sep
                        * (hex_num - 1)
                )
                y_pos = old_div(row * sep * (hex_num - 1) * 3 ** 0.5, 2)
                theta = np.arctan2(y_pos, x_pos)
                if (x_pos ** 2 + y_pos ** 2) ** 0.5 > sep * (hex_num + 1):
                    # These specific displacements of the outrigger sectors are designed specifically
                    # for redundant calibratability and "complete" uv-coverage, but also to avoid
                    # specific obstacles on the HERA site (e.g. a road to a MeerKAT antenna).
                    if 0 < theta <= old_div(2 * np.pi, 3) + 0.01:
                        positions.append(
                            np.asarray([x_pos, y_pos, 0]) - old_div(4 * (up_right + up_left), 3)
                        )
                    elif 0 >= theta > old_div(-2 * np.pi, 3):
                        positions.append(
                            np.asarray([x_pos, y_pos, 0]) - old_div(2 * (up_right + up_left), 3)
                        )
                    else:
                        positions.append(
                            np.asarray([x_pos, y_pos, 0]) - old_div(3 * (up_right + up_left), 3)
                        )

    return {i: pos for i, pos in enumerate(np.array(positions))}
