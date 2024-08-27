"""A module for creating antenna array configurations.

Input parameters vary between functions, but all functions return a
dictionary whose keys refer to antenna numbers and whose values refer
to the ENU position of the antennas.
"""

import numpy as np
from scipy.optimize import least_squares

from .components import component

# FIXME: old docstrings state that the positions are returned in topocentric
# coordinates, but this is contradictory to the claim that a linear array
# is constructed as purely east-west. Let's resolve this before publishing v1


@component
class Array:
    """Base class for constructing telescope array objects."""

    pass


class LinearArray(Array):
    """Build a linear (east-west) array configuration.

    Parameters
    ----------
    sep : float, optional
        The separation between adjacent antennas, in meters.
        Default separation is 14.6 meters.
    """

    def __init__(self, sep=14.6):
        super().__init__(sep=sep)

    def __call__(self, nants, **kwargs):
        """Compute the antenna positions.

        Parameters
        ----------
        nants : int
            The number of antennas in the configuration.

        Returns
        -------
        antpos : dict
            Dictionary of antenna numbers and ENU positions. Positions
            are given in meters.
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (sep,) = self._extract_kwarg_values(**kwargs)

        # make an ant : pos dictionary
        antpos = {j: np.asarray([j * sep, 0, 0]) for j in range(nants)}

        # and return the result
        return dict(antpos)


class HexArray(Array):
    """Build a hexagonal array configuration, nominally matching HERA.

    Parameters
    ----------
    sep : int, optional
        The separation between adjacent grid points, in meters.
        Default separation is 14.6 meters.
    split_core : bool, optional
        Whether to fracture the core into tridents that subdivide a
        hexagonal grid. Loses :math:`N` antennas. Default behavior
        is to split the core.
    outriggers : int, optional
        The number of rings of outriggers to add to the array. The
        outriggers tile with the core to produce a fully-sampled
        UV plane. The first ring corresponds to the exterior of a
        hex_num=3 hexagon. For :math:`R` outriggers, :math:`3R^2 + 9R`
        antennas are added to the array.
    """

    def __init__(self, sep=14.6, split_core=True, outriggers=2):
        super().__init__(sep=sep, split_core=split_core, outriggers=outriggers)

    def __call__(self, hex_num, **kwargs):
        """Compute the positions of the antennas.

        Parameters
        ----------
        hex_num : int
            The hexagon (radial) number of the core configuration. The
            number of core antennas returned is :math:`3N^2 - 3N + 1`.

        Returns
        -------
        antpos : dict
            Dictionary of antenna numbers and positions, in ENU
            coordinates. Antenna positions are given in units of meters.
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # now unpack them
        (sep, split_core, outriggers) = self._extract_kwarg_values(**kwargs)

        # construct the main hexagon
        positions = []
        for row in range(hex_num - 1, -hex_num + split_core, -1):
            # adding split_core deletes a row if it's true
            for col in range(2 * hex_num - abs(row) - 1):
                x_pos = sep * ((2 - (2 * hex_num - abs(row))) / 2 + col)
                y_pos = row * sep * np.sqrt(3) / 2
                positions.append([x_pos, y_pos, 0])

        # basis vectors (normalized to sep)
        up_right = sep * np.asarray([0.5, np.sqrt(3) / 2, 0])
        up_left = sep * np.asarray([-0.5, np.sqrt(3) / 2, 0])

        # split the core if desired
        if split_core:
            new_pos = []
            for pos in positions:
                # find out which sector the antenna is in
                theta = np.arctan2(pos[1], pos[0])
                if pos[0] == 0 and pos[1] == 0:
                    new_pos.append(pos)
                elif -np.pi / 3 < theta < np.pi / 3:
                    new_pos.append(np.asarray(pos) + (up_right + up_left) / 3)
                elif np.pi / 3 <= theta < np.pi:
                    new_pos.append(np.asarray(pos) + up_left - (up_right + up_left) / 3)
                else:
                    new_pos.append(pos)
            # update the positions
            positions = new_pos

        # add outriggers if desired
        if outriggers:
            # The specific displacements of the outrigger sectors are
            # designed specifically for redundant calibratability and
            # "complete" uv-coverage, but also to avoid specific
            # obstacles on the HERA site (e.g. a road to a MeerKAT antenna)
            exterior_hex_num = outriggers + 2
            for row in range(exterior_hex_num - 1, -exterior_hex_num, -1):
                for col in range(2 * exterior_hex_num - abs(row) - 1):
                    x_pos = (
                        ((2 - (2 * exterior_hex_num - abs(row))) / 2 + col)
                        * sep
                        * (hex_num - 1)
                    )
                    y_pos = row * sep * (hex_num - 1) * np.sqrt(3) / 2
                    theta = np.arctan2(y_pos, x_pos)
                    if np.sqrt(x_pos**2 + y_pos**2) > sep * (hex_num + 1):
                        if 0 < theta <= 2 * np.pi / 3 + 0.01:
                            positions.append(
                                np.asarray([x_pos, y_pos, 0])
                                - 4 * (up_right + up_left) / 3
                            )
                        elif 0 >= theta > -2 * np.pi / 3:
                            positions.append(
                                np.asarray([x_pos, y_pos, 0])
                                - 2 * (up_right + up_left) / 3
                            )
                        else:
                            positions.append(
                                np.asarray([x_pos, y_pos, 0])
                                - 3 * (up_right + up_left) / 3
                            )

        antpos = dict(enumerate(np.array(positions)))

        return dict(antpos)


linear_array = LinearArray()
hex_array = HexArray()


def idealize_antpos(
    antpos: dict[int, np.ndarray], bl_error_tol: float = 1.0
) -> dict[int, np.ndarray]:
    """Snap antenna positions to a grid that ensures perfect redundancy.

    Parameters
    ----------
    antpos
        The antenna positions, in ENU coordinates, to be idealized. The format is a
        dict, where the key is the antenna number, and the value is the length-3 vector
        of ENU coordinates.

    Returns
    -------
    idealized_antpos
        A dict in the same format as the input antpos, where the positions have been
        snapped to the redundancy grid.
        _type_: _description_
    """
    from hera_cal import redcal

    reds = redcal.get_reds(antpos, bl_error_tol=bl_error_tol)
    idealized_antpos = redcal.reds_to_antpos(reds)

    for ant in idealized_antpos:
        idealized_antpos[ant] = np.array(
            [
                idealized_antpos[ant][0],
                idealized_antpos[ant][1],
                0.0,
            ]
        )

    def transform_points(M, source_points):
        """
        Apply the transformation to the source points.

        M is a 12-element array
        [r11, r12, r13, r21, r22, r23, r31, r32, r33, tx, ty, tz]
        representing a 3x3 rotation matrix and a 3x1 translation vector.
        """
        R = M[:9].reshape(3, 3)  # Rotation matrix
        t = M[9:].reshape(3, 1)  # Translation vector

        # Apply transformation: R * source_points + t
        transformed_points = np.dot(R, source_points) + t
        return transformed_points

    def residuals(M, source_points, target_points):
        """Calculate residuals between transformed source points and target points."""
        transformed_points = transform_points(M, source_points)
        return (transformed_points - target_points).ravel()

    # Assuming source_points and target_points are your 3x350 numpy arrays
    target_points = np.array([antpos[k] for k in sorted(antpos.keys())]).T
    source_points = np.array(
        [idealized_antpos[k] for k in sorted(idealized_antpos.keys())]
    ).T

    # Initial guess for the parameters:
    # [identity matrix for rotation, zero vector for translation]
    initial_guess = np.hstack((np.eye(3).ravel(), np.zeros(3)))

    # Perform the least squares optimization
    result = least_squares(
        residuals, initial_guess, args=(source_points, target_points)
    )

    # Extract the optimal transformation matrix and translation
    optimal_params = result.x
    R_optimal = optimal_params[:9].reshape(3, 3)
    t_optimal = optimal_params[9:].reshape(3)

    idealized_positions_in_real_space = {
        ant: R_optimal @ pos + t_optimal for ant, pos in idealized_antpos.items()
    }
    return idealized_positions_in_real_space
