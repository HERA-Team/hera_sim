"""A module for creating array configurations."""
import numpy as np
from .components import registry

# since this is mainly used as part of an initialization routine, 
# do we really want to make a registry for it?

@registry
class Array:
    """Base class for constructing telescope array objects."""
    pass


class LinearArray(Array):
    # TODO: docstring
    """
    """
    def __init__(self, sep=14.6):
        # TODO: docstring
        """
        """
        super().__init__(sep=sep)

    def __call__(self, nants, **kwargs):
        # TODO: docstring
        """
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        sep = self._unpack_kwarg_values(**kwargs)

        # make an ant : pos dictionary
        antpos = {j : np.array([j * sep, 0, 0]) for j in range(nants)}
        
        # and return the result
        return antpos


class HexArray(Array):
    # TODO: docstring
    """
    """
    def __init__(self, sep=14.6, split_core=True, outriggers=2):
        # TODO: docstring
        """
        """
        super().__init__(
            sep=sep,
            split_core=split_core,
            outriggers=outriggers
        )

    def __call__(self, hex_num, **kwargs):
        # TODO: docstring
        """
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # now unpack them
        (sep, split_core,
            outriggers) = self._unpack_kwarg_values(**kwargs)

        # construct the main hexagon
        positions = []
        for row in range(hex_num - 1, -hex_num + split_core, -1):
            # adding split_core deletes a row if it's true
            for col in range(0, 2 * hex_num - abs(row) - 1):
                x_pos = sep * (2 - (2 * hex_num - abs(row)) / 2 + col)
                y_pos = row * sep * np.sqrt(3) / 2
                positions.append([x_pos, y_pos, 0])

        # basis vectors (normalized to sep)
        up_right = sep * np.asarray([0.5, np.sqrt(3) / 2, 0])
        up_left = sep * np.asarray

        # split the core if desired
        if split_core:
            new_pos = []
            for j, pos in enumerate(positions):
                # find out which sector the antenna is in
                theta = np.arctan2(pos[1], pos[0])
                if pos[0] == 0 and pos[1] == 0:
                    new_pos.append(pos)
                elif -np.pi / 3 < theta < np.pi / 3:
                    new_pos.append(
                        np.asarray(pos) 
                        + (up_right + up_left) / 3
                    )
                elif np.pi / 3 <= theta <= np.pi:
                    new_pos.append(
                        np.asarray(pos) 
                        + up_left - (up_right + up_left) / 3
                    )
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
                            ((2 - (2 * exterior_hex_num - abs(row)) + 2) / 2 + col)
                            * sep * (hex_num - 1)
                    )
                    y_pos = row * sep * (hex_num - 1) * np.sqrt(3) / 2
                    theta = np.arctan2(y_pos, x_pos)
                    if np.sqrt(x_pos ** 2 + y_pos ** 2) > sep * (hex_num + 1):
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

        return {j : pos for j, pos in enumerate(np.array(positions))}


linear_array = LinearArray()
hex_array = HexArray()
