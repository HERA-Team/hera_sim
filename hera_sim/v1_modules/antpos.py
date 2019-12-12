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
    pass

linear_array = LinearArray()
hex_array = HexArray()
