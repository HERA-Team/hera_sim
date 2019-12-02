"""A module for creating array configurations."""

from .components import registry

# since this is mainly used as part of an initialization routine, 
# do we really want to make a registry for it?

class Array:
    """Base class for constructing telescope array objects."""

    def __init__(self, sep=14.6, *args, **kwargs):
        """Some basic need-to-knows for characterizing the array.
        
        Parameters
        ----------
        sep : float
            Separation between adjacent antennas in meters.
        """
        self.sep = sep

class LinearArray(Array):
    pass

class HexArray(Array):
    pass

linear_array = LinearArray()
hex_array = HexArray()
