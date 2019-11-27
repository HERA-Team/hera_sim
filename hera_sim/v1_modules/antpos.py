"""A module for creating array configurations."""

class Array:
    """Base class for constructing telescope array objects."""

    def __init__(self, sep, *args, **kwargs):
        """Some basic need-to-knows for characterizing the array.
        
        Parameters
        ----------
        sep : float
            Separation between adjacent antennas.
        """
        self.sep = sep
        self.config = None

