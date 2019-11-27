"""Re-imagining of the simulation module."""

import astropy.units as u

class SimulatorBase:
    """The lowest-level of a simulation object."""

    def __init__(self, freqs):
        """What is something that everyone agrees on?"""
        self.freqs = freqs * u.Hz

    def convert(self, quantity, new_units):
        new_quantity = getattr(self, quantity).to(new_units)
        setattr(self, quantity, new_quantity)

class Sky(SimulatorBase):
    """To represent the sky."""

    def __init__(self, lsts, freqs, model):
        self.lsts = lsts * u.rad
        self.model = model
        super().__init__(freqs)

    def __call__(self, lsts=None, freqs=None):
        if lsts is None:
            lsts = self.lsts.value
        if freqs is None:
            freqs = self.freqs.value
        return self.model(lsts, freqs)

class DiffuseForeground(Sky):
    """For diffuse foregrounds."""

    def __init__(self, lsts, freqs, 
                 Tsky_mdl=None, catalog=None, omega_p=None,
                 fringe_filter_kwargs={}, delay_filter_kwargs={}):
        """Something about making an instance."""
        self._fringe_filter_kwargs = fringe_filter_kwargs
        self._delay_filter_kwargs = delay_filter_kwargs
        self._catalog = catalog
        self._omega_p = omega_p
        self._model = Tsky_mdl if Tsky_mdl is not None else catalog
        # are catalogs just interpolation objects?
        super().__init__(freqs, lsts, self._model)

    def __call__(self, bl, lsts=None, freqs=None):
        # calculate new model, then
        return super().__call__(lsts, freqs)
