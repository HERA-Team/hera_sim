"""Make some noise."""

import astropy.constants as const
import astropy.units as u
from .components import registry
from . import utils

@registry
class Noise:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ThermalNoise(Noise):
    __aliases__ = ("thermal_noise", )

    def __init__(self, Tsky_mdl=None, omega_p=None, 
                 integration_time=None, channel_width=None,
                 Trx=0):
        # TODO: docstring
        """
        """
        super().__init__(
            Tsky_mdl=Tsky_mdl,
            omega_p=omega_p,
            integration_time=integration_time,
            channel_width=channel_width,
            Trx=Trx)
    # XXX update this

    def __call__(self, lsts, freqs, **kwargs):
        # TODO: docstring
        """
        """
        # validate the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (Tsky_mdl, omega_p, integration_time, channel_width, 
            Trx) = self._extract_kwarg_values(**kwargs)

        # get the channel width if not specified
        if channel_width is None:
            channel_width = np.mean(np.diff(freqs))
        # get the integration time if not specified
        if integration_time is None:
            integration_time = np.mean(np.diff(lsts)) / (2*np.pi)
            integration_time *= u.sday.to("s")

        # resample the sky temperature model
        Tsky = Tsky_mdl(lsts, freqs)
        # calculate noise visibility in units of K, assuming Tsky
        # is in units of K
        vis = Tsky / np.sqrt(integration_time * channel_width)
        # add receiver temperature
        vis += Trx
        # convert vis to Jy
        # XXX why the reshape?
        vis /= utils.Jy2T(freqs, omega_p).reshape(1, -1)
        # make it noisy
        return utils.gen_white_noise(size=vis.shape) * vis
