"""
Wrapper around the healvis package for producing visibilities from
healpix maps.
"""

import healpy
import numpy as np
from cached_property import cached_property
from healvis.beam_model import AnalyticBeam

from .simulators import VisibilitySimulator
from astropy import constants as cnst
import pyuvdata
from pyuvsim.analyticbeam import AnalyticBeam
from RIMEz import sky_models, beam_models, rime_funcs, utils, management


class RIMEz(VisibilitySimulator):

    def __init__(self,  **kwargs):
        
        raise notImplementedError("To be implemented.")

        # A bit of a hack here because healvis uses its own AnalyticBeam,
        # and doesn't check if you are using pyuvsim's one. This should be fixed.
        beams = kwargs.get("beams", [])
        if beams == []:
            beams = [beam_models.uniform_beam_funcs]

        if isinstance(beams[0], AnalyticBeam):
            if beams[0].type == "uniform":
                beams[0] = [beam_models.uniform_beam_funcs]
            elif beams[0].type == 'gaussian':
                beams[0] = [beam_models.gaussian_beam_funcs]
        super(RIMEz, self).__init__(**kwargs)


    def validate(self):
        super(RIMEz, self).validate()

        assert len(self.beams) == 1, "RIMEz currently supports a single beam function." \
                                     " However, that beam function takes as its first " \
                                     "argument an index corresponding to antenna."

        if any([isinstance(b, pyuvdata.UVBeam) for b in self.beams]):
            raise NotImplementedError("RIMEz support for UVBeam is not currently in hera_sim")

    def _simulate(self):
        """
        Runs the zRIME algorithm
        """
        visibility = []
        for pol in self.uvdata.get_pols():
            # calculate visibility
            visibility.append(
                self.observatory.make_visibilities(self.sky_model, Nprocs=self._nprocs, beam_pol=pol)[0]
            )

        visibility = np.moveaxis(visibility, 0, -1)

        return visibility[:, 0][:, np.newaxis, :, :]


def func(list_of_njits):

    @njit
    def fnc(i, *args):
        return list_of_njits[i](*args)

    return fnc


