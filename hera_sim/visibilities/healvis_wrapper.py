"""
Wrapper around the healvis package for producing visibilities from
healpix maps.
"""
from __future__ import division

from past.utils import old_div
import healpy
import numpy as np
from cached_property import cached_property

import pyuvsim

try:
    from healvis.beam_model import AnalyticBeam
    from healvis.simulator import setup_observatory_from_uvdata
    from healvis.sky_model import SkyModel
except ImportError:
    raise ImportError("to use the healvis wrapper, you must install healvis!")

from .simulators import VisibilitySimulator
from astropy import constants as cnst


class HealVis(VisibilitySimulator):
    point_source_ability = False

    def __init__(self, fov=180, nprocesses=1, sky_ref_chan=0, **kwargs):
        self.fov = fov
        self._nprocs = nprocesses
        self._sky_ref_chan = sky_ref_chan

        # A bit of a hack here because healvis uses its own AnalyticBeam,
        # and doesn't check if you are using pyuvsim's one. This should be fixed.
        

        
        if "beams" not in kwargs:
            kwargs['beams'] = [AnalyticBeam("uniform")]
            
        # Check if pyuvsim.analyticbeam and switch to healvis.beam_model
        if isinstance(kwargs['beams'][0], pyuvsim.analyticbeam.AnalyticBeam):
            old_args = kwargs['beams'][0].__dict__
            
            if old_args['type'] == "gaussian":
                raise NotImplementedError("HEALVIS DOES NOT PERMIT GAUSSIAN BEAM + DIAMETER")
            
            beam_type=old_args['type']
            ref_freq=old_args['ref_freq']
            spectral_index=old_args['spectral_index']
            diameter=old_args['diameter']
            kwargs['beams'] = [AnalyticBeam(beam_type=beam_type, ref_freq=ref_freq, spectral_index=spectral_index, diameter=diameter)]
            
        super(HealVis, self).__init__(**kwargs)

    def validate(self):
        super(HealVis, self).validate()

        assert self.n_beams == 1, "HealVis assumes a single beam for all antennae"

    @cached_property
    def sky_model(self):
        """A SkyModel, compatible with healvis, constructed from the input healpix sky model"""
        sky = SkyModel()
        sky.Nside = healpy.npix2nside(self.sky_intensity.shape[1])
        sky.freqs = self.sky_freqs
        sky.Nskies = 1
        sky.ref_chan = self._sky_ref_chan

        # convert from Jy/sr to K
        intensity = 10**-26 * self.sky_intensity.T
        intensity *= old_div((old_div(cnst.c.to("m/s").value,self.sky_freqs))**2, (2 * cnst.k_B.value))

        sky.data = intensity[np.newaxis, :, :]
        sky._update()

        return sky

    @cached_property
    def observatory(self):
        """A healvis :class:`healvis.observatory.Observatory` instance"""
        obs =  setup_observatory_from_uvdata(
            self.uvdata, fov=self.fov, beam=self.beams[0]
        )
        
        return obs

    def _simulate(self):
        """
        Runs the healvis algorithm
        """
        visibility = []
        for pol in self.uvdata.get_pols():
            # calculate visibility
            visibility.append(
                self.observatory.make_visibilities(self.sky_model, Nprocs=self._nprocs, beam_pol=pol)[0]
            )

        visibility = np.moveaxis(visibility, 0, -1)

        return visibility[:, 0][:, np.newaxis, :, :]
