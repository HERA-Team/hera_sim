"""
Wrapper around the healvis package for producing visibilities from
healpix maps.
"""

from .simulators import VisibilitySimulator
from healvis.simulator import setup_observatory_from_uvdata
from healvis.sky_model import SkyModel
from healvis.beam_model import AnalyticBeam
from cached_property import cached_property
import numpy as np
import healpy


class HealVis(VisibilitySimulator):
    def __init__(self, fov=180, nprocesses=1, sky_ref_chan=0, **kwargs):
        self.fov = fov
        self._nprocs = nprocesses
        self._sky_ref_chan = sky_ref_chan

        # A bit of a hack here because healvis uses its own AnalyticBeam,
        # and doesn't check if you are using pyuvsim's one. This should be fixed.
        if "beams" not in kwargs:
            kwargs['beams'] = [AnalyticBeam("uniform")]

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

        sky.data = self.sky_intensity.T[np.newaxis, :, :]
        sky._update()

        return sky

    @cached_property
    def observatory(self):
        """A healvis :class:`healvis.observatory.Observatory` instance"""
        return setup_observatory_from_uvdata(
            self.uvdata, fov=self.fov, beam=self.beams[0]
        )

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

        # Add the visibility to the uvdata object
        self.uvdata.data_array += visibility[:, 0][:, np.newaxis, :, :]

        return visibility