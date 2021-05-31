"""Wrapper for healvis so that it accepts pyuvsim configuration inputs."""
from __future__ import division

import astropy_healpix as aph
import numpy as np
from cached_property import cached_property

import pyuvsim

from .simulators import VisibilitySimulator
from astropy import constants as cnst

try:
    from healvis.beam_model import AnalyticBeam
    from healvis.simulator import setup_observatory_from_uvdata
    from healvis.sky_model import SkyModel

    HAVE_HEALVIS = True
except ImportError:
    HAVE_HEALVIS = False


class HealVis(VisibilitySimulator):
    """Wrapper for healvis to produce visibilities from HEALPix maps.

    Parameters
    ----------
    fov : float
        Field of view (diameter) in degrees. Defaults to 180.
    nprocesses : int
        Number of concurrent processes. Defaults to 1.
    sky_ref_chan : float
        Frequency reference channel. Defaults to 0.

    Other Parameters
    ----------------
    Passed through to :class:`VisibilitySimulator`.
    """

    point_source_ability = False

    def __init__(self, fov=180, nprocesses=1, sky_ref_chan=0, **kwargs):
        if not HAVE_HEALVIS:
            raise ImportError("to use the healvis wrapper, you must install healvis!")

        self.fov = fov
        self._nprocs = nprocesses
        self._sky_ref_chan = sky_ref_chan

        # A bit of a hack here because healvis uses its own AnalyticBeam, and
        # doesn't check if you are using pyuvsim's one. This should be fixed.

        if "beams" not in kwargs:
            kwargs["beams"] = [AnalyticBeam("uniform")]

        super(HealVis, self).__init__(**kwargs)

        # Check if pyuvsim.analyticbeam and switch to healvis.beam_model
        if isinstance(self.beams[0], pyuvsim.analyticbeam.AnalyticBeam):
            old_args = self.beams[0].__dict__

            gauss_width = None
            if old_args["type"] == "gaussian":
                if old_args["sigma"] is None:
                    raise NotImplementedError(
                        "Healvis does not permit " "gaussian beam with diameter."
                    )
                raise NotImplementedError(
                    "Healvis interprets gaussian beams "
                    "as per-baseline and not "
                    "per-antenna as required here."
                )
                # Healvis expects degrees
                gauss_width = old_args["sigma"] * 180 / np.pi

            beam_type = old_args["type"]
            spectral_index = old_args["spectral_index"]
            diameter = old_args["diameter"]
            self.beams = [
                AnalyticBeam(
                    beam_type=beam_type,
                    gauss_width=gauss_width,
                    diameter=diameter,
                    spectral_index=spectral_index,
                )
            ]

    def validate(self):
        """Validate that all data is correct.

        In addition to standard parameter restrictions, HealVis requires a single beam
        for all antennae.
        """
        super(HealVis, self).validate()
        assert self.n_beams == 1

    @cached_property
    def sky_model(self):
        """
        A ``SkyModel`` compatible with healvis.

        Returns
        -------
        SkyModel object
            healvis SkyModel constructed from the input HEALPix sky
            model.
        """
        sky = SkyModel()
        sky.Nside = aph.npix_to_nside(self.sky_intensity.shape[1])
        sky.freqs = self.sky_freqs
        sky.Nskies = 1
        sky.ref_chan = self._sky_ref_chan

        # convert from Jy/sr to K
        intensity = 10 ** -26 * self.sky_intensity.T
        intensity *= (cnst.c.to("m/s").value / self.sky_freqs) ** 2 / (
            2 * cnst.k_B.value
        )

        sky.data = intensity[np.newaxis, :, :]
        sky._update()

        return sky

    @cached_property
    def observatory(self):
        """
        A healvis :class:`healvis.observatory.Observatory` instance.

        Returns
        -------
        Observatory object
            healvis Observatory constructed from input telescope
            parameters.
        """
        return setup_observatory_from_uvdata(
            self.uvdata, fov=self.fov, beam=self.beams[0],
        )

    def _simulate(self):
        """
        Runs the healvis algorithm.

        Returns
        -------
        Visibility from all sources.
            Shape=self.uvdata.data_array.shape.
        """
        visibility = []
        for pol in self.uvdata.get_pols():
            # calculate visibility
            visibility.append(
                self.observatory.make_visibilities(
                    self.sky_model, Nprocs=self._nprocs, beam_pol=pol
                )[0]
            )

        visibility = np.moveaxis(visibility, 0, -1)

        return visibility[:, 0][:, np.newaxis, :, :]
