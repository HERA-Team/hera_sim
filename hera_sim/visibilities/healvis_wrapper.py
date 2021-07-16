"""Wrapper for healvis so that it accepts pyuvsim configuration inputs."""
from __future__ import division

import numpy as np

import pyuvsim

from .simulators import VisibilitySimulator, ModelData, SkyModel

from astropy import constants as cnst
from astropy import units

try:
    from healvis.beam_model import AnalyticBeam
    from healvis.simulator import setup_observatory_from_uvdata
    from healvis import sky_model as hvsm
    from healvis.observatory import Observatory

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
    **kwargs
        Passed through to :class:`~.simulators.VisibilitySimulator`.
    """

    point_source_ability = False
    diffuse_ability = True

    def __init__(self, fov=180, nprocesses=1, sky_ref_chan=0):
        if not HAVE_HEALVIS:
            raise ImportError("to use the healvis wrapper, you must install healvis!")

        self.fov = fov
        self._nprocs = nprocesses
        self._sky_ref_chan = sky_ref_chan

    def validate(self, model_data: ModelData):
        """Validate that all data is correct.

        In addition to standard parameter restrictions, HealVis requires a single beam
        for all antennae.
        """
        assert model_data.n_beams == 1

        # Check if pyuvsim.analyticbeam and switch to healvis.beam_model
        # TODO: we shouldn't silently modify model_data.beams...
        if isinstance(model_data.beams[0], pyuvsim.analyticbeam.AnalyticBeam):
            old_args = model_data.beams[0].__dict__

            gauss_width = None
            if old_args["type"] == "gaussian":
                if old_args["sigma"] is None:
                    raise NotImplementedError(
                        "Healvis does not permit gaussian beam with diameter."
                    )
                raise NotImplementedError(
                    "Healvis interprets gaussian beams "
                    "as per-baseline and not "
                    "per-antenna as required here."
                )

            beam_type = old_args["type"]
            spectral_index = old_args["spectral_index"]
            diameter = old_args["diameter"]
            model_data.beams = [
                AnalyticBeam(
                    beam_type=beam_type,
                    gauss_width=gauss_width,
                    diameter=diameter,
                    spectral_index=spectral_index,
                )
            ]

    def get_sky_model(self, sky_model: SkyModel):
        """
        A ``SkyModel`` compatible with healvis.

        Returns
        -------
        SkyModel object
            healvis SkyModel constructed from the input HEALPix sky
            model.
        """
        sky = hvsm.SkyModel()
        sky.Nside = sky_model.nside
        sky.freqs = sky_model.freq_array.to("Hz").value
        sky.Nskies = 1
        sky.ref_chan = self._sky_ref_chan

        # convert from Jy/sr to K
        if sky_model.stokes.unit == units.Jy / units.sr:
            intensity = 10 ** -26 * sky_model.stokes[0].T.value
            intensity *= (
                cnst.c.to("m/s").value / sky_model.freq_array.to("Hz").value
            ) ** 2 / (2 * cnst.k_B.value)
        elif sky_model.stokes.unit == units.K:
            intensity = sky_model.stokes[0].T.value
        else:
            raise ValueError(
                f"Units of {sky_model.stokes.unit} are not compatible with healvis"
            )
        sky.data = intensity[np.newaxis, :, :]
        sky._update()

        return sky

    def get_observatory(self, data_model: ModelData) -> Observatory:
        """
        A healvis :class:`healvis.observatory.Observatory` instance.

        Returns
        -------
        Observatory object
            healvis Observatory constructed from input telescope
            parameters.
        """
        return setup_observatory_from_uvdata(
            data_model.uvdata,
            fov=self.fov,
            beam=data_model.beams[0],
        )

    def simulate(self, data_model: ModelData):
        """
        Runs the healvis algorithm.

        Returns
        -------
        Visibility from all sources.
            Shape=self.uvdata.data_array.shape.
        """
        obs = self.get_observatory(data_model)
        sky = self.get_sky_model(data_model.sky_model)
        visibility = [
            obs.make_visibilities(sky, Nprocs=self._nprocs, beam_pol=pol)[0]
            for pol in data_model.uvdata.get_pols()
        ]
        visibility = np.moveaxis(visibility, 0, -1)

        return visibility[:, 0][:, np.newaxis, :, :]
