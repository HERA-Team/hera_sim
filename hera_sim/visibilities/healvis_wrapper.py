"""Wrapper for healvis so that it accepts pyuvsim configuration inputs."""
from __future__ import division

import numpy as np

import pyuvsim
import warnings
from .simulators import VisibilitySimulator, ModelData, SkyModel

from astropy import constants as cnst
from astropy import units

try:
    from healvis.beam_model import AnalyticBeam
    from healvis.simulator import setup_observatory_from_uvdata
    from healvis import sky_model as hvsm
    from healvis.observatory import Observatory
    import healvis as hv

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
    __version__ = hv.__version__

    def __init__(self, fov=180, nprocesses=1, sky_ref_chan=0):
        if not HAVE_HEALVIS:
            raise ImportError("to use the healvis wrapper, you must install healvis!")

        warnings.warn(
            (
                "The healvis package is deprecated. Please use pyuvsim instead. "
                "The healvis wrapper will be removed from hera_sim in version 4",
            ),
            category=DeprecationWarning,
        )

        self.fov = fov
        self._nprocs = nprocesses
        self._sky_ref_chan = sky_ref_chan

    def validate(self, model_data: ModelData):
        """Validate that all data is correct.

        In addition to standard parameter restrictions, HealVis requires a single beam
        for all antennae.
        """
        if model_data.n_beams > 1:
            raise ValueError("healvis must use the same beam for all antennas.")

        # Check if pyuvsim.analyticbeam and switch to healvis.beam_model
        # TODO: we shouldn't silently modify model_data.beams...
        if isinstance(model_data.beams[0], pyuvsim.analyticbeam.AnalyticBeam):
            warnings.warn(
                "Using pyuvsim.AnalyticBeam for healvis is not really supported. "
                "model_data.beams is being automatically modified to be a single "
                "healvis.AnalyticBeam of the same type."
            )
            old_args = model_data.beams[0].__dict__

            if old_args["type"] == "gaussian":
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
                    gauss_width=None,
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
        if sky_model.stokes.unit.is_equivalent(units.Jy / units.sr):
            conversion = (
                1e-26
                * sky_model.stokes.unit.to(units.Jy / units.sr)
                * (cnst.c.si.value / sky_model.freq_array.si.value) ** 2
                / (2 * cnst.k_B.si.value)
            )
        elif sky_model.stokes.unit.is_equivalent(units.K):
            conversion = sky_model.stokes.unit.to("K")
        else:
            raise ValueError(
                f"Units of {sky_model.stokes.unit} are not compatible with healvis"
            )
        intensity = conversion * sky_model.stokes[0].T.value

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
        visibilities = []

        # Simulate the visibilities for each polarization.
        for pol in data_model.uvdata.get_pols():
            if pol in ["xx", "yy"]:
                visibility, _, baselines = obs.make_visibilities(
                    sky, Nprocs=self._nprocs, beam_pol=pol
                )  # Shape (Nblts, Nskies, Nfreqs)
            else:
                # healvis doesn't support polarization
                visibility = np.zeros(
                    (data_model.uvdata.Nblts, 1, data_model.uvdata.Nfreqs)
                )

            # AnalyticBeams do not use polarization at all in healvis, and are
            # equivalent to "pI" polarization. To match our definition of linear pols
            # we divide by two if this is the case.
            if isinstance(data_model.beams[0], AnalyticBeam) and "p" not in pol:
                visibility /= 2
            visibilities.append(visibility[:, 0, :][:, np.newaxis, :])

        # Transform from shape (Npols, Nblts, 1, Nfreqs) to  (Nblts, 1, Nfreqs, Npols).
        visibilities = np.moveaxis(visibilities, 0, -1)

        # Now get the blt-order correct. healvis constructs the observatory such
        # that the baselines are sorted in order of increasing baseline integer. So
        # to get the mapping right, we need to first get the unique baseline integers
        # sorted in increasing order. This doesn't necessarily match the order of the
        # data array, so we need to reorder the simulated data so it does match.
        vis = np.zeros_like(data_model.uvdata.data_array)
        unique_bls = list(np.unique(data_model.uvdata.baseline_array))
        for ai, aj in data_model.uvdata.get_antpairs():
            # First, retrieve the integer for the current baseline.
            baseline = data_model.uvdata.antnums_to_baseline(ai, aj)
            # Then, find out where the baseline sits in the ordered list.
            baseline_indx = unique_bls.index(baseline)
            # ``baselines`` is sorted the same way as the visibilities, so this gives
            # the visibilities simulated for this baseline in the simulation data.
            sim_indx = np.argwhere(baselines == baseline_indx).flatten()
            # This gives us the slice of the data array where this baseline lives.
            data_indx = data_model.uvdata.antpair2ind(ai, aj)
            # Finally, put the simulated data into the data array in the right order.
            vis[data_indx, ...] = visibilities[sim_indx, ...]

        return vis
