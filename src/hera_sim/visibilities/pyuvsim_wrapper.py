"""Wrapper for the pyuvsim simulator."""

import warnings

import numpy as np
import pyuvsim

from .simulators import ModelData, VisibilitySimulator


class UVSim(VisibilitySimulator):
    """A wrapper around the pyuvsim simulator.

    Parameters
    ----------
    quiet
        If True, don't print anything.
    """

    _blt_order_kws = {"order": "time", "minor_order": "baseline"}

    _functions_to_profile = (pyuvsim.uvsim.run_uvdata_uvsim,)

    def __init__(self, quiet: bool = True):
        self.quiet = quiet

    def simulate(self, data_model: ModelData):
        """Simulate the visibilities."""
        beam_dict = {
            ant: data_model.beam_ids[ant] for ant in data_model.uvdata.antenna_names
        }

        # TODO: this can be removed once
        # https://github.com/RadioAstronomySoftwareGroup/pyuvsim/pull/357
        # is mereged.
        if data_model.sky_model.name is not None:
            data_model.sky_model.name = np.array(data_model.sky_model.name)

        warnings.warn(
            "UVSim requires time-ordered data. Ensuring that order in UVData...",
            stacklevel=1,
        )
        data_model.uvdata.reorder_blts("time")

        # The UVData object must have correctly ordered pols.
        # TODO: either remove this when pyuvsim fixes bug with ordering
        # (https://github.com/RadioAstronomySoftwareGroup/pyuvsim/issues/370) or
        # at least check whether reordering is necessary once uvdata has that ability.
        if np.any(data_model.uvdata.polarization_array != np.array([-5, -6, -7, -8])):
            warnings.warn(
                "In UVSim, polarization array must be in AIPS order. Reordering...",
                stacklevel=1,
            )
            data_model.uvdata.reorder_pols("AIPS")

        out_uv = pyuvsim.uvsim.run_uvdata_uvsim(
            input_uv=data_model.uvdata,
            beam_list=data_model.beams,
            beam_dict=beam_dict,
            catalog=pyuvsim.simsetup.SkyModelData(data_model.sky_model),
            quiet=self.quiet,
        )
        return out_uv.data_array
