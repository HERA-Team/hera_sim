"""Wrapper for the pyuvsim simulator."""
import pyuvsim
from .simulators import VisibilitySimulator, ModelData
import numpy as np


class UVSim(VisibilitySimulator):
    """A wrapper around the pyuvsim simulator.

    Parameters
    ----------
    quiet
        If True, don't print anything.
    """

    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def simulate(self, data_model: ModelData):
        """Simulate the visibilities."""
        beam_dict = {
            ant: data_model.beam_ids[idx]
            for idx, ant in enumerate(data_model.uvdata.antenna_names)
        }

        # TODO: this can be removed once
        # https://github.com/RadioAstronomySoftwareGroup/pyuvsim/pull/357
        # is mereged.
        if data_model.sky_model.name is not None:
            data_model.sky_model.name = np.array(data_model.sky_model.name)

        out_uv = pyuvsim.uvsim.run_uvdata_uvsim(
            input_uv=data_model.uvdata,
            beam_list=data_model.beams,
            beam_dict=beam_dict,
            catalog=pyuvsim.simsetup.SkyModelData(data_model.sky_model),
            quiet=self.quiet,
        )
        return out_uv.data_array
