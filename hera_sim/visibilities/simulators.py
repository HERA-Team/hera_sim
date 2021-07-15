"""Module defining a high-level visibility simulator wrapper."""
from __future__ import division, annotations

import numpy as np
from cached_property import cached_property
from pyuvsim import analyticbeam as ab, BeamList
from pyuvsim.simsetup import (
    initialize_uvdata_from_params,
    initialize_catalog_from_params,
    uvdata_to_telescope_config,
    _complete_uvdata,
)
from os import path
from abc import ABCMeta, abstractmethod
from astropy import units
from pyuvdata import UVData, UVBeam
from typing import List, Dict
from pyradiosky import SkyModel
from pathlib import Path
from dataclasses import dataclass
import astropy_healpix as aph


def _is_power_of_2(n):
    # checking if a power of 2 from https://stackoverflow.com/a/57025941/1467820
    return (n & (n - 1) == 0) and n != 0


def _isnpixok(npix):
    n = npix // 12
    return npix % 12 == 0 and _is_power_of_2(n)


BeamListType = BeamList | List[ab.AnalyticBeam | UVBeam]


class ModelData:
    """
    An object containing all the information required to perform visibility simulation.

    Parameters
    ----------
    uvdata
        A :class:`pyuvdata.UVData` object contain information about
        the "observation". Initalized from `obsparams`, if included.
    sky_model
        A model for the sky to simulate.
    beams
        UVBeam models for as many antennae as have unique beams.
        Initialized from `obsparams`, if included. Defaults to a
        single uniform beam is applied for every antenna. Each beam
        is the response of an individual antenna and NOT a
        per-baseline response.
        Shape=(N_BEAMS,).
    beam_ids
        List of integers specifying which beam model each antenna
        uses (i.e. the index of `beams` which it should refer to).
        Initialized from `obsparams`, if included. By default, all
        antennas use the same beam (beam 0).
        Shape=(N_ANTS,).

    Notes
    -----
    Input beam models represent the responses of individual
    antennas and are NOT the same as per-baseline "primary
    beams". This interpretation of a "primary beam" would be the
    product of the responses of two input antenna beams.
    """

    def __init__(
        self,
        *,
        uvdata: UVData,
        sky_model: SkyModel,
        beam_ids: Dict[str, int] | None = None,
        beams: BeamListType | None = None,
    ):

        self.uvdata = uvdata
        self.beams = [ab.AnalyticBeam("uniform")] if beams is None else beams
        self.n_ant = (
            self.uvdata.Nants_data
        )  # NOT Nants because we only want ants with data

        assert isinstance(self.uvdata, UVData)
        assert isinstance(self.beams, (list, BeamList))

        # Set the beam_ids.
        if beam_ids is None:
            if len(self.beams) == 1:
                self.beam_ids = np.zeros(self.n_ant, dtype=int)
            elif len(self.beams) == self.n_ant:
                self.beam_ids = np.arange(self.n_ant, dtype=int)
            else:
                raise ValueError(
                    "Need to give beam_ids if beams is given and not one per ant."
                )
        else:
            self.beam_ids = np.array(beam_ids, dtype=int)

        assert isinstance(self.beam_ids, np.ndarray)
        assert self.beam_ids.dtype == int
        assert self.beam_ids.max < (self.n_beams - 1)
        assert len(self.beam_ids) == self.n_ant

        self.sky_model = sky_model
        assert isinstance(self.sky_model, SkyModel)

    @classmethod
    def from_config(cls, config_file: str | Path) -> ModelData:
        """Initialize the :class:`ModelData` from a pyuvsim-compatible config."""
        uvdata, beams, beam_ids = initialize_uvdata_from_params(config_file)
        catalog = initialize_catalog_from_params(config_file, return_recarray=False)[0]
        catalog.at_frequencies(np.unique(uvdata.freq_array) * units.Hz)

        # convert the beam_ids dict to an array of ints
        nms = list(uvdata.antenna_names)
        tmp_ids = np.zeros(len(beam_ids), dtype=int)
        for name, beam_id in beam_ids.items():
            tmp_ids[nms.index(name)] = beam_id
        beam_ids = tmp_ids
        beams.set_obj_mode()
        _complete_uvdata(uvdata, inplace=True)

        return ModelData(
            uvdata=uvdata,
            beams=beams,
            beam_ids=beam_ids,
            sky_model=catalog,
        )

    @cached_property
    def lsts(self):
        """The LSTs at which data is defined."""
        return self.uvdata.lst_array[:: self.uvdata.Nbls]

    @cached_property
    def freqs(self) -> np.ndarray:
        """Frequnecies at which data is defined."""
        return self.uvdata.freq_array[0]

    @cached_property
    def n_beams(self) -> int:
        """Number of beam models used."""
        return len(self.beams)

    def write_config_file(
        self, filename, direc=".", beam_filepath=None, antenna_layout_path=None
    ):
        """
        Writes a YAML config file corresponding to the current UVData object.

        Parameters
        ----------
        filename : str
            Filename of the config file.
        direc : str
            Directory in which to place the config file and its
            supporting files.
        beam_filepath : str, optional
            Where to put the beam information. Default is to place it alongside
            the config file, but with extension '.beams'.
        antenna_layout_path : str, optional
            Where to put the antenna layout CSV file. Default is alongside the
            main config file, but appended with '_antenna_layout.csv'.
        """
        if beam_filepath is None:
            beam_filepath = path.basename(filename) + ".beams"

        if antenna_layout_path is None:
            antenna_layout_path = path.basename(filename) + "_antenna_layout.csv"

        uvdata_to_telescope_config(
            self.uvdata,
            beam_filepath=beam_filepath,
            layout_csv_name=antenna_layout_path,
            telescope_config_name=filename,
            return_names=False,
            path_out=direc,
        )

    @cached_property
    def ant_list(self) -> np.ndarray:
        """An orderd list of active antennas."""
        # Get antpos for active antennas only
        # self.antpos = self.uvdata.get_ENU_antpos()[0].astype(self._real_dtype)
        return self.uvdata.get_ants()  # ordered list of active ants

    @cached_property
    def active_antpos(self) -> np.ndarray:
        """Positions of active antennas."""
        antpos = []
        _antpos = self.uvdata.get_ENU_antpos()[0]
        for ant in self.ant_list:
            # uvdata.get_ENU_antpos() and uvdata.antenna_numbers have entries
            # for all telescope antennas, even ones that aren't included in the
            # data_array. This extracts only the data antennas.
            idx = np.where(ant == self.uvdata.antenna_numbers)
            antpos.append(_antpos[idx].flatten())
        return np.array(antpos)


@dataclass
class VisibilitySimulation:
    """An object representing a visibility simulation, including data and simulator."""

    data_model: ModelData
    simulator: VisibilitySimulator
    n_side: int = 2 ** 5

    def __post_init__(self):
        """Perform simple validation on combined attributes."""
        self.simulator.validate(self.data_model)

        # Convert the sky model to either point source or healpix depending on the
        # simulator's capabilities.
        sky_model = self.data_model.sky_model
        if not self.simulator.diffuse_ability and sky_model.component_type == "healpix":
            sky_model.healpix_to_point()
        if (
            not self.simulator.point_source_ability
            and self.data_model.component_type == "point"
        ):
            sky_model.n_side = self.n_side
            sky_model.hpx_inds = np.arange(aph.nside_to_npix, dtype=int)
            sky_model.point_to_healpix()

    def _write_history(self):
        """Write pertinent details of simulation to the UVData's history."""
        class_name = self.simulator.__class__.__name__
        self.uvdata.history += (
            f"Visibility Simulation performed with hera_sim's {class_name} simulator\n"
        )
        self.uvdata.history += f"Class Repr: {repr(self.simulator)}"

    def simulate(self):
        """Perform the visibility simulation."""
        self._write_history()
        vis = self.simulator.simulate(self.data_model)
        self.uvdata.data_array += vis
        return vis

    @property
    def uvdata(self) -> UVData:
        """A simple view into the UVData object in the :attr:`data_model`."""
        return self.data_model.uvdata


class VisibilitySimulator(meta=ABCMeta):
    """Base class for all hera_sim compatible visibility simulators."""

    #: Whether this particular simulator has the ability to simulate point
    #: sources directly.
    point_source_ability = True

    #: Whether this particular simulator has the ability to simulate diffuse
    #: maps directly.
    diffuse_ability = False

    @abstractmethod
    def simulate(self, data_model: ModelData) -> np.ndarray:
        """Simulate the visibilities."""
        pass

    def validate(self, data_model: ModelData):
        """Check that the data model complies with the assumptions of the simulator."""
        pass
