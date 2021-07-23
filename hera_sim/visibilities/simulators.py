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
from typing import List, Dict, Union
from pyradiosky import SkyModel
from pathlib import Path
from dataclasses import dataclass
import astropy_healpix as aph

BeamListType = Union[BeamList, List[Union[ab.AnalyticBeam, UVBeam]]]


class ModelData:
    """
    An object containing all the information required to perform visibility simulation.

    Parameters
    ----------
    uvdata
        A :class:`pyuvdata.UVData` object contain information about
        the "observation". If a path, must point to a UVData-readable file.
    sky_model
        A model for the sky to simulate.
    beams
        UVBeam models for as many antennae as have unique beams.
        Initialized from `obsparams`, if included. Defaults to a
        single uniform beam which is applied for every antenna. Each beam
        is the response of an individual antenna and NOT a per-baseline response.
        Shape=(N_BEAMS,).
    beam_ids
        List of integers specifying which beam model each antenna uses (i.e. the index
        of `beams` which it should refer to). Also accepts a dictionary in the format
        used by pyuvsim (i.e. ``antenna_name: index``), which is converted to such a
        list. By default, if one beam is given all antennas use the same beam, whereas
        if a beam is given per antenna, they are used in their given order.
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
        uvdata: UVData | str | Path,
        sky_model: SkyModel,
        beam_ids: Dict[str, int] | np.typing.ArrayLike[int] | None = None,
        beams: BeamListType | None = None,
    ):

        self.uvdata = self._validate_uvdata(uvdata)
        self.beams = [ab.AnalyticBeam("uniform")] if beams is None else beams

        # NOT Nants because we only want ants with data
        self.n_ant = self.uvdata.Nants_data

        if not isinstance(self.beams, BeamList):
            self.beams = BeamList(self.beams)

        self.beam_ids = self._process_beam_ids(beam_ids, self.beams)

        self.sky_model = sky_model
        self.sky_model.at_frequencies(self.freqs * units.Hz)
        assert isinstance(self.sky_model, SkyModel)

    def _validate_uvdata(self, uvdata: UVData | str | Path):
        if isinstance(uvdata, UVData):
            return uvdata
        elif isinstance(UVData, (str, Path)):
            out = UVData()
            out.read(uvdata)
            return out
        else:
            raise TypeError(
                "uvdata must be a UVData object or path to a compatible file."
            )

    def _process_beam_ids(
        self,
        beam_ids: Dict[str, int] | np.typing.ArrayLike[int] | None,
        beams: BeamList,
    ):
        # Set the beam_ids.
        if beam_ids is None:
            if len(beams) == 1:
                beam_ids = np.zeros(self.n_ant, dtype=int)
            elif len(beams) == self.n_ant:
                beam_ids = np.arange(self.n_ant, dtype=int)
            else:
                raise ValueError(
                    "Need to give beam_ids if beams is given and not one per ant."
                )
        elif isinstance(beam_ids, dict):
            beam_ids = np.array(list(beam_ids.values()), dtype=int)
        else:
            beam_ids = np.array(beam_ids, dtype=int)

        assert beam_ids.max() < len(beams)
        assert len(beam_ids) == self.n_ant

        return beam_ids

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
        """An orderd list of active antenna numbers."""
        return self.uvdata.get_ants()

    @cached_property
    def active_antpos(self) -> np.ndarray:
        """Positions of active antennas."""
        return self.uvdata.get_ENU_antpos()[0][self.ant_list]


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
            and sky_model.component_type == "point"
        ):
            self.data_model.sky_model = self._convert_point_to_healpix(sky_model)

    def _convert_point_to_healpix(self, sky_model) -> SkyModel:
        # TODO: update this to just use SkyModel native functionality when available
        npix = aph.nside_to_npix(self.n_side)
        hmap = np.zeros((len(sky_model.freq_array), npix)) * units.Jy / units.sr

        # Get which pixel every point source lies in.
        pix = aph.lonlat_to_healpix(
            lon=sky_model.ra,
            lat=sky_model.dec,
            nside=self.n_side,
        )

        hmap[:, pix] += sky_model.stokes[0].to("Jy") / aph.nside_to_pixel_area(
            self.n_side
        )

        return SkyModel(
            stokes=np.array(
                [
                    hmap.value,
                    np.zeros_like(hmap),
                    np.zeros_like(hmap),
                    np.zeros_like(hmap),
                ]
            )
            * units.Jy
            / units.sr,
            component_type="healpix",
            nside=self.n_side,
            hpx_inds=np.arange(npix),
            spectral_type="full",
            freq_array=sky_model.freq_array,
        )

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


class VisibilitySimulator(metaclass=ABCMeta):
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
