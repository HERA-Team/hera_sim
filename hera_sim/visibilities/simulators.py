"""Module defining a high-level visibility simulator wrapper."""
from __future__ import annotations

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
from typing import List, Dict, Union, Sequence
from pyradiosky import SkyModel
from pathlib import Path
from dataclasses import dataclass
import astropy_healpix as aph
from .. import __version__
from .. import visibilities as vis
import importlib
import yaml

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
    normalize_beams
        Whether to peak-normalize the beams. This removes the bandpass from the beams'
        data arrays and moves it into their ``bandpass_array`` attributes.

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
        beam_ids: Dict[str, int] | Sequence[int] | None = None,
        beams: BeamListType | None = None,
        normalize_beams: bool = False,
    ):

        self.uvdata = self._process_uvdata(uvdata)

        # NOT Nants because we only want ants with data
        self.n_ant = self.uvdata.Nants_data

        self.beams = self._process_beams(beams, normalize_beams)
        self.beam_ids = self._process_beam_ids(beam_ids, self.beams)
        self._validate_beam_ids(self.beam_ids, self.beams)

        self.sky_model = sky_model
        self.sky_model.at_frequencies(self.freqs * units.Hz)
        if not isinstance(self.sky_model, SkyModel):
            raise TypeError("sky_model must be a SkyModel instance.")

        self._validate()

    def _process_uvdata(self, uvdata: UVData | str | Path):

        if isinstance(uvdata, (str, Path)):
            out = UVData()
            out.read(str(uvdata))
            uvdata = out

        if not isinstance(uvdata, UVData):
            raise TypeError(
                "uvdata must be a UVData object or path to a compatible file. Got "
                f"{uvdata}, type {type(uvdata)}"
            )

        # Temporary fix for future array shape - to be removed after v3.
        if uvdata.future_array_shapes:
            uvdata.use_current_array_shapes()

        return uvdata

    @classmethod
    def _process_beams(cls, beams: BeamListType | None, normalize_beams: bool):
        if beams is None:
            beams = [ab.AnalyticBeam("uniform")]

        if not isinstance(beams, BeamList):
            beams = BeamList(beams)

        if beams.string_mode:
            beams.set_obj_mode()

        if len({beam.beam_type for beam in beams}) != 1:
            # TODO: replace with beam.check_consistency() when that is available in
            # pyuvsim.
            raise ValueError("All beams must be of the same beam_type!")

        if normalize_beams:
            for beam in beams:
                if beam.data_normalization != "peak":
                    beam.peak_normalize()

        return beams

    def _process_beam_ids(
        self,
        beam_ids: Dict[str, int] | np.typing.ArrayLike[int] | None,
        beams: BeamList,
    ) -> Dict[str, int]:
        # beam ids maps antenna name to INDEX of the beam in the beam list.

        # Set the beam_ids.
        if beam_ids is None:
            if len(beams) == 1:
                beam_ids = {nm: 0 for nm in self.uvdata.antenna_names}
            elif len(beams) == self.n_ant:
                beam_ids = {nm: i for i, nm in enumerate(self.uvdata.antenna_names)}
            else:
                raise ValueError(
                    "Need to give beam_ids if beams is given and not one per ant."
                )
        elif isinstance(beam_ids, (list, tuple, np.ndarray)):
            if len(beam_ids) != self.n_ant:
                raise ValueError("Number of beam_ids given must match n_ant")

            beam_ids = {
                nm: int(beam_ids[i]) for i, nm in enumerate(self.uvdata.antenna_names)
            }
        elif not isinstance(beam_ids, dict):
            raise TypeError("beam_ids should be a dict or sequence of integers")

        return beam_ids

    def _validate_beam_ids(self, beam_ids, beams):
        if max(beam_ids.values()) >= len(beams):
            raise ValueError(
                "There is at least one beam_id that points to a non-existent beam. "
                f"Number of given beams={len(beams)} but maximum"
                f" beam_id={max(beam_ids.values())}."
            )

        if len(beam_ids) != self.n_ant:
            raise ValueError(
                f"Length of beam_ids ({len(beam_ids)}) must match the "
                f"number of ants ({self.n_ant})."
            )

    @classmethod
    def from_config(
        cls, config_file: str | Path, normalize_beams: bool = False
    ) -> ModelData:
        """Initialize the :class:`ModelData` from a pyuvsim-compatible config."""
        uvdata, beams, beam_ids = initialize_uvdata_from_params(config_file)
        catalog = initialize_catalog_from_params(config_file, return_recarray=False)[0]

        _complete_uvdata(uvdata, inplace=True)

        return ModelData(
            uvdata=uvdata,
            beams=beams,
            beam_ids=beam_ids,
            sky_model=catalog,
            normalize_beams=normalize_beams,
        )

    @cached_property
    def lsts(self):
        """Local Sidereal Times in radians."""
        # This process retrieves the unique LSTs while respecting phase wraps.
        _, unique_inds = np.unique(self.uvdata.time_array, return_index=True)
        return self.uvdata.lst_array[unique_inds]

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

    def _validate(self):
        """Perform validation of the full ModelData instance.

        The idea here is to validate the combination of inputs -- uvdata, uvbeam list
        and sky model, checking for inconsistencies that would be wrong for _any_
        simulator.
        """
        if any(b.beam_type == "power" for b in self.beams) and np.any(
            self.sky_model.stokes[1:] != 0
        ):
            raise TypeError(
                "Cannot use power beams when the sky model contains polarized sources."
            )


@dataclass
class VisibilitySimulation:
    """An object representing a visibility simulation, including data and simulator."""

    data_model: ModelData
    simulator: VisibilitySimulator
    n_side: int = 2**5

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
        self.uvdata.history += f"Class Repr: {repr(self.simulator)}\n"
        self.uvdata.history += f"hera_sim version: {__version__}"
        self.uvdata.history += f"Simulator Version: {self.simulator.__version__}"

    def simulate(self):
        """Perform the visibility simulation."""
        # Order the baselines/times in the order expected by the simulator.
        vis = self.simulator.simulate(self.data_model)

        self.uvdata.data_array += vis
        self._write_history()

        return vis

    @property
    def uvdata(self) -> UVData:
        """A simple view into the UVData object in the :attr:`data_model`."""
        return self.data_model.uvdata


class VisibilitySimulator(metaclass=ABCMeta):
    """Base class for all hera_sim-compatible visibility simulators.

    To define a new simulator, make a subclass. The subclass should overwrite available
    class-attributes as necessary, and specify a ``__version__`` of the simulator code
    itself.

    The :meth:`simulate` abstract method *must* be overwritten in the subclass, to
    perform the actual simulation. The :meth:`validate` method *may* also be
    overwritten to validate the given `UVData` input for the particular simulator.

    The subclass may define any number of simulator-specific parameters as part of its
    init method.

    Finally, to enable constructing the simulator in command-line applications, a
    :meth:`from_yaml` method is provided. This will load a YAML file's contents as a
    dictionary, and then instantiate the subclass with the parameters in that dict.
    To enable some control over this process, the subclass can overwrite the
    :meth:`_from_yaml_dict` private method, which takes in the dictionary read from the
    YAML file, and transforms any necessary parameters before constructing the class.
    For example, if the class required a set of data from a file, the YAML might contain
    the filename itself, and in :meth:`_from_yaml_dict`, the file would be read and the
    data itself passed to the constructor.
    """

    #: Whether this particular simulator has the ability to simulate point
    #: sources directly.
    point_source_ability = True

    #: Whether this particular simulator has the ability to simulate diffuse
    #: maps directly.
    diffuse_ability = False

    __version__ = "unknown"

    @abstractmethod
    def simulate(self, data_model: ModelData) -> np.ndarray:
        """Simulate the visibilities."""
        pass

    def validate(self, data_model: ModelData):
        """Check that the data model complies with the assumptions of the simulator."""
        pass

    @classmethod
    def from_yaml(cls, yaml_config: dict | str | Path) -> VisibilitySimulator:
        """Generate the simulator from a YAML file or dictionary."""
        if not isinstance(yaml_config, dict):
            with open(yaml_config, "r") as fl:
                yaml_config = yaml.safe_load(fl)

        # In general, we allow to specify which simulator to use in the config,
        # but that shouldn't be passed on to the constructor of a particular simulator.
        if "simulator" in yaml_config:
            del yaml_config["simulator"]

        return cls._from_yaml_dict(yaml_config)

    @classmethod
    def _from_yaml_dict(cls, cfg: dict) -> VisibilitySimulator:
        """Generate the simulator from a dictionary read from YAML.

        This method should be overloaded in subclasses if class generation is more
        complex than simply setting parameters from the dictionary.
        """
        return cls(**cfg)


def load_simulator_from_yaml(config: Path | str) -> VisibilitySimulator:
    """Construct a visibility simulator from a YAML file."""
    with open(config, "r") as fl:
        cfg = yaml.safe_load(fl)

    simulator_cls = cfg.pop("simulator")

    if "." not in simulator_cls:
        # Use a built-in simulator
        try:
            simulator_cls = getattr(vis, simulator_cls)
        except AttributeError:
            raise AttributeError(
                f"The given simulator '{simulator_cls}' is not available in hera_sim."
            )
    else:  # pragma: nocover
        module = ".".join(simulator_cls.split(".")[:-1])
        module = importlib.import_module(module)
        simulator_cls = getattr(module, simulator_cls.split(".")[-1])

    if not issubclass(simulator_cls, VisibilitySimulator):
        raise TypeError(
            f"Specified simulator {simulator_cls} is not a subclass of"
            "VisibilitySimulator!"
        )

    assert issubclass(simulator_cls, VisibilitySimulator)
    return simulator_cls.from_yaml(cfg)
