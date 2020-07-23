from __future__ import division
from builtins import object
import warnings

import healpy
import numpy as np
from cached_property import cached_property
from pyuvsim import analyticbeam as ab
from pyuvsim.simsetup import (
    initialize_uvdata_from_params,
    initialize_catalog_from_params,
    uvdata_to_telescope_config,
    _complete_uvdata
)
from os import path
from abc import ABCMeta, abstractmethod
from astropy import units

class VisibilitySimulator(object):
    __metaclass__ = ABCMeta
    """
    Base VisibilitySimulator class.

    Any actual visibility simulator should be sub-classed from this one.
    This class provides several convenience methods and defines the API.
    """

    # Whether this particular simulator has the ability to simulate point
    # sources directly.
    point_source_ability = True

    # Whether this particular simulator has the ability to simulate diffuse
    # maps directly.
    diffuse_ability = True

    def __init__(self, obsparams=None, uvdata=None, sky_freqs=None,
                 beams=None, beam_ids=None, sky_intensity=None,
                 point_source_pos=None, point_source_flux=None, nside=2**5):
        """
        Parameters
        ----------
        obsparams : dict or filepath, optional
            Exactly the expected input to `pyuvsim`'s
            :func:`pyuvsim.simsetup.initialize_uvdata_from_params`
            function. By default `uvdata`, `beams`, and `beam_ids`
            are used instead.
        uvdata : UVData object, optional
            A :class:`pyuvdata.UVData` object contain information about
            the "observation". Initalized from `obsparams`, if included.
        sky_freqs : array_like, optional
            Frequencies at which the sky intensity and/or point sources
            are defined in [Hz]. Defaults to the unique frequencies in
            `uvdata` Shape=(NFREQS,).
        beams : array_like of `pyuvsim.analyticbeam.AnalyticBeam`,
                optional
            UVBeam models for as many antennae as have unique beams.
            Initialized from `obsparams`, if included. Defaults to a
            single uniform beam is applied for every antenna. Each beam
            is the response of an individual antenna and NOT a 
            per-baseline response.
            Shape=(N_BEAMS,).
        beam_ids : array_like of int, optional
            List of integers specifying which beam model each antenna
            uses (i.e. the index of `beams` which it should refer to).
            Initialized from `obsparams`, if included. By default, all
            antennas use the same beam (beam 0).
            Shape=(N_ANTS,).
        sky_intensity : array_like, optional
            A healpix model for the intensity of the sky emission, in
            [Jy/sr]. Shape=(NFREQS, N_PIX_SKY).
        point_source_pos : array_like, optional
            An array of point sources. For each source, the entries are
            (ra, dec) [rad] (assumed to be in J2000).
            Shape=(N_SOURCES, 2).
        point_source_flux : array_like, optional
            An array of fluxes of the given point sources, per
            frequency. Fluxes in [Jy]. Shape=(NFREQS, N_SOURCES).
        nside : int, optional
            Only used if sky_intensity is *not* given but the simulator
            is incapable of directly dealing with point sources. In this
            case, it sets the resolution of the healpix map to which the
            sources will be allocated.
            
        Notes
        -----
            Input beam models represent the responses of individual
            antennas and are NOT the same as per-baseline "primary
            beams". This interpretation of a "primary beam" would be the
            product of the responses of two input antenna beams. 
        """
        if obsparams:
            (self.uvdata,
             self.beams,
             self.beam_ids) = initialize_uvdata_from_params(obsparams)

            if point_source_pos is None:
                try:
                    # Try setting up point sources from the obsparams.
                    # Will only work, of course, if the "catalog" key is in obsparams['sources'].
                    # If it's not there, it will raise a KeyError.
                    catalog = initialize_catalog_from_params(obsparams, return_recarray=False)[0]
                    catalog.at_frequencies(np.unique(self.uvdata.freq_array) * units.Hz)

                    try:
                        point_source_pos = np.array([catalog.ra.rad, catalog.dec.rad]).T
                        # This gets the 'I' component of the flux density
                        point_source_flux = np.atleast_2d(catalog.stokes[0].to('Jy').value).T
                    except units.UnitConversionError:
                        # If the catalog is healpix, converting Stokes 'I' to 'Jy' will give
                        # `UnitConversionError`. Then, we get the 'I' component as sky intensity.
                        sky_intensity = np.atleast_2d(catalog.stokes[0].to('K').value).T
                except KeyError:
                    # If 'catalog' was not defined in obsparams, that's fine. We assume
                    # the user has passed some sky model directly (we'll catch it later).
                    pass

            # convert the beam_ids dict to an array of ints
            nms = list(self.uvdata.antenna_names)
            tmp_ids = np.zeros(len(self.beam_ids), dtype=int)
            for name, id in self.beam_ids.items():
                tmp_ids[nms.index(name)] = id
            self.beam_ids = tmp_ids
            self.beams.set_obj_mode()
            _complete_uvdata(self.uvdata, inplace=True)
        else:
            if uvdata is None:
                raise ValueError("if obsparams is not given, uvdata must be.")

            self.uvdata = uvdata

            self.beams = [ab.AnalyticBeam("uniform")] if beams is None else beams
            if beam_ids is None:
                self.beam_ids = np.zeros(self.n_ant, dtype=np.int)
            else:
                self.beam_ids = beam_ids

        self._nside = nside
        self.sky_intensity = sky_intensity

        if sky_freqs is None:
            self.sky_freqs = np.unique(self.uvdata.freq_array)
        else:
            self.sky_freqs = sky_freqs

        self.point_source_pos = point_source_pos
        self.point_source_flux = point_source_flux

        self.validate()

    def validate(self):
        """Checks for correct input format."""
        if (self.point_source_pos is None) != (self.point_source_flux is None):
            raise ValueError("Either both or neither of point_source_pos and "
                             "point_source_flux must be given.")

        if not (self.sky_intensity is None or healpy.isnpixok(self.n_pix)):
            raise ValueError("The sky_intensity map is not compatible with "
                             "healpy.")

        if self.point_source_pos is None and self.sky_intensity is None:
            raise ValueError("You must pass at least one of sky_intensity or "
                             "point_sources.")

        if np.max(self.beam_ids) >= self.n_beams:
            raise ValueError("The number of beams provided must be at least "
                             "as great as the greatest beam_id.")

        if (
            self.point_source_flux is not None
            and self.point_source_flux.shape[0] != self.sky_freqs.shape[0]
        ):
            if self.point_source_flux.shape[0] == 1:
                self.point_source_flux = np.repeat(self.point_source_flux, self.sky_freqs.shape[0]).reshape((self.sky_freqs.shape[0], -1))
            else:
                raise ValueError(
                    f"point_source_flux must have the same number of freqs as sky_freqs. "
                    f"point_source_flux.shape = {self.point_source_flux.shape}."
                    f"sky_freq.shape = {self.sky_freqs.shape}"
                )

        if self.point_source_flux is not None:
            flux_shape = self.point_source_flux.shape
            pos_shape = self.point_source_pos.shape
            if flux_shape[1] != pos_shape[0]:
                raise ValueError("Number of sources in point_source_flux and "
                                 "point_source_pos is different.")

        if not (
            self.sky_intensity is None
            or self.sky_intensity.shape[0] == self.sky_freqs.shape[0]
        ):
            raise ValueError("sky_intensity has a different number of freqs "
                             "than sky_freqs.")

        if self.sky_intensity is not None and self.sky_intensity.ndim != 2:
            raise ValueError("sky_intensity must be a 2D array (a healpix map "
                             "per frequency).")

        if not (self.point_source_ability or self.point_source_pos is None):
            warnings.warn("This visibility simulator is unable to explicitly "
                          "simulate point sources. Adding point sources to "
                          "diffuse pixels.")
            if self.sky_intensity is None:
                self.sky_intensity = 0
            self.sky_intensity += self.convert_point_sources_to_healpix(
                self.point_source_pos, self.point_source_flux, self.nside
            )

        if not (self.diffuse_ability or self.sky_intensity is None):
            warnings.warn("This visibility simulator is unable to explicitly "
                          "simulate diffuse structure. Converting diffuse "
                          "intensity to approximate points.")

            (pos,
             flux) = self.convert_healpix_to_point_sources(self.sky_intensity)

            if self.point_source_pos is None:
                self.point_source_pos = pos
                self.point_source_flux = flux
            else:
                self.point_source_flux = \
                    np.hstack((self.point_source_flux, flux))
                self.point_source_pos = np.hstack((self.point_source_pos, pos))

            self.sky_intensity = None

    @staticmethod
    def convert_point_sources_to_healpix(point_source_pos, point_source_flux,
                                         nside=2**5):
        """
        Convert point sources to an approximate diffuse HEALPix model.

        The healpix map returned is in RING scheme.

        Parameters
        ----------
        point_source_pos : array_like
            An array of point sources. For each source, the entries are
            (ra, dec) [rad] (assumed to be in J2000).
            Shape=(N_SOURCES, 2).
        point_source_flux : array_like
            point_source_flux : array_like, optional
            An array of fluxes of the given point sources, per
            frequency. Fluxes in [Jy]. Shape=(NFREQS, N_SOURCES).
        nside : int, optional
            HEALPix nside parameter (must be a power of 2).

        Returns
        -------
        array_like
            The HEALPix diffuse model. Shape=(NFREQ, NPIX).
        """

        hmap = np.zeros((len(point_source_flux), healpy.nside2npix(nside)))

        # Get which pixel every point source lies in.
        pix = healpy.ang2pix(nside, np.pi/2 - point_source_pos[:, 1],
                             point_source_pos[:, 0])

        hmap[:, pix] += point_source_flux / healpy.nside2pixarea(nside)

        return hmap

    @staticmethod
    def convert_healpix_to_point_sources(hmap):
        """
        Convert a HEALPix map to a set of point sources.

        The point sources are placed at the center of each pixel.

        Parameters
        ----------
        hmap : array_like
            The HEALPix map. Shape=(NFREQ, NPIX).

        Returns
        -------
            array_like
                The point source approximation. Positions in (ra, dec) (J2000).
                Shape=(N_SOURCES, 2). Fluxes in [Jy]. Shape=(NFREQ, N_SOURCES).
        """
        nside = healpy.get_nside(hmap[0])
        ra, dec = healpy.pix2ang(nside, np.arange(len(hmap[0])), lonlat=True)
        flux = hmap * healpy.nside2pixarea(nside)
        return np.array([ra*np.pi/180, dec*np.pi/180]).T, flux

    def simulate(self):
        """Perform the visibility simulation."""
        self._write_history()
        vis = self._simulate()
        self.uvdata.data_array += vis
        return vis

    @abstractmethod
    def _simulate(self):
        """Subclass-specific simulation method, to be overwritten."""
        pass

    @property
    def nside(self):
        """Nside parameter of the sky healpix map."""
        try:
            return healpy.get_nside(self.sky_intensity[0])
        except TypeError:

            if not healpy.isnsideok(self._nside):
                raise ValueError("nside must be a power of 2")

            return self._nside

    @cached_property
    def n_pix(self):
        """Number of pixels in the sky map."""
        return self.sky_intensity.shape[1]

    @cached_property
    def n_ant(self):
        """Number of antennas in array."""
        return self.uvdata.get_ants().shape[0]

    @cached_property
    def n_beams(self):
        """Number of beam models used."""
        return len(self.beams)

    def _write_history(self):
        """Write pertinent details of simulation to the UVData's history."""
        class_name = self.__class__.__name__
        self.uvdata.history += ("Visibility Simulation performed with "
                                "hera_sim's {} simulator\n").format(class_name)
        self.uvdata.history += "Class Repr: {}".format(repr(self))

    def write_config_file(self, filename, direc='.', beam_filepath=None,
                          antenna_layout_path=None):
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
            antenna_layout_path = (path.basename(filename)
                                   + "_antenna_layout.csv")

        uvdata_to_telescope_config(
            self.uvdata, beam_filepath=beam_filepath,
            layout_csv_name=antenna_layout_path,
            telescope_config_name=filename, return_names=False, path_out=direc
        )
