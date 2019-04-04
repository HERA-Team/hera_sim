import warnings

import healpy
import numpy as np
from cached_property import cached_property
from pyuvsim import analyticbeam as ab
from pyuvsim.simsetup import initialize_uvdata_from_params, uvdata_to_config_file


class VisibilitySimulator(object):
    # Whether this particular simulator has the ability to simulate point sources
    # directly
    _point_source_ability = True

    # Whether this particular simulator has the ability to simulate diffuse maps
    # directly
    _diffuse_ability = True

    def __init__(self, obsparams=None, uvdata=None, sky_freqs=None, beams=None,
                 beam_ids=None,
                 sky_intensity=None, point_source_pos=None, point_source_flux=None,
                 nside=50, ):
        """
        Base VisibilitySimulator class.

        Any actual visibility simulator should be sub-classed from this one.
        This class provides several convenience methods and defines the API.

        Args:
            obsparams (dict or filepath):
                Exactly the expected input to `pyuvsim`'s
                :func:`pyuvsim.simsetup.initialize_uvdata_from_params` function.
                If provided, `uvdata`, `beams`, and `beam_ids` are not required.
            uvdata (UVData):
                A :class:`pyuvdata.UVData` object contain information about
                the "observation".
            sky_freqs (1D array, shape=[NFREQS]):
                Frequencies at which the sky intensity and/or point sources
                are defined. [Hz]
            beams (list, optional, shape=[N_BEAMS]):
                UVBeam models for as many antennae as have unique beams.
                By default, a single uniform beam is applied for every antenna.
            beam_ids (1D int array, optional, shape=[N_ANTS]):
                List of integers specifying which beam model each antenna uses
                (i.e. the index of `beams` which it should refer to). By default,
                all antennas use the same beam (beam 0).
            sky_intensity (2D array, shape=[NFREQS, N_PIX_SKY]):
                A healpix model for the intensity of the sky emission.
            point_source_pos (2D array, optional, shape=[N_SOURCES, 2]):
                An array of point sources. For each source, the entries are
                (ra, dec) [rad].
            point_source_flux (2D array, optional, shape=[NFREQS, N_SOURCES]):
                An array of fluxes of the given point sources, per frequency.
                Fluxes in [Jy].
            nside (int, optional):
                Only used if sky_intensity is *not* given but the simulator
                is incapable of directly dealing with point sources. In this
                case, it sets the resolution of the healpix map to which the
                sources will be allocated.
        """
        if obsparams:
            self.uvdata, self.beams, self.beam_ids = initialize_uvdata_from_params(obsparams)
        else:
            if uvdata is None:
                raise ValueError("if obsparams is not given, uvdata must be")

            self.uvdata = uvdata

            self.beams = [ab.AnalyticBeam("uniform")] if beams is None else beams
            self.beam_ids = np.zeros(self.n_ant, dtype=np.int) if beam_ids is None else beam_ids

        self._nside = nside
        self.sky_intensity = sky_intensity

        self.sky_freqs = sky_freqs

        self.point_source_pos = point_source_pos
        self.point_source_flux = point_source_flux

        self.validate()

    def validate(self):
        if ((self.point_source_pos is not None and self.point_source_flux is None) or
                (self.point_source_flux is not None and self.point_source_pos is None)):
            raise ValueError("Either both or neither of point_source_pos and point_source_flux must be given.")

        if self.sky_intensity is not None and not healpy.isnpixok(self.n_pix):
            raise ValueError("The sky_intensity map is not compatible with healpy")

        if self.point_source_pos is None and self.sky_intensity is None:
            raise ValueError("You must pass at least one of sky_intensity or "
                             "point_sources.")

        if np.max(self.beam_ids) >= self.n_beams:
            raise ValueError("The number of beams provided must be at least as "
                             "great as the greatest beam_id")

        if self.point_source_flux is not None:
            if self.point_source_flux.shape[0] != self.sky_freqs.shape[0]:
                raise ValueError("point_source_flux must have the same number of freqs as sky_freqs")

        if self.point_source_flux is not None:
            if self.point_source_flux.shape[1] != self.point_source_pos.shape[0]:
                raise ValueError("Number of sources in point_source_flux and point_source_pos is different")

        if self.sky_intensity is not None and self.sky_intensity.shape[0] != self.sky_freqs.shape[0]:
            raise ValueError("sky_intensity has a different number of freqs than sky_freqs")

        if self.sky_intensity is not None and self.sky_intensity.ndim != 2:
            raise ValueError("sky_intensity must be a 2D array (a healpix map per frequency)")

        if not self._point_source_ability and self.point_sources is not None:
            warnings.warn("This visibility simulator is unable to explicitly "
                          "simulate point sources. Adding point sources to "
                          "diffuse pixels")
            if self.sky_intensity is None: self.sky_intensity = np.zeros(healpy.nside2npix(self.nside))
            self.sky_intensity += self.convert_point_sources_to_healpix(
                self.point_sources, self.nside
            )

        if not self._diffuse_ability and self.sky_intensity is not None:
            warnings.warn("This visibility simulator is unable to explicitly "
                          "simulate diffuse structure. Converting diffuse "
                          "intensity to approximate points")
            if self.point_sources is None: self.point_sources = 0
            self.point_sources += self.convert_healpix_to_point_sources(self.sky_intensity)

    @staticmethod
    def convert_point_sources_to_healpix(point_sources, nside=40):
        """
        Convert a set of point sources to an approximate diffuse healpix model.

        The healpix map returned is in RING scheme.

        Returns:
            1D array: the healpix diffuse model.
        """

        hmap = np.zeros(healpy.nside2npix(nside))

        # Get which pixel every point source lies in.
        pix = healpy.ang2pix(nside, point_sources[:, 0], point_sources[:, 1])

        hmap[pix] += point_sources[:, 2] / healpy.nside2pixarea(nside)

        return hmap

    @staticmethod
    def convert_healpix_to_point_sources(hmap):
        """
        Convert a healpix map to a set of point sources located at the centre
        of each pixel.

        Args:
            hmap (1D array):
                The healpix map.
        Returns:
            2D array: the point sources
        """
        nside = healpy.get_nside(hmap)
        ra, dec = healpy.pix2ang(nside, np.arange(len(hmap)))
        flux = hmap * healpy.nside2pixarea(nside)
        return np.array([ra, dec, flux])

    def simulate(self):
        """Perform the visibility simulation"""
        self._write_history()
        self._simulate()

    def _simulate(self):
        """Subclass-specific simulation method, to be overwritten."""
        pass

    @property
    def nside(self):
        """Nside parameter of the sky healpix map"""
        if self.sky_intensity is not None:
            return healpy.get_nside(self.sky_intensity[0])
        else:
            return self._nside

    @cached_property
    def n_pix(self):
        """Number of pixels in the sky map"""
        return self.sky_intensity.shape[1]

    @cached_property
    def n_ant(self):
        """Number of antennas in array"""
        return self.uvdata.get_ants().shape[0]

    @cached_property
    def n_beams(self):
        """Number of beam models used."""
        return len(self.beams)

    def _write_history(self):
        """
        Write pertinent details of simulation to the UVData's history.
        """
        self.uvdata.history += "Visibility Simulation performed with hera_sim's {} simulator\n".format(
            self.__class__.__name__)
        self.uvdata.history += "Class Repr: {}".format(repr(self))

    def write_config_file(self, **kwargs):
        """
        Write out a YAML config file corresponding to the current UVData
        object.

        Args:
            **kwargs: any options are passed to :func:`uvdata_to_config_file`.
        """
        uvdata_to_config_file(self.uvdata, **kwargs)

    # @cached_property
    # def n_pix_beam(self):
    #     """Number of pixels in the beam maps"""
    #     try:
    #         return self.beams.shape[1]
    #     except IndexError:
    #         return 0
