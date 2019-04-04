import warnings

import healpy
import numpy as np
from cached_property import cached_property


class VisibilitySimulator(object):
    _point_source_ability = True
    _diffuse_ability = True

    def __init__(self, freq, antpos, latitude, lsts, beams=None, beam_ids=None,
                 sky_intensity=None, point_sources=None, nside=50):
        """
        Base VisibilitySimulator class.

        Any actual visibility simulator should be sub-classed from this one.
        This class provides several convenience methods and defines the API.

        Note that for reasons of memory, the simulator deals with frequencies
        independently, and so only one frequency is passed. This allows you to
        load/unload data for different frequencies between calls.

        Args:
            freq (float):
                instrumental frequency [GHz]
            antpos (2D float array, shape=[N_ANTS, 3]):
                Antenna positions in ENU co-ordinates [metres]
            latitude (float):
                Latitude of centre of antenna array [rad].
            beams (2D array, optional, shape=[N_BEAMS, N_PIX_BEAM]:
                Healpix beam models for as many antennae as have unique beams.
                The array can be empty in order to not apply any beams.
            beam_ids (1D int array, optional, shape=[N_ANTS]):
                List of integers specifying which beam model each antenna uses
                (i.e. the index of `beams` which it should refer to). Negative
                values indicate absence of beam.
            lsts (1D array, shape=[N_LSTS]):
                The LSTs at which to generate visibilities.
            sky_intensity (1D array, shape=[N_PIX_SKY]):
                A healpix model for the intensity of the sky emission.
            point_sources (2D array, optional, shape=[N_SOURCES, 3]):
                An array of point sources. For each source, the entries are
                (ra, dec, flux_density [Jy]).
            nside (int, optional):
                Only used if sky_intensity is *not* given but the simulator
                is incapable of directly dealing with point sources. In this
                case, it sets the resolution of the healpix map to which the
                sources will be allocated.
        """
        self.freq = freq
        self.antpos = antpos
        self.beams = np.array([]) if beams is None else beams
        self.lsts = lsts
        self.sky_intensity = sky_intensity
        self.beam_ids = -1 * np.ones(self.n_ant, dtype=np.int) if beam_ids is None else beam_ids
        self.latitude = latitude
        self._nside = nside

        self.point_sources = point_sources

        self.validate()

    def validate(self):
        if self.sky_intensity is not None and not healpy.isnpixok(self.n_pix):
            raise ValueError("The sky_intensity map is not compatible with healpy")

        if len(self.beams) and not healpy.isnpixok(self.beams.shape[1]):
            raise ValueError("The beam maps are not compatible with healpy")

        if self.point_sources is None and self.sky_intensity is None:
            raise ValueError("You must pass at least one of sky_intensity or "
                             "point_sources.")

        if self.antpos.shape[1] != 3:
            raise ValueError("The number of co-ordinate dimensions for antennae "
                             "should be three.")

        if np.max(self.beam_ids) >= self.n_beams:
            raise ValueError("The number of beams provided must be at least as "
                             "great as the greatest beam_id")

        if self.sky_intensity is not None and self.sky_intensity.ndim != 1:
            raise ValueError("sky_intensity must be a 1D array (a healpix map)")

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
        pass

    @property
    def nside(self):
        if self.sky_intensity is not None:
            return healpy.get_nside(self.sky_intensity)
        else:
            return self._nside

    @cached_property
    def n_ant(self):
        """Number of antennas in array"""
        return self.antpos.shape[0]

    @cached_property
    def n_lsts(self):
        """Number of times (LSTs)"""
        return self.lsts.shape[0]

    @cached_property
    def n_beams(self):
        """Number of beam models used."""
        return self.beams.shape[0]

    @cached_property
    def n_pix(self):
        """Number of pixels in the sky map"""
        return self.sky_intensity.size

    @cached_property
    def n_pix_beam(self):
        """Number of pixels in the beam maps"""
        try:
            return self.beams.shape[1]
        except IndexError:
            return 0

