"""Wrapper for vis_cpu visibility simulator."""
from __future__ import division
import numpy as np
import pyuvdata
import astropy_healpix as aph

from . import conversions
from .simulators import VisibilitySimulator

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

from vis_cpu import vis_cpu, vis_gpu, HAVE_GPU
from vis_cpu import conversions as convs


class VisCPU(VisibilitySimulator):
    """
    vis_cpu visibility simulator.

    This is a fast, simple visibility simulator that is intended to be
    replaced by vis_gpu. It extends :class:`~.simulators.VisibilitySimulator`.

    Note that that output of `simulate()` in this class always has ordering
    in which the baselines are in increasing order of antenna number.

    Parameters
    ----------
    bm_pix : int, optional
        The number of pixels along a side in the beam map when
        converted to (l, m) coordinates. Defaults to 100.
    use_pixel_beams : bool, optional
        Whether to use primary beams that have been pixelated onto a 2D
        grid, or directly evaluate the primary beams using the available
        UVBeam objects. Default: True.
    polarized : bool, optional
        Whether to calculate polarized visibilities or not.
    precision : int, optional
        Which precision level to use for floats and complex numbers.
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128
    use_gpu : bool, optional
        Whether to use the GPU version of vis_cpu or not. Default: False.
    mpi_comm : MPI communicator
        MPI communicator, for parallelization.
    **kwargs
        Passed through to :class:`~.simulators.VisibilitySimulator`.
    """

    def __init__(
        self,
        bm_pix=100,
        use_pixel_beams=True,
        polarized=False,
        precision=1,
        use_gpu=False,
        mpi_comm=None,
        **kwargs
    ):

        assert precision in (1, 2)
        self._precision = precision
        if precision == 1:
            self._real_dtype = np.float32
            self._complex_dtype = np.complex64
        else:
            self._real_dtype = float
            self._complex_dtype = complex

        if use_gpu and mpi_comm is not None and mpi_comm.Get_size() > 1:
            raise RuntimeError("MPI is not yet supported in GPU mode")

        if use_gpu and not HAVE_GPU:
            raise ImportError(
                "GPU acceleration requires hera_gpu (`pip install hera_sim[gpu]`)."
            )

        if use_gpu and not use_pixel_beams:
            raise RuntimeError(
                "GPU can only be used with pixel beams (use_pixel_beams=True)"
            )
        if use_gpu and polarized:
            raise RuntimeError(
                "GPU support is currently only available when polarized=False"
            )

        self.polarized = polarized
        self._vis_cpu = vis_gpu if use_gpu else vis_cpu
        self.bm_pix = bm_pix

        self.use_gpu = use_gpu
        self.use_pixel_beams = use_pixel_beams
        self.polarized = polarized
        self.mpi_comm = mpi_comm

        super(VisCPU, self).__init__(validate=False, **kwargs)

        # If beam ids and beam lists are mis-matched, expand the beam list
        # or raise an error
        if len(self.beams) != len(self.beam_ids):

            # If N_beams > 1 and N_beams != N_ants, raise an error
            if len(self.beams) > 1:
                raise ValueError(
                    "Specified %d beams for %d antennas"
                    % (len(self.beams), len(self.beam_ids))
                )

            # If there is only one beam, assume it's the same for all ants
            if len(self.beams) == 1:
                beam = self.beams[0]
                self.beams = [beam for b in self.beam_ids]

        # Convert some arguments to simpler forms for vis_cpu.
        self.freqs = self.uvdata.freq_array[0]

        # Get antpos for active antennas only
        # self.antpos = self.uvdata.get_ENU_antpos()[0].astype(self._real_dtype)
        self.ant_list = self.uvdata.get_ants()  # ordered list of active ants
        self.antpos = []
        _antpos = self.uvdata.get_ENU_antpos()[0].astype(self._real_dtype)
        for ant in self.ant_list:
            # uvdata.get_ENU_antpos() and uvdata.antenna_numbers have entries
            # for all telescope antennas, even ones that aren't included in the
            # data_array. This extracts only the data antennas.
            idx = np.where(ant == self.uvdata.antenna_numbers)
            self.antpos.append(_antpos[idx].flatten())
        self.antpos = np.array(self.antpos)

        # Keep track of whether a source position correction has been applied
        self._point_source_pos_correction_applied = False

        # Validate
        self.validate()

    @property
    def lsts(self):
        """
        Sets LSTs from uvdata if not already set.

        Returns
        -------
        array_like
            LSTs of observations. Shape=(NTIMES,).
        """
        try:
            return self.__lsts
        except AttributeError:
            self.__lsts = self.uvdata.lst_array[:: self.uvdata.Nbls]

            return self.__lsts

    def validate(self):
        """Checks for correct input format."""
        super(VisCPU, self).validate()

        # This one in particular requires that every baseline is used!
        N = len(self.uvdata.get_ants())

        # N(N-1)/2 unique cross-correlations + N autocorrelations.
        if len(self.uvdata.get_antpairs()) != N * (N + 1) / 2:
            raise ValueError(
                "VisCPU requires using every pair of antennas, "
                "but the UVData object does not comply."
            )

        if len(self.uvdata.data_array) != len(self.uvdata.get_antpairs()) * len(
            self.lsts
        ):
            raise ValueError(
                "VisCPU requires that every baseline uses the " "same LSTS."
            )

        # Check to make sure enough beams are specified
        if not self.use_pixel_beams:
            for ant in self.ant_list:
                assert len(np.where(self.beam_ids == ant)[0]), (
                    "No beam found for antenna %d" % ant
                )

    def correct_point_source_pos(self, obstime, frame="icrs"):
        """Apply correction to source RA and Dec positions to improve accuracy.

        This uses an astropy-based coordinate correction, computed for a single
        reference time, to shift the source RA and Dec coordinates to ones that
        produce more accurate Alt/Az positions at the location of the array.

        The ``self.point_source_pos`` array is updated by this function.

        Parameters
        ----------
        obstime : str or astropy.Time
            Specifies the time of the reference observation used to compute the
            coordinate correction. If specified as a string, this must use the
            'isot' format and 'utc' scale.

        frame : str, optional
            Which frame that the original RA and Dec positions are specified
            in. Any system recognized by ``astropy.SkyCoord`` can be used.
        """
        # Check whether correction has already been applied
        if self._point_source_pos_correction_applied:
            raise ValueError("`correct_point_source_pos()` has already been applied.")

        # Check input reference time
        if isinstance(obstime, str):
            obstime = Time(obstime, format="isot", scale="utc")
        elif not isinstance(obstime, Time):
            raise TypeError("`obstime` must be a string or astropy.Time object")

        # Get reference location
        lat, lon, alt = self.uvdata.telescope_location_lat_lon_alt_degrees
        # location = EarthLocation.from_geodetic(lat=lat, lon=lon, height=alt)
        location = EarthLocation.from_geocentric(
            *self.uvdata.telescope_location, unit=u.m
        )

        # Apply correction to point source positions
        shape = self.point_source_pos.shape
        ra, dec = self.point_source_pos.T
        new_ra, new_dec = convs.equatorial_to_eci_coords(
            ra, dec, obstime, location, unit="rad", frame=frame
        )

        # Combine back into single position array
        self.point_source_pos = np.array([new_ra, new_dec]).T
        assert self.point_source_pos.shape == shape
        self._point_source_pos_correction_applied = True

    def get_beam_lm(self):
        """
        Obtain the beam pattern in (l,m) co-ordinates for each antenna.

        Returns
        -------
        bm_cube : array_like
            The beam pattern in (l,m) for each antenna. If `self.polarized=True`,
            its shape is (NANT, NAXES, NFEEDS, NFREQS, BM_PIX, BM_PIX),
            otherwise (NANT, NFREQS, BM_PIX, BM_PIX).

        Notes
        -----
            Due to using the verbatim :func:`vis_cpu.vis_cpu` function, the beam
            cube must have an entry for each antenna, which is a bit of
            a waste of memory in some cases. If this is changed in the
            future, this method can be modified to only return one
            matrix for each beam.
        """
        return np.asarray(
            [
                convs.uvbeam_to_lm(
                    self.beams[np.where(self.beam_ids == ant)[0][0]],
                    self.freqs,
                    n_pix_lm=self.bm_pix,
                    polarized=self.polarized,
                )
                for ant in self.ant_list
            ]
        )

    def get_diffuse_crd_eq(self):
        """
        Calculate equatorial coords of HEALPix sky pixels (Cartesian).

        Returns
        -------
        array_like of self._real_dtype
            The equatorial co-ordinates of each pixel.
            Shape=(12*NPIX^2, 3).
        """
        diffuse_eq = conversions.healpix_to_crd_eq(self.sky_intensity[0])
        return diffuse_eq.astype(self._real_dtype)

    def get_point_source_crd_eq(self):
        """
        Get array of point source locations.

        Returns
        -------
        array_like
            Equatorial coordinates of sources, in Cartesian system.
            Shape=(3, NSRCS).
        """
        ra, dec = self.point_source_pos.T
        return convs.point_source_crd_eq(ra, dec)

    def get_eq2tops(self):
        """
        Calculate transformations from equatorial to topocentric coords.

        Returns
        -------
        array_like of self._real_dtype
            The set of 3x3 transformation matrices converting equatorial
            to topocenteric co-ordinates at each LST.
            Shape=(NTIMES, 3, 3).
        """
        latitude = self.uvdata.telescope_location_lat_lon_alt[0]  # rad
        eq2tops = np.array(
            [convs.eci_to_enu_matrix(lst, latitude) for lst in self.lsts],
            dtype=self._real_dtype,
        )
        return eq2tops

    def _base_simulate(self, crd_eq, I_sky):
        """
        Calls :func:vis_cpu to perform the visibility calculation.

        Parameters
        ----------
        crd_eq : array_like
            Rotation matrix to convert between source coords and equatorial
            coords.

        I : array_like
            Flux for each source in each frequency channel.

        Returns
        -------
        array_like of self._complex_dtype
            Visibilities. Shape=self.uvdata.data_array.shape.
        """
        if self.use_gpu and self.polarized:
            raise NotImplementedError(
                "use_gpu not currently supported if " "polarized=True"
            )

        # Setup MPI info if enabled
        if self.mpi_comm is not None:
            myid = self.mpi_comm.Get_rank()
            nproc = self.mpi_comm.Get_size()

        # Convert equatorial to topocentric coords
        eq2tops = self.get_eq2tops()

        # Get pixelized beams if required
        if self.use_pixel_beams:
            beam_lm = self.get_beam_lm()
            # polarized=True:  (NANT, NAXES, NFEEDS, NFREQS, BM_PIX, BM_PIX)
            # polarized=False: (NANT, NFREQS, BM_PIX, BM_PIX)
            # if not self.polarized:
            #    beam_lm = beam_lm[np.newaxis, np.newaxis, :, :, :]
        else:
            beam_list = [
                self.beams[np.where(self.beam_ids == ant)[0][0]]
                for ant in self.ant_list
            ]

        # Get required pols and map them to the right output index
        if self.polarized:
            avail_pols = {"nn": (0, 0), "ne": (0, 1), "en": (1, 0), "ee": (1, 1)}
        else:
            avail_pols = {
                "ee": (1, 1),
            }  # only xx = ee

        req_pols = []
        for pol in self.uvdata.polarization_array:

            # Get x_orientation
            x_orient = self.uvdata.x_orientation
            if x_orient is None:
                self.uvdata.x_orientation = "e"  # set in UVData object
                x_orient = "e"  # default to east

            # Get polarization strings in terms of n/e feeds
            polstr = pyuvdata.utils.polnum2str(pol, x_orientation=x_orient).lower()

            # Check if polarization can be formed
            if polstr not in avail_pols.keys():
                raise KeyError(
                    "Simulation UVData object expecting polarization"
                    " '%s', but only polarizations %s can be formed."
                    % (polstr, list(avail_pols.keys()))
                )

            # If polarization can be formed, specify which is which in the
            # output polarization_array (ordered list)
            req_pols.append(avail_pols[polstr])

        # Empty visibility array
        visfull = np.zeros_like(self.uvdata.data_array, dtype=self._complex_dtype)

        for i, freq in enumerate(self.freqs):

            # Divide tasks between MPI workers if needed
            if self.mpi_comm is not None and i % nproc != myid:
                continue

            # Determine correct shape of beam cube if pixel beams are used
            if self.use_pixel_beams:
                if self.polarized:
                    _beam_lm = beam_lm[:, :, :, i, :, :]
                    # (NANT, NAXES, NFEEDS, NFREQS, BM_PIX, BM_PIX)
                else:
                    _beam_lm = beam_lm[:, i, :, :]
                    # (NANT, NFREQS, BM_PIX, BM_PIX)

            # Call vis_cpu function to simulate visibilities
            vis = self._vis_cpu(
                antpos=self.antpos,
                freq=freq,
                eq2tops=eq2tops,
                crd_eq=crd_eq,
                I_sky=I_sky[i],
                beam_list=beam_list if not self.use_pixel_beams else None,
                bm_cube=_beam_lm if self.use_pixel_beams else None,
                precision=self._precision,
                polarized=self.polarized,
            )

            # Assign simulated visibilities to UVData data_array
            if self.polarized:
                indices = np.triu_indices(vis.shape[3])
                for p, pidxs in enumerate(req_pols):
                    p1, p2 = pidxs
                    vis_upper_tri = vis[p1, p2, :, indices[0], indices[1]]
                    visfull[:, 0, i, p] = vis_upper_tri.flatten()
                    # Shape: (Nblts, Nspws, Nfreqs, Npols)
            else:
                # Only one polarization (vis is returned without first 2 dims)
                indices = np.triu_indices(vis.shape[1])
                vis_upper_tri = vis[:, indices[0], indices[1]]
                visfull[:, 0, i, 0] = vis_upper_tri.flatten()

        # Reduce visfull array if in MPI mode
        if self.mpi_comm is not None:
            from mpi4py.MPI import SUM

            _visfull = np.zeros(visfull.shape, dtype=visfull.dtype)
            self.mpi_comm.Reduce(visfull, _visfull, op=SUM, root=0)
            if myid == 0:
                return _visfull
            else:
                return 0  # workers return 0

        return visfull

    def _simulate_diffuse(self):
        """
        Simulate diffuse sources.

        Returns
        -------
        array_like
            Visibility from point sources.
            Shape=self.uvdata.data_array.shape.
        """
        crd_eq = self.get_diffuse_crd_eq()
        # Multiply intensity by pix area because the algorithm doesn't.
        return self._base_simulate(
            crd_eq,
            self.sky_intensity
            * aph.nside_to_pixel_area(self.nside).to(u.rad ** 2).value,
        )

    def _simulate_points(self):
        """
        Simulate point sources.

        Returns
        -------
        array_like
            Visibility from diffuse sources.
            Shape=self.uvdata.data_array.shape.
        """
        crd_eq = self.get_point_source_crd_eq()
        return self._base_simulate(crd_eq, self.point_source_flux)

    def _simulate(self):
        """
        Simulate diffuse and point sources.

        Returns
        -------
        array_like
            Visibility from all sources.
            Shape=self.uvdata.data_array.shape.
        """
        vis = 0
        if self.sky_intensity is not None:
            vis += self._simulate_diffuse()
        if self.point_source_flux is not None:
            vis += self._simulate_points()
        return vis
