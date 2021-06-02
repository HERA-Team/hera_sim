"""Wrapper for vis_cpu visibility simulator."""
from __future__ import division
import numpy as np
import astropy_healpix as aph

from . import conversions
from .simulators import VisibilitySimulator

from astropy.units import rad

from vis_cpu import vis_cpu, vis_gpu, HAVE_GPU


class VisCPU(VisibilitySimulator):
    """
    vis_cpu visibility simulator.

    This is a fast, simple visibility simulator that is intended to be
    replaced by vis_gpu. It extends :class:`VisibilitySimulator`.

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
    precision : int, optional
        Which precision level to use for floats and complex numbers.
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128
    use_gpu : bool, optional
        Whether to use the GPU version of vis_cpu or not. Default: False.
    mpi_comm : MPI communicator
        MPI communicator, for parallelization.

    Other Parameters
    ----------------
    Passed through to :class:`VisibilitySimulator`.
    """

    def __init__(
        self,
        bm_pix=100,
        use_pixel_beams=True,
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
            raise RuntimeError("Can't use multiple MPI processes with GPU (yet)")

        if use_gpu and not HAVE_GPU:
            raise ImportError(
                "GPU acceleration requires hera_gpu (`pip install hera_sim[gpu]`)."
            )

        if use_gpu and not use_pixel_beams:
            raise RuntimeError(
                "GPU can only be used with pixel beams (use_pixel_beams=True)"
            )

        self._vis_cpu = vis_gpu if use_gpu else vis_cpu
        self.bm_pix = bm_pix

        self.use_gpu = use_gpu
        self.use_pixel_beams = use_pixel_beams
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

            # # If there is only one beam, assume it's the same for all ants
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

    def get_beam_lm(self):
        """
        Obtain the beam pattern in (l,m) co-ordinates for each antenna.

        Returns
        -------
        array_like
            The beam pattern in (l,m) for each antenna.
            Shape=(NANT, BM_PIX, BM_PIX).

        Notes
        -----
            Due to using the verbatim :func:`vis_cpu` function, the beam
            cube must have an entry for each antenna, which is a bit of
            a waste of memory in some cases. If this is changed in the
            future, this method can be modified to only return one
            matrix for each beam.
        """
        return np.asarray(
            [
                conversions.uvbeam_to_lm(
                    self.beams[np.where(self.beam_ids == ant)[0][0]],
                    self.freqs,
                    self.bm_pix,
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
        Calculate approximate HEALPix map of point sources.

        Returns
        -------
        array_like
            equatorial coordinates of Healpix pixels, in Cartesian
            system. Shape=(3, NPIX).
        """
        ra, dec = self.point_source_pos.T
        return np.asarray(
            [np.cos(ra) * np.cos(dec), np.cos(dec) * np.sin(ra), np.sin(dec)]
        )

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
        sid_time = self.lsts
        eq2tops = np.empty((len(sid_time), 3, 3), dtype=self._real_dtype)

        for i, st in enumerate(sid_time):
            dec = self.uvdata.telescope_location_lat_lon_alt[0]
            eq2tops[i] = conversions.eq2top_m(-st, dec)

        return eq2tops

    def _base_simulate(self, crd_eq, I_sky):
        """
        Calls :func:vis_cpu to perform the visibility calculation.

        Returns
        -------
        array_like of self._complex_dtype
            Visibilities. Shape=self.uvdata.data_array.shape.
        """
        # Setup MPI info if enabled
        if self.mpi_comm is not None:
            myid = self.mpi_comm.Get_rank()
            nproc = self.mpi_comm.Get_size()

        # Convert equatorial to topocentric coords
        eq2tops = self.get_eq2tops()

        # Get pixelized beams if required
        if self.use_pixel_beams:
            beam_lm = self.get_beam_lm()
        else:
            beam_list = [
                self.beams[np.where(self.beam_ids == ant)[0][0]]
                for ant in self.ant_list
            ]

        visfull = np.zeros_like(self.uvdata.data_array, dtype=self._complex_dtype)

        for i, freq in enumerate(self.freqs):

            # Divide tasks between MPI workers if needed
            if self.mpi_comm is not None:
                if i % nproc != myid:
                    continue

            if self.use_pixel_beams:
                # Use pixelized primary beams
                vis = self._vis_cpu(
                    antpos=self.antpos,
                    freq=freq,
                    eq2tops=eq2tops,
                    crd_eq=crd_eq,
                    I_sky=I_sky[i],
                    bm_cube=beam_lm[:, i],
                    precision=self._precision,
                )
            else:
                # Use UVBeam objects directly
                vis = self._vis_cpu(
                    antpos=self.antpos,
                    freq=freq,
                    eq2tops=eq2tops,
                    crd_eq=crd_eq,
                    I_sky=I_sky[i],
                    beam_list=beam_list,
                    precision=self._precision,
                )

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
            self.sky_intensity * aph.nside_to_pixel_area(self.nside).to(rad ** 2).value,
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
