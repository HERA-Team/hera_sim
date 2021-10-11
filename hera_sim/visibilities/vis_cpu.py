"""Wrapper for vis_cpu visibility simulator."""
from __future__ import division
import numpy as np
import pyuvdata

from .simulators import VisibilitySimulator, ModelData
from typing import Tuple, Union, Optional

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

from vis_cpu import vis_cpu, vis_gpu, HAVE_GPU, __version__
from vis_cpu import conversions as convs
from pyuvdata import UVData


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
    ref_time
        A reference time for computing adjustments to the co-ordinate transforms using
        astropy. For best fidelity, set this to a mid-point of your observation times.
        By default, if `correct_source_positions` is True, will use the first
        observation time.
    correct_source_positions
        Whether to correct the source positions using astropy and the reference time.
        Default is True if `ref_time` is given otherwise False.
    """

    conjugation_convention = "ant1<ant2"
    time_ordering = "time"

    diffuse_ability = False
    __version__ = __version__

    def __init__(
        self,
        bm_pix: int = 100,
        use_pixel_beams: bool = True,
        polarized: bool = False,
        precision: int = 1,
        use_gpu: bool = False,
        mpi_comm=None,
        ref_time: Optional[Union[str, Time]] = None,
        correct_source_positions: bool | None = None,
    ):

        assert precision in {1, 2}
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
        self.ref_time = ref_time
        self.correct_source_positions = (
            (ref_time is not None)
            if correct_source_positions is None
            else correct_source_positions
        )

    def validate(self, data_model: ModelData):
        """Checks for correct input format."""
        # N(N-1)/2 unique cross-correlations + N autocorrelations.
        if data_model.uvdata.Nbls != data_model.n_ant * (data_model.n_ant + 1) / 2:
            raise ValueError(
                "VisCPU requires using every pair of antennas, "
                "but the UVData object does not comply."
            )

        if len(data_model.uvdata.data_array) != len(
            data_model.uvdata.get_antpairs()
        ) * len(data_model.lsts):
            raise ValueError("VisCPU requires that every baseline uses the same LSTS.")

        if any(
            len(data_model.uvdata.antpair2ind(ai, aj)) > 0
            and len(data_model.uvdata.antpair2ind(aj, ai)) > 0
            for ai, aj in data_model.uvdata.get_antpairs()
            if ai != aj
        ):
            raise ValueError(
                "VisCPU requires that baselines be in a conjugation in which antenna "
                "order doesn't change with time!"
            )

        # if self.polarized and len(data_model.uvdata.polarization_array) != 4:
        #     raise ValueError(
        #         "You are trying to do a polarized simulation but your input UVData"
        #         "object has only a single polarization."
        #     )

        # if not self.polarized and len(data_model.uvdata.polarization_array) > 1:
        #     raise ValueError(
        #         "Your UVData object has multiple polarizations, but you are not "
        #         "including polarization in your simulation!"
        #     )

    def correct_point_source_pos(
        self,
        data_model: ModelData,
        obstime: Optional[Union[str, Time]] = None,
        frame: str = "icrs",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply correction to source RA and Dec positions to improve accuracy.

        This uses an astropy-based coordinate correction, computed for a single
        reference time, to shift the source RA and Dec coordinates to ones that
        produce more accurate Alt/Az positions at the location of the array.

        Parameters
        ----------
        obstime : str or astropy.Time
            Specifies the time of the reference observation used to compute the
            coordinate correction. If specified as a string, this must use the
            'isot' format and 'utc' scale.

        frame : str, optional
            Which frame that the original RA and Dec positions are specified
            in. Any system recognized by ``astropy.SkyCoord`` can be used.

        Returns
        -------
        ra, dec
            The updated source positions.
        """
        # Check input reference time
        if self.ref_time is not None:
            obstime = self.ref_time

        if isinstance(obstime, str):
            obstime = Time(obstime, format="isot", scale="utc")
        elif not isinstance(obstime, Time):
            raise TypeError("`obstime` must be a string or astropy.Time object")

        # Get reference location
        # location = EarthLocation.from_geodetic(lat=lat, lon=lon, height=alt)
        location = EarthLocation.from_geocentric(
            *data_model.uvdata.telescope_location, unit=u.m
        )

        # Apply correction to point source positions
        ra, dec = data_model.sky_model.ra, data_model.sky_model.dec
        return convs.equatorial_to_eci_coords(
            ra, dec, obstime, location, unit="rad", frame=frame
        )

    def get_beam_lm(self, data_model: ModelData) -> np.ndarray:
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
                    data_model.beams[data_model.beam_ids[ant]],
                    data_model.freqs,
                    n_pix_lm=self.bm_pix,
                    polarized=self.polarized,
                )
                for ant, num in zip(
                    data_model.uvdata.antenna_names, data_model.uvdata.antenna_numbers
                )
                if num in data_model.uvdata.get_ants()
            ]
        )

    def get_eq2tops(self, uvdata: UVData, lsts: np.ndarray):
        """
        Calculate transformations from equatorial to topocentric coords.

        Returns
        -------
        array_like of self._real_dtype
            The set of 3x3 transformation matrices converting equatorial
            to topocenteric co-ordinates at each LST.
            Shape=(NTIMES, 3, 3).
        """
        latitude = uvdata.telescope_location_lat_lon_alt[0]  # rad
        return np.array(
            [convs.eci_to_enu_matrix(lst, latitude) for lst in lsts],
            dtype=self._real_dtype,
        )

    def simulate(self, data_model):
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

        if self.correct_source_positions:
            # TODO: check if this is the right time to be using...
            ra, dec = self.correct_point_source_pos(
                data_model, obstime=Time(data_model.uvdata.time_array[0], format="jd")
            )
        else:
            ra, dec = data_model.sky_model.ra, data_model.sky_model.dec

        crd_eq = convs.point_source_crd_eq(ra, dec)

        # Convert equatorial to topocentric coords
        eq2tops = self.get_eq2tops(data_model.uvdata, data_model.lsts)

        # ant_list = data_model.uvdata.get_ants()
        # The following are antenna positions in the order that they are
        # in the uvdata.data_array
        active_antpos, ant_list = data_model.uvdata.get_ENU_antpos(pick_data_ants=True)

        # Get pixelized beams if required
        if self.use_pixel_beams:
            beam_lm = self.get_beam_lm(data_model)
            if self.polarized:
                beam_lm = np.transpose(beam_lm, (1, 2, 0, 3, 4, 5))
        else:
            beam_list = [
                data_model.beams[data_model.beam_ids[name]]
                for number, name in zip(
                    data_model.uvdata.antenna_numbers, data_model.uvdata.antenna_names
                )
                if number in ant_list
            ]

        # Get x_orientation
        x_orient = data_model.uvdata.x_orientation
        if x_orient is None:
            data_model.uvdata.x_orientation = "e"  # set in UVData object
            x_orient = "e"  # default to east

        # Get required pols and map them to the right output index
        if self.polarized:
            avail_pols = {"nn": (0, 0), "ne": (0, 1), "en": (1, 0), "ee": (1, 1)}
        else:
            avail_pols = {"ee": (1, 1)} if x_orient == "e" else {"nn": (0, 0)}

        req_pols = []
        for pol in data_model.uvdata.polarization_array:
            # Get polarization strings in terms of n/e feeds
            polstr = pyuvdata.utils.polnum2str(pol, x_orientation=x_orient).lower()

            # Check if polarization can be formed
            if polstr not in avail_pols.keys():
                raise KeyError(
                    "Simulation UVData object expecting polarization"
                    f" '{polstr}', but only polarizations {list(avail_pols.keys())} "
                    "can be formed."
                )

            # If polarization can be formed, specify which is which in the
            # output polarization_array (ordered list)
            req_pols.append(avail_pols[polstr])

        # Empty visibility array
        visfull = np.zeros_like(data_model.uvdata.data_array, dtype=self._complex_dtype)

        for i, freq in enumerate(data_model.freqs):

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
                antpos=active_antpos,
                freq=freq,
                eq2tops=eq2tops,
                crd_eq=crd_eq,
                I_sky=data_model.sky_model.stokes[0, i].to("Jy").value,
                beam_list=beam_list if not self.use_pixel_beams else None,
                bm_cube=_beam_lm if self.use_pixel_beams else None,
                precision=self._precision,
                polarized=self.polarized,
            )
            indices = (
                np.triu_indices(vis.shape[3])
                if self.polarized
                else np.triu_indices(vis.shape[1])
            )

            # Order output correctly
            if not self.polarized:
                req_pols = [(0, (0, 0))]

            for p, pidxs in enumerate(req_pols):
                p1, p2 = pidxs
                for ant1, ant2 in zip(*indices):  # go through indices in output
                    if self.polarized:
                        vis_here = vis[p1, p2, :, ant1, ant2]
                    else:
                        vis_here = vis[:, ant1, ant2]

                    # get official "antenna numbers" corresponding to these indices
                    antnum1, antnum2 = ant_list[ant1], ant_list[ant2]

                    # get all blt indices corresponding to this antpair
                    indx = data_model.uvdata.antpair2ind(antnum1, antnum2)
                    if len(indx) == 0:
                        # maybe we chose the wrong ordering according to the data. Then
                        # we just conjugate.
                        indx = data_model.uvdata.antpair2ind(antnum2, antnum1)
                        vis_here = np.conj(vis_here)

                    visfull[indx, 0, i, p] = vis_here

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
