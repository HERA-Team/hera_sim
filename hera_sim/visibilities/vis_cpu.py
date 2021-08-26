"""Wrapper for vis_cpu visibility simulator."""
from __future__ import division, annotations
import numpy as np
import pyuvdata

from .simulators import VisibilitySimulator, ModelData
from typing import Tuple, Union, Optional, List

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
import warnings

from vis_cpu import vis_cpu, vis_gpu, HAVE_GPU, __version__
from vis_cpu import conversions as convs
from pyuvdata import UVData


class VisCPU(VisibilitySimulator):
    """
    vis_cpu visibility simulator.

    This is a fast, simple visibility simulator that is intended to be
    replaced by vis_gpu. It extends :class:`~.simulators.VisibilitySimulator`.

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
        Whether to calculate polarized visibilities or not. By default does polarization
        iff multiple polarizations exist in the UVData object. The behaviour of the
        simulator is that if requesting polarized output and only a subset of the
        simulated pols are available in the UVdata object, the code will issue a warning
        but otherwise continue happily, throwing away the simulated pols it can't store
        in the UVdata object. Conversely, if polarization is not requested and multiple
        polarizations are present on the UVData object, it will error unless
        ``allow_empty_pols`` is set to True (in which case it will warn but continue).
        The "unpolarized" output of ``vis_cpu`` is expected to be XX polarization, which
        corresponds to whatever the UVData object considers to be the x-direction
        (default East).
    precision : int, optional
        Which precision level to use for floats and complex numbers.
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128
    use_gpu : bool, optional
        Whether to use the GPU version of vis_cpu or not. Default: False.
    mpi_comm : MPI communicator
        MPI communicator, for parallelization.
    allow_empty_pols
        Whether to allow the simulation to proceed if it would leave some of the
        polarizations in the UVData object unsimulated.
    **kwargs
        Passed through to :class:`~.simulators.VisibilitySimulator`.
    """

    conjugation_convention = "ant1<ant2"
    time_ordering = "time"

    diffuse_ability = False
    __version__ = __version__

    def __init__(
        self,
        bm_pix: int = 100,
        use_pixel_beams: bool = True,
        polarized: bool | None = None,
        precision: int = 1,
        use_gpu: bool = False,
        mpi_comm=None,
        ref_time: Optional[Union[str, Time]] = None,
        correct_source_positions: bool = False,
        allow_empty_pols: bool = False,
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
        self.correct_source_positions = correct_source_positions
        self.allow_empty_pols = allow_empty_pols

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
            its shape is (NFREQS, NAXES, NFEEDS, NANT, BM_PIX, BM_PIX),
            otherwise (NFREQS, NANT, BM_PIX, BM_PIX).

        Notes
        -----
        Due to using the verbatim :func:`vis_cpu.vis_cpu` function, the beam
        cube must have an entry for each antenna, which is a bit of
        a waste of memory in some cases. If this is changed in the
        future, this method can be modified to only return one
        matrix for each beam.
        """
        out = np.asarray(
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

        if self.polarized:
            # shape FREQ, NAXES, NFEEDS, NANT, NPIX, NPIX
            return np.transpose(out, (3, 1, 2, 0, 4, 5))
        else:
            return np.transpose(out, (1, 0, 2, 3))

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

        Returns
        -------
        array_like of self._complex_dtype
            Visibilities. Shape=self.uvdata.data_array.shape.
        """
        if self.polarized is None:
            polarized = len(data_model.uvdata.polarization_array) > 1
        else:
            polarized = self.polarized

        if self.use_gpu and polarized:
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
                data_model, obstime=data_model.uvdata.time_array[0]
            )
        else:
            ra, dec = data_model.sky_model.ra, data_model.sky_model.dec

        crd_eq = convs.point_source_crd_eq(ra, dec)

        # Convert equatorial to topocentric coords
        eq2tops = self.get_eq2tops(data_model.uvdata, data_model.lsts)

        # The following are antenna positions in the order that they are
        # in the uvdata.data_array
        active_antpos, ant_list = data_model.uvdata.get_ENU_antpos(pick_data_ants=True)

        # Get pixelized beams if required
        if self.use_pixel_beams:
            beam_lm = self.get_beam_lm(data_model)
        else:
            beam_list = [
                data_model.beams[data_model.beam_ids[name]]
                for number, name in zip(
                    data_model.uvdata.antenna_numbers, data_model.uvdata.antenna_names
                )
                if number in ant_list
            ]

        # Get all the polarizations required to be simulated.
        req_pols = self._get_req_pols(data_model.uvdata, polarized)

        # Empty visibility array
        visfull = np.zeros_like(data_model.uvdata.data_array, dtype=self._complex_dtype)

        for i, freq in enumerate(data_model.freqs):
            # Divide tasks between MPI workers if needed
            if self.mpi_comm is not None and i % nproc != myid:
                continue

            # Call vis_cpu function to simulate visibilities
            vis = self._vis_cpu(
                antpos=active_antpos,
                freq=freq,
                eq2tops=eq2tops,
                crd_eq=crd_eq,
                I_sky=data_model.sky_model.stokes[0, i].to("Jy").value,
                beam_list=beam_list if not self.use_pixel_beams else None,
                bm_cube=beam_lm[i] if self.use_pixel_beams else None,
                precision=self._precision,
                polarized=self.polarized,
            )

            self._reorder_vis(
                req_pols, data_model.uvdata, visfull[:, 0, i], vis, ant_list, polarized
            )

        # Reduce visfull array if in MPI mode
        if self.mpi_comm is not None:
            return self._reduce_mpi(visfull, myid)

        return visfull

    def _reorder_vis(self, req_pols, uvdata, visfull, vis, ant_list, polarized):
        indices = (
            np.triu_indices(vis.shape[3])
            if polarized
            else np.triu_indices(vis.shape[1])
        )

        for p, (p1, p2) in enumerate(req_pols):
            for ant1, ant2 in zip(*indices):  # go through indices in output
                vis_here = (
                    vis[p1, p2, :, ant1, ant2] if polarized else vis[:, ant1, ant2]
                )
                # get official "antenna numbers" corresponding to these indices
                antnum1, antnum2 = ant_list[ant1], ant_list[ant2]

                # get all blt indices corresponding to this antpair
                indx = uvdata.antpair2ind(antnum1, antnum2)
                if len(indx) == 0:
                    # maybe we chose the wrong ordering according to the data. Then
                    # we just conjugate.
                    indx = uvdata.antpair2ind(antnum2, antnum1)
                    vis_here = np.conj(vis_here)

                visfull[indx, p] = vis_here

    def _get_req_pols(self, uvdata, polarized) -> List[Tuple[int, int]]:
        # Get x_orientation
        x_orient = uvdata.x_orientation
        if x_orient is None:
            uvdata.x_orientation = "e"  # set in UVData object
            x_orient = "e"  # default to east

        # Get available pols in the vis_cpu output,  and map them to the right output
        # index.
        if polarized:
            avail_pols = {"nn": (0, 0), "ne": (0, 1), "en": (1, 0), "ee": (1, 1)}
        else:
            avail_pols = {"ee": (1, 1)} if x_orient == "e" else {"nn": (0, 0)}

        req_pols = []
        for pol in uvdata.polarization_array:
            # Get polarization strings in terms of n/e feeds
            polstr = pyuvdata.utils.polnum2str(pol, x_orientation=x_orient).lower()

            # Check if polarization can be formed
            if polstr in avail_pols:
                # If polarization can be formed, specify which is which in the
                # output polarization_array (ordered list)
                req_pols.append(avail_pols[polstr])

            else:
                msg = (
                    "Simulation UVData object expecting polarization"
                    f" '{polstr}', but only polarizations {list(avail_pols)} "
                    "can be formed."
                )

                if not self.allow_empty_pols:
                    raise KeyError(msg)
                else:
                    warnings.warn(msg)

        return req_pols

    def _reduce_mpi(self, visfull, myid):
        from mpi4py.MPI import SUM

        _visfull = np.zeros(visfull.shape, dtype=visfull.dtype)
        self.mpi_comm.Reduce(visfull, _visfull, op=SUM, root=0)
        if myid == 0:
            return _visfull
        else:
            return 0  # workers return 0
