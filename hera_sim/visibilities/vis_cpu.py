"""Wrapper for vis_cpu visibility simulator."""
from __future__ import annotations

import astropy.units as u
import itertools
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from pyuvdata import UVData
from pyuvdata import utils as uvutils

from vis_cpu import HAVE_GPU, __version__
from vis_cpu import conversions as convs
from vis_cpu import vis_cpu, vis_gpu
from vis_cpu.cpu import _evaluate_beam_cpu, _wrangle_beams

from .simulators import ModelData, VisibilitySimulator


class VisCPU(VisibilitySimulator):
    """
    vis_cpu visibility simulator.

    This is a fast, simple visibility simulator that is intended to be
    replaced by vis_gpu.

    Parameters
    ----------
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
    ref_time
        A reference time for computing adjustments to the co-ordinate transforms using
        astropy. For best fidelity, set this to a mid-point of your observation times.
        If specified as a string, this must either use the 'isot' format and 'utc'
        scale, or be one of "mean", "min" or "max". If any of the latter, the value
        ll be calculated from the input data directly.
    correct_source_positions
        Whether to correct the source positions using astropy and the reference time.
        Default is True if `ref_time` is given otherwise False.
    **kwargs
        Passed through to :class:`~.simulators.VisibilitySimulator`.

    """

    conjugation_convention = "ant1<ant2"
    time_ordering = "time"

    diffuse_ability = False
    __version__ = __version__

    def __init__(
        self,
        precision: int = 1,
        use_gpu: bool = False,
        mpi_comm=None,
        ref_time: str | Time | None = None,
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

        self._vis_cpu = vis_gpu if use_gpu else vis_cpu

        self.use_gpu = use_gpu
        self.mpi_comm = mpi_comm
        self.ref_time = ref_time
        self.correct_source_positions = (
            (ref_time is not None)
            if correct_source_positions is None
            else correct_source_positions
        )

        self._functions_to_profile = (self._vis_cpu, _wrangle_beams, _evaluate_beam_cpu)

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

        uvbeam = data_model.beams[0]  # Representative beam
        uvdata = data_model.uvdata

        # Now check that we only have linear polarizations (don't allow pseudo-stokes)
        if any(pol not in [-5, -6, -7, -8] for pol in uvdata.polarization_array):
            raise ValueError(
                """
                While UVData allows non-linear polarizations, they are not suitable
                for generating simulations. Please convert your UVData object to use
                linear polarizations before simulating (and convert back to
                other polarizations afterwards if necessary).
                """
            )

        do_pol = self._check_if_polarized(data_model)
        if do_pol:
            # Number of feeds must be two if doing polarized
            try:
                nfeeds = uvbeam.data_array.shape[2]
            except AttributeError:
                # TODO: the following assumes that analytic beams are 2 feeds unless
                # otherwise specified. This should be fixed at the AnalyticBeam API
                # level.
                nfeeds = getattr(uvbeam, "Nfeeds", 2)

            assert nfeeds == 2

    def estimate_memory(self, data_model: ModelData) -> float:
        """
        Estimates the memory usage of the model.

        Parameters
        ----------
        data_model : ModelData
            The model data.

        Returns
        -------
        float
            Estimated memory usage in GB.
        """
        bm = data_model.beams[0]
        nt = len(data_model.lsts)
        nax = getattr(bm, "Naxes_vec", 1)
        nfd = getattr(bm, "Nfeeds", 1)
        nant = len(data_model.uvdata.antenna_names)
        nsrc = len(data_model.sky_model.ra)
        nbeam = len(data_model.beams)
        nf = len(data_model.freqs)

        try:
            nbmpix = bm.data_array[..., 0, :].size
        except AttributeError:
            nbmpix = 0

        all_floats = (
            nf * nt * nfd**2 * nant**2
            + nant * nsrc * nax * nfd / 2  # visibilities
            + nf * nbeam * nbmpix  # per-antenna vis
            + nax * nfd * nbeam * nsrc / 2  # raw beam
            + 3 * nant  # interpolated beam
            + nsrc * nf  # antenna positions
            + nt * 9  # source fluxes
            + 3 * nsrc  # rotation matrices
            + 3 * nsrc
            + nant * nsrc / 2  # source positions (topo and eq)  # tau.
        )

        return all_floats * self._precision * 4 / 1024**3

    def correct_point_source_pos(
        self,
        data_model: ModelData,
        obstime: str | Time | None = None,
        frame: str = "icrs",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply correction to source RA and Dec positions to improve accuracy.

        This uses an astropy-based coordinate correction, computed for a single
        reference time, to shift the source RA and Dec coordinates to ones that
        produce more accurate Alt/Az positions at the location of the array.

        Parameters
        ----------
        obstime : str or astropy.Time
            Specifies the time of the reference observation used to compute the
            coordinate correction. If specified as a string, this must either use the
            'isot' format and 'utc' scale, or be one of "mean", "min" or "max". If any
            of the latter, the ``data_model`` will be used to generate the reference
            time.

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
            if obstime == "mean":
                obstime = Time(data_model.uvdata.time_array.mean(), format="jd")
            elif obstime == "min":
                obstime = Time(data_model.uvdata.time_array.min(), format="jd")
            elif obstime == "max":
                obstime = Time(data_model.uvdata.time_array.max(), format="jd")
            else:
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

    def _check_if_polarized(self, data_model: ModelData) -> bool:
        p = data_model.uvdata.polarization_array
        # We only do a non-polarized simulation if UVData has only XX or YY polarization
        return len(p) != 1 or uvutils.polnum2str(p[0]) not in ["xx", "yy"]

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

    def get_feed(self, uvdata) -> str:
        """Get the feed to use from the beam, given the UVData object.

        Only applies for an *unpolarized* simulation (for a polarized sim, all feeds
        are used).
        """
        return uvutils.polnum2str(uvdata.polarization_array[0])[0]

    def simulate(self, data_model):
        """
        Calls :func:vis_cpu to perform the visibility calculation.

        Returns
        -------
        array_like of self._complex_dtype
            Visibilities. Shape=self.uvdata.data_array.shape.
        """
        polarized = self._check_if_polarized(data_model)
        feed = self.get_feed(data_model.uvdata)

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

        # The following are antenna positions in the order that they are
        # in the uvdata.data_array
        active_antpos, ant_list = data_model.uvdata.get_ENU_antpos(pick_data_ants=True)

        # Get pixelized beams if required

        beam_list = [
            convs.prepare_beam(
                beam,
                polarized=polarized,
                use_feed=feed,
            )
            for beam in data_model.beams
        ]
        beam_ids = np.array(
            [
                data_model.beam_ids[nm]
                for i, nm in zip(
                    data_model.uvdata.antenna_numbers, data_model.uvdata.antenna_names
                )
                if i in ant_list
            ]
        )

        # Get all the polarizations required to be simulated.
        req_pols = self._get_req_pols(
            data_model.uvdata, data_model.beams[0], polarized=polarized
        )

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
                beam_list=beam_list,
                beam_idx=beam_ids,
                precision=self._precision,
                polarized=polarized,
            )

            self._reorder_vis(
                req_pols, data_model.uvdata, visfull[:, 0, i], vis, ant_list, polarized
            )

        # Reduce visfull array if in MPI mode
        if self.mpi_comm is not None:
            visfull = self._reduce_mpi(visfull, myid)

        return visfull

    def _reorder_vis(self, req_pols, uvdata, visfull, vis, ant_list, polarized):
        indices = np.triu_indices(vis.shape[-1])

        for p, (p1, p2) in enumerate(req_pols):
            for ant1, ant2 in zip(*indices):  # go through indices in output
                vis_here = (
                    vis[:, p1, p2, ant1, ant2] if polarized else vis[:, ant1, ant2]
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

    def _get_req_pols(self, uvdata, uvbeam, polarized: bool) -> list[tuple[int, int]]:
        if not polarized:
            return [(0, 0)]

        # TODO: this can be updated to just access uvbeam.feed_array once the
        # AnalyticBeam API has been improved.
        feeds = list(getattr(uvbeam, "feed_array", ["x", "y"]))

        # In order to get all 4 visibility polarizations for a dual feed system
        vispols = set()
        for p1, p2 in itertools.combinations_with_replacement(feeds, 2):
            vispols.add(p1 + p2)
            vispols.add(p2 + p1)
        avail_pols = {
            vispol: (feeds.index(vispol[0]), feeds.index(vispol[1]))
            for vispol in vispols
        }
        # Get the mapping from uvdata pols to uvbeam pols
        uvdata_pols = [
            uvutils.polnum2str(polnum, getattr(uvbeam, "x_orientation", None))
            for polnum in uvdata.polarization_array
        ]
        if any(pol not in avail_pols for pol in uvdata_pols):
            raise ValueError(
                "Not all polarizations in UVData object are in your beam. "
                f"UVData polarizations = {uvdata_pols}. "
                f"UVBeam polarizations = {list(avail_pols.keys())}"
            )

        return [avail_pols[pol] for pol in uvdata_pols]

    def _reduce_mpi(self, visfull, myid):  # pragma: no cover
        from mpi4py.MPI import SUM

        _visfull = np.zeros(visfull.shape, dtype=visfull.dtype)
        self.mpi_comm.Reduce(visfull, _visfull, op=SUM, root=0)
        if myid == 0:
            return _visfull
        else:
            return 0  # workers return 0
