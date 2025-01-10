"""Wrapper for matvis visibility simulator."""

from __future__ import annotations

import itertools
import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from matvis import HAVE_GPU, __version__, cpu

if HAVE_GPU:
    from matvis import gpu

from matvis import coordinates
from matvis.core.beams import prepare_beam_unpolarized
from pyuvdata import UVData
from pyuvdata import utils as uvutils

from .simulators import ModelData, VisibilitySimulator

logger = logging.getLogger(__name__)


class MatVis(VisibilitySimulator):
    """
    matvis visibility simulator.

    This is a fast, matrix-based visibility simulator.

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
        The "unpolarized" output of ``matvis`` is expected to be XX polarization, which
        corresponds to whatever the UVData object considers to be the x-direction
        (default East).
    precision : int, optional
        Which precision level to use for floats and complex numbers.
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128
    use_gpu : bool, optional
        Whether to use the GPU version of matvis or not. Default: False.
    mpi_comm : MPI communicator
        MPI communicator, for parallelization.
    ref_time
        A reference time for computing adjustments to the co-ordinate transforms using
        astropy. For best fidelity, set this to a mid-point of your observation times.
        If specified as a string, this must either use the 'isot' format and 'utc'
        scale, or be one of "mean", "min" or "max". If any of the latter, the value
        ll be calculated from the input data directly.
    check_antenna_conjugation
        Whether to check the antenna conjugation. Default is True. This is a fairly
        heavy operation if there are many antennas and/or many times, and can be
        safely ignored if the data_model was created from a config file.
    **kwargs
        Passed through to :class:`~.simulators.VisibilitySimulator`.

    """

    conjugation_convention = "ant1<ant2"
    time_ordering = "time"

    diffuse_ability = False
    __version__ = __version__

    def __init__(
        self,
        precision: int = 2,
        use_gpu: bool = False,
        mpi_comm=None,
        check_antenna_conjugation: bool = True,
        **kwargs,
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
                "GPU acceleration requires installing with `pip install hera_sim[gpu]`."
            )

        self._matvis = gpu.simulate if use_gpu else cpu.simulate

        self.use_gpu = use_gpu
        self.mpi_comm = mpi_comm
        self.check_antenna_conjugation = check_antenna_conjugation
        self._functions_to_profile = (self._matvis,)
        self.kwargs = kwargs

    def validate(self, data_model: ModelData):
        """Checks for correct input format."""
        # N(N-1)/2 unique cross-correlations + N autocorrelations.
        if data_model.uvdata.Nbls != data_model.n_ant * (data_model.n_ant + 1) / 2:
            raise ValueError(
                "MatVis requires using every pair of antennas, "
                "but the UVData object does not comply."
            )

        logger.info("Checking baseline-time axis shape")
        if not data_model.uvdata.blts_are_rectangular:
            raise ValueError("MatVis requires that every baseline uses the same LSTS.")

        if self.check_antenna_conjugation:
            logger.info("Checking antenna conjugation")
            # TODO: the following is extremely slow. If possible, it would be good to
            # find a better way to do it.
            if any(
                data_model.uvdata.antpair2ind(ai, aj) is not None
                and data_model.uvdata.antpair2ind(aj, ai) is not None
                for ai, aj in data_model.uvdata.get_antpairs()
                if ai != aj
            ):
                raise ValueError(
                    "MatVis requires that baselines be in a conjugation in which "
                    "antenna order doesn't change with time!"
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
            assert uvbeam.beam.Nfeeds == 2

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

    def _check_if_polarized(self, data_model: ModelData) -> bool:
        p = data_model.uvdata.polarization_array
        # We only do a non-polarized simulation if UVData has only XX or YY polarization
        return len(p) != 1 or uvutils.polnum2str(p[0]) not in ["xx", "yy"]

    def get_feed(self, uvdata) -> str:
        """Get the feed to use from the beam, given the UVData object.

        Only applies for an *unpolarized* simulation (for a polarized sim, all feeds
        are used).
        """
        return uvutils.polnum2str(uvdata.polarization_array[0])[0]

    def simulate(self, data_model):
        """
        Calls :func:matvis to perform the visibility calculation.

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

        # The following are antenna positions in the order that they are
        # in the uvdata.data_array
        active_antpos, ant_list = data_model.uvdata.get_ENU_antpos(pick_data_ants=True)


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
        if np.all(data_model.uvdata.data_array == 0):
            # Here, we don't make new memory, because that is just a whole extra copy
            # of the largest array in the calculation. Instead we fill the data_array
            # directly.
            visfull = data_model.uvdata.data_array
        else:
            visfull = np.zeros_like(
                data_model.uvdata.data_array, dtype=self._complex_dtype
            )

        antpairs = data_model.uvdata.get_antpairs()
        antlist = ant_list.tolist()
        antpairs = np.array([[antlist.index(a), antlist.index(b)] for a,b in antpairs])

        for i, freq in enumerate(data_model.freqs):
            # Divide tasks between MPI workers if needed
            if self.mpi_comm is not None and i % nproc != myid:
                continue

            logger.info(f"Simulating Frequency {i + 1}/{len(data_model.freqs)}")

            # Call matvis function to simulate visibilities
            vis = self._matvis(
                antpos=active_antpos,
                freq=freq,
                times=Time(data_model.times, format="jd"),
                skycoords=data_model.sky_model.skycoord,
                telescope_loc=data_model.uvdata.telescope.location,
                I_sky=data_model.sky_model.stokes[0, i].to("Jy").value,
                beam_list=data_model.beams,
                beam_idx=beam_ids,
                beam_spline_opts=data_model.beams.spline_interp_opts,
                precision=self._precision,
                polarized=polarized,
                antpairs=antpairs,
                **self.kwargs,
            )

            logger.info("... re-ordering visibilities...")
            self._reorder_vis(
                req_pols, data_model.uvdata, visfull[:, i], vis, ant_list, polarized
            )

        # Reduce visfull array if in MPI mode
        if self.mpi_comm is not None:
            visfull = self._reduce_mpi(visfull, myid)

        if visfull is data_model.uvdata.data_array:
            # In the case that we were just fulling up the data array the whole time,
            # we return zero, because this will be added to the data_array in the
            # wrapper simulate() function.
            return 0
        else:
            return visfull

    def _reorder_vis(self, req_pols, uvdata: UVData, visfull, vis, ant_list, polarized):

        if (
            uvdata.blts_are_rectangular and
            not uvdata.time_axis_faster_than_bls and
            sorted(req_pols) == req_pols
        ):
            logger.info("Using direct setting of data without reordering")
            # This is the best case scenario -- no need to reorder anything.
            # It is also MUCH MUCH faster!
            vis.shape = (uvdata.Nblts, uvdata.Npols)
            visfull[:] = vis
            return

        logger.info(
            f"Reordering baselines. Pols sorted: {sorted(req_pols) == req_pols}. "
            f"Pols = {req_pols}. blt_order = {uvdata.blt_order}"
        )

        for i, (ant1, ant2) in enumerate(uvdata.get_antpairs()):
            # get all blt indices corresponding to this antpair
            indx = uvdata.antpair2ind(ant1, ant2)
            vis_here = vis[:, i]

            if polarized:
                for p, (p1, p2) in enumerate(req_pols):
                    visfull[indx, p] = vis_here[:, p1, p2]
            else:
                visfull[indx, 0] = vis_here
    @staticmethod
    def _get_req_pols(uvdata, uvbeam, polarized: bool) -> list[tuple[int, int]]:
        if not polarized:
            return [(0, 0)]

        feeds = uvbeam.feed_array
        if isinstance(feeds, np.ndarray):
            feeds = feeds.tolist()  # convert to list if necessary

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

    def compress_data_model(self, data_model: ModelData):
        data_model.uvdata.uvw_array = 0
        # data_model.uvdata.baseline_array = 0
        data_model.uvdata.integration_time = data_model.uvdata.integration_time.item(0)

    def restore_data_model(self, data_model: ModelData):
        uv_obj = data_model.uvdata
        uv_obj.integration_time = np.repeat(
            uv_obj.integration_time, uv_obj.Nbls * uv_obj.Ntimes
        )
        uv_obj.set_uvws_from_antenna_positions()
