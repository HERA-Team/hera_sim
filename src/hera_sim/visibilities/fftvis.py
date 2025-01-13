"""Wrapper for matvis visibility simulator."""

from __future__ import annotations

import itertools
import logging

import fftvis
import numpy as np
from astropy.time import Time
from fftvis.beams import _evaluate_beam
from matvis.core.beams import prepare_beam_unpolarized
from pyuvdata import utils as uvutils

from .matvis import MatVis
from .simulators import ModelData, VisibilitySimulator

logger = logging.getLogger(__name__)


class FFTVis(VisibilitySimulator):
    """
    fftvis visibility simulator.

    This is a fast visibility simulator based on the Flatiron Non-Uniform Fast Fourier
    Transform (https://github.com/flatironinstitute/finufft). This class calls the
    fftvis package (https://github.com/tyler-a-cox/fftvis) which utilizes the finufft
    algorithm to evaluate the measurement equation by gridding and fourier transforming
    an input sky model. The simulated visibilities agree with matvis to high precision,
    and are often computed more quickly than matvis. FFTVis is particularly well-suited
    for simulations with compact arrays with large numbers of antennas, and sky models
    with many sources.

    Parameters
    ----------
    precision : int, optional
        Which precision level to use for floats and complex numbers.
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128
    mpi_comm : MPI communicator
        MPI communicator, for parallelization.
    check_antenna_conjugation
        Whether to check the antenna conjugation. Default is True. This is a fairly
        heavy operation if there are many antennas and/or many times, and can be
        safely ignored if the data_model was created from a config file.
    **kwargs
        Passed through to `:func:fftvis.simulate.simulate` function.

    """

    conjugation_convention = "ant1<ant2"
    time_ordering = "time"
    _functions_to_profile = (fftvis.simulate.simulate, _evaluate_beam)
    diffuse_ability = False
    __version__ = "1.0.0"  # Fill in the version number here

    def __init__(
        self,
        *,
        precision: int = 2,
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

        self.mpi_comm = mpi_comm
        self.check_antenna_conjugation = check_antenna_conjugation
        self.kwargs = kwargs

    def _check_if_polarized(self, data_model: ModelData) -> bool:
        p = data_model.uvdata.polarization_array
        # We only do a non-polarized simulation if UVData has only XX or YY polarization
        return len(p) != 1 or uvutils.polnum2str(p[0]) not in ["xx", "yy"]

    def validate(self, data_model: ModelData):
        """Checks for correct input format."""
        logger.info("Checking baseline-time axis shape")
        if not data_model.uvdata.blts_are_rectangular:
            raise ValueError("FFTVis requires that every baseline uses the same LSTS.")

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
                    "FFTVis requires that baselines be in a conjugation in which "
                    "antenna order doesn't change with time!"
                )

        if len(data_model.beams) != 1:
            raise ValueError("FFTVis only supports a single beam.")

        uvbeam = data_model.beams[0]  # Representative beam
        uvdata = data_model.uvdata

        # Check that the UVData object is in the correct format
        # if not uvdata.future_array_shapes:
        #    uvdata.use_future_array_shapes()

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

        if self._check_if_polarized(data_model):
            # Number of feeds must be two if doing polarized
            try:
                nfeeds = uvbeam.data_array.shape[1]
            except AttributeError:
                nfeeds = uvbeam.beam.Nfeeds

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

        # Estimate size of the FFT grid used to compute the visibilities
        active_antpos_array, _ = data_model.uvdata.get_ENU_antpos(pick_data_ants=True)
        # Estimate the size of the grid used to compute the visibilities
        max_blx, max_bly, _ = np.abs(
            active_antpos_array.max(axis=0) - active_antpos_array.min(axis=0)
        )
        avg_freq = np.mean(data_model.freqs)
        n_gridx = int(8 * avg_freq * max_blx / 3e8)  # number of grid points in u/l axis
        n_gridy = int(8 * avg_freq * max_bly / 3e8)  # number of grid points in v/m axis

        try:
            nbmpix = bm.data_array[..., 0, :].size
        except AttributeError:
            nbmpix = 0

        all_floats = (
            nf * nt * nfd**2 * nant**2  # visibilities
            + n_gridx * n_gridy  # FFT grid
            + nf * nbeam * nbmpix  # raw beam
            + nax * nfd * nbeam * nsrc / 2  # interpolated beam
            + 3 * nant  # antenna positions
            + nsrc * nf  # source fluxes
            + nt * 9  # rotation matrices
            + 3 * nsrc
            + 3 * nsrc  # source positions (topo and eq)
        )

        return all_floats * self._precision * 4 / 1024**3

    def get_feed(self, uvdata) -> str:
        """Get the feed to use from the beam, given the UVData object.

        Only applies for an *unpolarized* simulation (for a polarized sim, all feeds
        are used).
        """
        return uvutils.polnum2str(uvdata.polarization_array[0])[0]

    @staticmethod
    def _get_req_pols(uvdata, uvbeam, polarized: bool) -> list[tuple[int, int]]:
        return MatVis._get_req_pols(uvdata, uvbeam, polarized)

    def simulate(self, data_model):
        """
        Calls :func:`fftvis` to perform the visibility calculation.

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

        ra, dec = data_model.sky_model.ra.rad, data_model.sky_model.dec.rad

        # The following are antenna positions in the order that they are
        # in the uvdata.data_array
        active_antpos_array, ant_list = data_model.uvdata.get_ENU_antpos(
            pick_data_ants=True
        )
        active_antpos = dict(zip(ant_list, active_antpos_array))

        # since pyuvdata v3, get_antpairs always returns antpairs in the right order.
        antpairs = data_model.uvdata.get_antpairs()

        # Get pixelized beams if required
        logger.info("Preparing Beams...")
        if not polarized:
            beam = prepare_beam_unpolarized(data_model.beams[0], use_feed=feed)
        else:
            beam = data_model.beams[0]

        # Get all the polarizations required to be simulated.
        req_pols = self._get_req_pols(data_model.uvdata, beam, polarized=polarized)

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

        for i, freq in enumerate(data_model.freqs):
            # Divide tasks between MPI workers if needed
            if self.mpi_comm is not None and i % nproc != myid:
                continue

            logger.info(f"Simulating Frequency {i + 1}/{len(data_model.freqs)}")

            # Call fftvis function to simulate visibilities
            vis = fftvis.simulate.simulate(
                ants=active_antpos,
                freqs=np.array([freq]),
                ra=ra,
                dec=dec,
                times=data_model.times,
                telescope_loc=data_model.uvdata.telescope.location,
                beam=beam,
                fluxes=data_model.sky_model.stokes[0, [i]].to("Jy").value.T,
                beam_spline_opts=data_model.beams.spline_interp_opts,
                precision=self._precision,
                polarized=polarized,
                baselines=antpairs,
                **self.kwargs,
            )[0]

            logger.info("... re-ordering visibilities...")
            self._reorder_vis(
                req_pols, data_model.uvdata, visfull[:, i], vis, antpairs, polarized
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

    def _reorder_vis(self, req_pols, uvdata, visfull, vis, antpairs, polarized):
        if polarized:
            if uvdata.time_axis_faster_than_bls:
                vis = vis.transpose((3, 0, 1, 2))
            else:
                vis = vis.transpose((0, 3, 1, 2))

        if polarized:
            for p, (p1, p2) in enumerate(req_pols):
                visfull[:, p] = vis[..., p1, p2].reshape(-1)
        else:
            visfull[:, 0] = vis.reshape(-1)
