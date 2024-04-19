"""Wrapper for matvis visibility simulator."""

from __future__ import annotations

import fftvis
import logging
import numpy as np
from astropy.time import Time
from fftvis.beams import _evaluate_beam
from fftvis.utils import get_pos_reds
from matvis import conversions as convs

from .matvis import MatVis
from .simulators import ModelData, VisibilitySimulator

logger = logging.getLogger(__name__)


class FFTVis(MatVis):
    """
    fftvis visibility simulator.

    This is a fast, non-uniform FFT-based visibility simulator.

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
        The "unpolarized" output of ``fftvis`` is expected to be XX polarization, which
        corresponds to whatever the UVData object considers to be the x-direction
        (default East).
    precision : int, optional
        Which precision level to use for floats and complex numbers.
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128
    use_gpu : bool, optional
        Whether to use the GPU version of fftvis or not. Default: False.
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
    # __version__ = __version__ # Fill in the version number here

    def __init__(
        self,
        precision: int = 1,
        use_gpu: bool = False,
        mpi_comm=None,
        ref_time: str | Time | None = None,
        correct_source_positions: bool | None = None,
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

        if use_gpu:
            raise ImportError("GPU acceleration is currently not supported for fftvis.")

        self._fftvis = fftvis.simulate.simulate

        self.mpi_comm = mpi_comm
        self.ref_time = ref_time
        self.correct_source_positions = (
            (ref_time is not None)
            if correct_source_positions is None
            else correct_source_positions
        )
        self.check_antenna_conjugation = check_antenna_conjugation
        self._functions_to_profile = (self._fftvis, _evaluate_beam)
        self.kwargs = kwargs

    def validate(self, data_model: ModelData):
        """Checks for correct input format."""
        # N(N-1)/2 unique cross-correlations + N autocorrelations.
        # if data_model.uvdata.Nbls != data_model.n_ant * (data_model.n_ant + 1) / 2:
        #    raise ValueError(
        #        "FFTVis requires using every pair of antennas, "
        #        "but the UVData object does not comply."
        #    )

        logger.info("Checking baseline-time axis shape")
        if len(data_model.uvdata.data_array) != len(
            data_model.uvdata.get_antpairs()
        ) * len(data_model.lsts):
            raise ValueError("FFTVis requires that every baseline uses the same LSTS.")

        if self.check_antenna_conjugation:
            logger.info("Checking antenna conjugation")
            # TODO: the following is extremely slow. If possible, it would be good to
            # find a better way to do it.
            if any(
                len(data_model.uvdata.antpair2ind(ai, aj)) > 0
                and len(data_model.uvdata.antpair2ind(aj, ai)) > 0
                for ai, aj in data_model.uvdata.get_antpairs()
                if ai != aj
            ):
                raise ValueError(
                    "MatVis requires that baselines be in a conjugation in which "
                    "antenna order doesn't change with time!"
                )

        if len(data_model.beams) != 1:
            raise ValueError("FFTVis only supports a single beam.")

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
                nfeeds = uvbeam.data_array.shape[1 if uvbeam.future_array_shapes else 2]
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

        # Estimate size of the FFT grid used to compute the visibilities
        active_antpos, _ = data_model.uvdata.get_ENU_antpos(pick_data_ants=True)
        reds = get_pos_reds(active_antpos)
        max_blx = max([np.abs(red[0][0] - red[1][0]) for red in reds])
        max_bly = max([np.abs(red[0][1] - red[1][1]) for red in reds])

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

    def simulate(self, data_model):
        """
        Calls :func:fftvis to perform the visibility calculation.

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
            ra, dec = self.correct_point_source_pos(data_model)
            logger.info("Done correcting source positions.")
        else:
            ra, dec = data_model.sky_model.ra, data_model.sky_model.dec

        logger.info("Getting Equatorial Coordinates")
        crd_eq = convs.point_source_crd_eq(ra, dec)

        # Convert equatorial to topocentric coords
        logger.info("Getting Rotation Matrices")
        eq2tops = self.get_eq2tops(data_model.uvdata, data_model.lsts)

        # The following are antenna positions in the order that they are
        # in the uvdata.data_array
        active_antpos_array, ant_list = data_model.uvdata.get_ENU_antpos(
            pick_data_ants=True
        )
        active_antpos = {
            ant_index: antpos
            for ant_index, antpos in zip(ant_list, active_antpos_array)
        }

        # Get pixelized beams if required
        logger.info("Preparing Beams...")
        beam = convs.prepare_beam(
            data_model.beams[0],
            polarized=polarized,
            use_feed=feed,
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

        for i, freq in enumerate(data_model.freqs):
            # Divide tasks between MPI workers if needed
            if self.mpi_comm is not None and i % nproc != myid:
                continue

            logger.info(f"Simulating Frequency {i+1}/{len(data_model.freqs)}")

            # Call fftvis function to simulate visibilities
            vis = self._fftvis(
                ants=active_antpos,
                freqs=np.array([freq]),
                eq2tops=eq2tops,
                crd_eq=crd_eq,
                beam=beam,
                fluxes=data_model.sky_model.stokes[0, [i]].to("Jy").value.T,
                beam_spline_opts=data_model.beams.spline_interp_opts,
                precision=self._precision,
                polarized=polarized,
                **self.kwargs,
            )[0]

            logger.info("... re-ordering visibilities...")
            self._reorder_vis(
                req_pols, data_model.uvdata, visfull[:, 0, i], vis, ant_list, polarized
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
