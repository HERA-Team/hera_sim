from __future__ import division
from builtins import range
import numpy as np
from scipy.interpolate import RectBivariateSpline
import healpy

from . import conversions
from .simulators import VisibilitySimulator

from astropy.constants import c


class VisCPU(VisibilitySimulator):
    """
    vis_cpu visibility simulator.

    This is a fast, simple visibility simulator that is intended to be
    replaced by vis_gpu. It extends :class:`VisibilitySimulator`.
    """

    def __init__(self, bm_pix=100, precision=1, use_gpu=False, **kwargs):
        """
        Parameters
        ----------
        bm_pix : int, optional
            The number of pixels along a side in the beam map when
            converted to (l, m) coordinates. Defaults to 100.
        real_dtype : {np.float32, np.float64}
            Data type for real-valued arrays.
        complex_dtype : {np.complex64, np.complex128}
            Data type for complex-valued arrays.
        **kwargs
            Arguments of :class:`VisibilitySimulator`.
        """

        assert precision in (1,2)
        self._precision = precision
        if precision == 1:
            self._real_dtype = np.float32
            self._complex_dtype = np.complex64
        else:
            self._real_dtype = np.float64
            self._complex_dtype = np.complex128

        if use_gpu:
            from hera_gpu.vis import vis_gpu
            self._vis_cpu = vis_gpu
        else:
            self._vis_cpu = vis_cpu
        self.bm_pix = bm_pix

        super(VisCPU, self).__init__(**kwargs)

        # Convert some arguments to forms more simple for vis_cpu.
        self.antpos = self.uvdata.get_ENU_antpos()[0].astype(self._real_dtype)
        self.freqs = self.uvdata.freq_array[0]

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
            self.__lsts = self.uvdata.lst_array[::self.uvdata.Nbls]

            return self.__lsts

    def validate(self):
        """Checks for correct input format."""
        super(VisCPU, self).validate()

        # This one in particular requires that every baseline is used!
        N = len(self.uvdata.antenna_numbers)
        # N(N-1)/2 unique cross-correlations + N autocorrelations.
        if len(self.uvdata.get_antpairs()) != N * (N + 1) / 2:
            raise ValueError("VisCPU requires using every pair of antennas, "
                             "but the UVData object does not comply.")

        if (len(self.uvdata.data_array) != len(self.uvdata.get_antpairs())
                * len(self.lsts)):
            raise ValueError("VisCPU requires that every baseline uses the "
                             "same LSTS.")

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
        return np.asarray([
            conversions.uvbeam_to_lm(
                self.beams[self.beam_ids[i]], self.freqs, self.bm_pix
            ) for i in range(self.n_ant)
        ])

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
        return np.asarray([np.cos(ra)*np.cos(dec), np.cos(dec)*np.sin(ra),
                         np.sin(dec)])

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

    def _base_simulate(self, crd_eq, I):
        """
        Calls :func:vis_cpu to perform the visibility calculation.

        Returns
        -------
        array_like of self._complex_dtype
            Visibilities. Shape=self.uvdata.data_array.shape.
        """
        eq2tops = self.get_eq2tops()
        beam_lm = self.get_beam_lm()

        visfull = np.zeros_like(self.uvdata.data_array,
                                dtype=self._complex_dtype)

        for i, freq in enumerate(self.freqs):
            vis = self._vis_cpu(
                antpos=self.antpos,
                freq=freq,
                eq2tops=eq2tops,
                crd_eq=crd_eq,
                I_sky=I[i],
                bm_cube=beam_lm[:, i],
                precision=self._precision
            )

            indices = np.triu_indices(vis.shape[1])
            vis_upper_tri = vis[:, indices[0], indices[1]]

            visfull[:, 0, i, 0] = vis_upper_tri.flatten()

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
            self.sky_intensity * healpy.nside2pixarea(self.nside)
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


def vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube,
            precision=1):
    """
    Calculate visibility from an input intensity map and beam model.

    Provided as a standalone function.

    Parameters
    ----------
    antpos : array_like
        Antenna position array. Shape=(NANT, 3).
    freq : float
        Frequency to evaluate the visibilities at [GHz].
    eq2tops : array_like
        Set of 3x3 transformation matrices converting equatorial
        coordinates to topocentric at each
        hour angle (and declination) in the dataset.
        Shape=(NTIMES, 3, 3).
    crd_eq : array_like
        Equatorial coordinates of Healpix pixels, in Cartesian system.
        Shape=(3, NPIX).
    I_sky : array_like
        Intensity distribution on the sky,
        stored as array of Healpix pixels. Shape=(NPIX,).
    bm_cube : array_like
        Beam maps for each antenna. Shape=(NANT, BM_PIX, BM_PIX).
    real_dtype {np.float32, np.float64}
        Data type to use for real-valued arrays.
    complex_dtype {np.complex64, np.complex128}
        Data type to use for complex-valued arrays.

    Returns
    -------
    array_like
        Visibilities. Shape=(NTIMES, NANTS, NANTS).
    """

    assert precision in (1,2)
    if precision == 1:
        real_dtype=np.float32
        complex_dtype=np.complex64
    else:
        real_dtype=np.float64
        complex_dtype=np.complex128
    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)."
    ntimes, ncrd1, ncrd2 = eq2tops.shape
    assert ncrd1 == 3 and ncrd2 == 3, "eq2tops must have shape (NTIMES, 3, 3)."
    ncrd, npix = crd_eq.shape
    assert ncrd == 3, "crd_eq must have shape (3, NPIX)."
    assert I_sky.ndim == 1 and I_sky.shape[0] == npix, \
        "I_sky must have shape (NPIX,)."
    bm_pix = bm_cube.shape[-1]
    assert bm_cube.shape == (
        nant,
        bm_pix,
        bm_pix,
    ), "bm_cube must have shape (NANTS, BM_PIX, BM_PIX)."

    # Intensity distribution (sqrt) and antenna positions. Does not support
    # negative sky.
    Isqrt = np.sqrt(I_sky).astype(real_dtype)
    antpos = antpos.astype(real_dtype)

    ang_freq = 2 * np.pi * freq

    # Empty arrays: beam pattern, visibilities, delays, complex voltages.
    A_s = np.empty((nant, npix), dtype=real_dtype)
    vis = np.empty((ntimes, nant, nant), dtype=complex_dtype)
    tau = np.empty((nant, npix), dtype=real_dtype)
    v = np.empty((nant, npix), dtype=complex_dtype)
    crd_eq = crd_eq.astype(real_dtype)

    bm_pix_x = np.linspace(-1, 1, bm_pix)
    bm_pix_y = np.linspace(-1, 1, bm_pix)

    # Loop over time samples.
    for t, eq2top in enumerate(eq2tops.astype(real_dtype)):
        tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)

        for i in range(nant):
            # Linear interpolation of primary beam pattern.
            spline = RectBivariateSpline(bm_pix_y, bm_pix_x, bm_cube[i], kx=1,
                                         ky=1)
            A_s[i] = spline(ty, tx, grid=False)

        A_s = np.where(tz > 0, A_s, 0)

        # Calculate delays, where TAU = (b * s) / c.
        np.dot(antpos, crd_top, out=tau)
        tau /= c.value

        np.exp(1.0j * (ang_freq * tau), out=v)

        # Complex voltages.
        v *= A_s * Isqrt

        # Compute visibilities (upper triangle only).
        for i in range(len(antpos)):
            np.dot(v[i: i + 1].conj(), v[i:].T, out=vis[t, i: i + 1, i:])

    return vis
