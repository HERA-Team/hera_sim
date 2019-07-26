"""
``vis_cpu`` visibility simulator.

This is a fast, simple visibility simulator that
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
import healpy

from . import conversions
from .simulators import VisibilitySimulator


class VisCPU(VisibilitySimulator):

    def __init__(self, bm_pix=31, real_dtype=np.float32, ####### bm_pix was = 100 on default
                 complex_dtype=np.complex64, **kwargs):
        """
        Fast visibility simulator on the CPU.

        Args:
            bm_pix (int, optional): the number of pixels along a side in the
                beam map when converted to (l,m).
            real_dtype: a valid numpy dtype
            complex_dtype: a valid numpy dtype
            **kwargs:
                All arguments of :class:`VisibilitySimulator`.

        """
        self._real_dtype = real_dtype
        self._complex_dtype = complex_dtype
        self.bm_pix = bm_pix


        super(VisCPU, self).__init__(**kwargs)

        # Convert some of our arguments to forms more simple for vis_cpu
        self.antpos = self.uvdata.get_ENU_antpos()[0].astype(self._real_dtype)
        self.freqs = self.uvdata.freq_array[0]

    @property
    def lsts(self):
        try:
            return self.__lsts
        except AttributeError:
            self.__lsts = np.unique(self.uvdata.lst_array)
            return self.__lsts

    def validate(self):
        super(VisCPU, self).validate()

        # This one in particular requires that every baseline is used!
        if len(self.uvdata.get_antpairs()) != len(self.uvdata.antenna_numbers) ** 2:
            raise ValueError("VisCPU requires using every pair of antennas, but the UVData object does not comply")

        if len(self.uvdata.data_array) != len(self.uvdata.get_antpairs()) * len(self.lsts):
            raise ValueError("VisCPU requires that every baseline uses the same LSTS")

    def get_beam_lm(self):
        """
        Obtain the beam pattern in (l,m) co-ordinates for each beam.

        Returns:
              3D array, shape[NANT, BM_PIX, BM_PIX]: the beam pattern in (l,m)
                  for each antenna.

        Note:
            Due to using the verbatim :func:`vis_cpu` function, the beam cube
            must have an entry for each antenna, which is a bit of a waste of
            memory in some cases. If this is changed in the future, this
            method can be modified to only return one matrix for each beam.

        """
        #print ("self.freqs", self.freqs)
        #print("self.bm_pix", self.bm_pix)

        ##############################
        return np.array([
            conversions.uvbeam_to_lm(
                self.beams[self.beam_ids[i]], self.freqs, self.bm_pix
            ) for i in range(self.n_ant)
        ])
        ###############################

    def get_diffuse_crd_eq(self):
        """Calculate the equatorial co-ordinates of the healpix sky pixels (in Cartesian co-ords)."""
        return conversions.healpix_to_crd_eq(self.sky_intensity[0]).astype(self._real_dtype)

    def get_point_source_crd_eq(self):
        ra, dec = self.point_source_pos.T
        return np.array([np.cos(ra)*np.cos(dec), np.cos(dec)*np.sin(ra), np.sin(dec)])

    def get_eq2tops(self):
        """
        Calculate the set of 3x3 transformation matrices converting equatorial
        coords to topocentric at each LST.
        """

        sid_time = np.unique(self.uvdata.lst_array)
        eq2tops = np.empty((len(sid_time), 3, 3), dtype=self._real_dtype)

        for i, st in enumerate(sid_time):
            eq2tops[i] = conversions.eq2top_m(-st, self.uvdata.telescope_lat_lon_alt[0])

        return eq2tops

    def _base_simulate(self, crd_eq, I):
        eq2tops = self.get_eq2tops()
        beam_lm = self.get_beam_lm()

        visfull = np.zeros_like(self.uvdata.data_array, dtype=self._complex_dtype)

        for i, freq in enumerate(self.freqs):

            #print("BEAM_LM", i, beam_lm[:, i])

            vis = vis_cpu(
                antpos=self.antpos,
                freq=freq,
                eq2tops=eq2tops,
                crd_eq=crd_eq,
                I_sky=I[i],
                bm_cube=beam_lm[:, i],
                real_dtype=self._real_dtype,
                complex_dtype=self._complex_dtype
            )

            visfull[:, 0, i, 0] = vis.flatten()

        return visfull

    def _simulate_diffuse(self):
        crd_eq = self.get_diffuse_crd_eq()
        # Multiply intensity by pix area because the algorithm doesn't.
        return self._base_simulate(
            crd_eq,
            self.sky_intensity * healpy.nside2pixarea(self.nside)
        )

    def _simulate_points(self):
        crd_eq = self.get_point_source_crd_eq()
        return self._base_simulate(crd_eq, self.point_source_flux)

    def _simulate(self):
        vis = 0
        if self.sky_intensity is not None:
            vis += self._simulate_diffuse()
        if self.point_source_flux is not None:
            vis += self._simulate_points()
        return vis


def vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float32,
            complex_dtype=np.complex64):
    """
    Calculate visibility from an input intensity map and beam model.

    Provided as a standalone function

    Args:
        antpos (array_like, shape: (NANT, 3)): antenna position array.
        freq (float): frequency to evaluate the visibilities at [GHz].
        eq2tops (array_like, shape: (NTIMES, 3, 3)): Set of 3x3 transformation
            matrices converting equatorial coordinates to topocentric at each
            hour angle (and declination) in the dataset.
        crd_eq (array_like, shape: (3, NPIX)):
            equatorial coordinates of Healpix pixels, in Cartesian system.
        I_sky (array_like, shape: (NPIX,)): intensity distribution on the sky,
            stored as array of Healpix pixels.
        bm_cube (array_like, shape: (NANT, BM_PIX, BM_PIX)):
            beam maps for each antenna.
        real_dtype (dtype, optional):
            data type to use for real-valued arrays.
        complex_dtype (dtype, optional):
            data type to use for complex-valued arrays.

    Returns:
        array_like, shape(NTIMES, NANTS, NANTS): visibilities
    """
    
    ####################################################################
    #print("I_sky", I_sky)
    #print("antpos", antpos)
    #print("freq", freq)
    #print("eq2tops", eq2tops)
    #print("crd_eq", crd_eq)
    #print("bm_cube", bm_cube)

    #print("MAX OF BM_CUBE", np.max(bm_cube))
    #bm_cube = np.ones_like(bm_cube)

    ####################################################################
    
    
    
    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)"
    ntimes, ncrd1, ncrd2 = eq2tops.shape
    assert ncrd1 == 3 and ncrd2 == 3, "eq2tops must have shape (NTIMES, 3, 3)"
    ncrd, npix = crd_eq.shape
    assert ncrd == 3, "crd_eq must have shape (3, NPIX)"
    assert I_sky.ndim == 1 and I_sky.shape[0] == npix, "I_sky must have shape (NPIX,)"
    bm_pix = bm_cube.shape[-1]
    assert bm_cube.shape == (
        nant,
        bm_pix,
        bm_pix,
    ), "bm_cube must have shape (NANTS, BM_PIX, BM_PIX)"

    # Intensity distribution (sqrt) and antenna positions
    Isqrt = np.sqrt(I_sky).astype(real_dtype)  # XXX does not support negative sky
    antpos = antpos.astype(real_dtype)
    ang_freq = 2 * np.pi * freq

    # Empty arrays: beam pattern, visibilities, delays, complex voltages
    A_s = np.empty((nant, npix), dtype=real_dtype)
    vis = np.empty((ntimes, nant, nant), dtype=complex_dtype)
    tau = np.empty((nant, npix), dtype=real_dtype)
    v = np.empty((nant, npix), dtype=complex_dtype)
    crd_eq = crd_eq.astype(real_dtype)

    bm_pix_x = np.linspace(-1, 1, bm_pix)
    bm_pix_y = np.linspace(-1, 1, bm_pix)

    # Loop over time samples
    for t, eq2top in enumerate(eq2tops.astype(real_dtype)):
        tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)
        for i in range(nant):
            # Linear interpolation of primary beam pattern
            spline = RectBivariateSpline(bm_pix_y, bm_pix_x, bm_cube[i], kx=1, ky=1)
            A_s[i] = spline(ty, tx, grid=False)

        A_s = np.where(tz > 0, A_s, 0)

        # Calculate delays
        np.dot(antpos, crd_top, out=tau)
        np.exp((1.0j * ang_freq) * tau, out=v)

        # Complex voltages
        v *= A_s * Isqrt

        # Compute visibilities (upper triangle only)
        for i in range(len(antpos)):
            np.dot(v[i: i + 1].conj(), v[i:].T, out=vis[t, i: i + 1, i:])

    # Conjugate visibilities
    np.conj(vis, out=vis)

    # Fill in whole visibility matrix from upper triangle
    for i in range(nant):
        vis[:, i + 1:, i] = vis[:, i, i + 1:].conj()
    
    return vis
