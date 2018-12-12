import numpy as np
from . import foregrounds, rfi, noise, sigchain
from scipy.interpolate import RectBivariateSpline

DEFAULT_LSTS = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
DEFAULT_FQS = np.linspace(0.1, 0.2, 1024, endpoint=False)


def vis_cpu(
    antpos,
    freq,
    eq2tops,
    crd_eq,
    I_sky,
    bm_cube,
    real_dtype=np.float32,
    complex_dtype=np.complex64,
):
    """
    Calculate visibility from an input intensity map and beam model.
    
    Parameters
    ----------
    antpos : array_like, shape: (NANT, 3)
        Antenna position array.
    
    freq : float
        Frequency to evaluate the visibilities at, in GHz.
    
    eq2tops : array_like, shape: (NTIMES, 3, 3)
        Set of 3x3 transformation matrices converting equatorial coordinates 
        to topocentric at each hour angle (and declination) in the dataset.
    
    crd_eq : array_like, shape: (3, NPIX)
        Equatorial coordinates of Healpix pixels.
    
    I_sky : array_like, shape: (NPIX,)
        Intensity distribution on the sky, stored as numpy array of Healpix 
        pixels.
    
    bm_cube : array_like, shape: (NANT, BM_PIX, BM_PIX)
        Beam maps for each antenna.
    
    real_dtype, complex_dtype : dtype, optional
        Data type to use for real and complex-valued arrays. Defaults: 
        np.float32, np.complex64.
    
    Returns
    -------
    vis : array_like, shape: (NTIMES, NANTS, NANTS)
        Visibilities.
    """
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
    ang_freq = 2 * np.pi

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
        for i in xrange(nant):
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
        for i in xrange(len(antpos)):
            np.dot(v[i : i + 1].conj(), v[i:].T, out=vis[t, i : i + 1, i:])

    # Conjugate visibilities
    np.conj(vis, out=vis)

    # Fill in whole visibility matrix from upper triangle
    for i in xrange(nant):
        vis[:, i + 1 :, i] = vis[:, i, i + 1 :].conj()

    return vis


def hmap_to_bm_cube(hmaps, beam_px=63):
    nant = len(hmaps)
    bm_cube = np.empty((nant, beam_px, beam_px), dtype=np.float32)
    # X is 3rd dim, Y is 2nd dim
    tx = np.linspace(-1, 1, beam_px, dtype=np.float32)
    tx = np.resize(tx, (beam_px, beam_px))
    ty = tx.T.copy()
    tx = tx.flatten()
    ty = ty.flatten()
    txty_sqr = tx ** 2 + ty ** 2
    tz = np.where(txty_sqr < 1, np.sqrt(1 - txty_sqr), -1)
    for i, hi in enumerate(hmaps):
        bmi = np.where(tz > 0, hi[tx, ty, tz], 0) / np.max(hi.map)
        bm_cube[i] = np.reshape(bmi, (beam_px, beam_px))
    return bm_cube


def aa_to_eq2tops(aa, jds):
    eq2tops = np.empty((len(jds), 3, 3), dtype=np.float32)
    for i, jd in enumerate(jds):
        aa.set_jultime(jd)
        eq2tops[i] = aa.eq2top_m
    return eq2tops


def hmap_to_crd_eq(h):
    px = np.arange(h.npix())
    crd_eq = np.array(h.px2crd(px, 3), dtype=np.float32)
    return crd_eq


def hmap_to_I(h):
    return h[np.arange(h.npix())].astype(np.float32)


def make_hera_obs(
    aa,
    lsts=DEFAULT_LSTS,
    fqs=DEFAULT_FQS,
    pols=["xx", "yy"],
    T_rx=150.0,
    inttime=10.7,
    rfi_impulse=0.02,
    rfi_scatter=0.001,
    nsrcs=200,
    gain_spread=0.1,
    dly_rng=(-20, 20),
    xtalk=3.0,
):
    ants = list(
        set(
            [
                ant
                for bls in reds
                for bl in bls
                for ant in [(bl[0], bl[2][0]), (bl[1], bl[2][1])]
            ]
        )
    )
    data, true_vis = {}, {}
    if gains is None:
        gains = {}
    else:
        gains = deepcopy(gains)
    for ant in ants:
        gains[ant] = gains.get(ant, 1 + gain_scatter * noise((1,))) * np.ones(
            shape, dtype=np.complex
        )
    for bls in reds:
        true_vis[bls[0]] = noise(shape)
        for (i, j, pol) in bls:
            data[(i, j, pol)] = (
                true_vis[bls[0]] * gains[(i, pol[0])] * gains[(j, pol[1])].conj()
            )
    return gains, true_vis, data


# XXX from hera_cal.redcal
def sim_red_data(reds, gains=None, shape=(10, 10), gain_scatter=0.1):
    """ Simulate noise-free random but redundant (up to differing gains) visibilities.

        Args:
            reds: list of lists of baseline-pol tuples where each sublist has only
                redundant pairs
            gains: pre-specify base gains to then scatter on top of in the
                {(index,antpol): np.array} format. Default gives all ones.
            shape: tuple of (Ntimes, Nfreqs). Default is (10,10).
            gain_scatter: Relative amplitude of per-antenna complex gain scatter. Default is 0.1.

        Returns:
            gains: true gains used in the simulation in the {(index,antpol): np.array} format
            true_vis: true underlying visibilities in the {(ind1,ind2,pol): np.array} format
            data: simulated visibilities in the {(ind1,ind2,pol): np.array} format
    """

    data, true_vis = {}, {}
    ants = list(
        set(
            [
                ant
                for bls in reds
                for bl in bls
                for ant in [(bl[0], bl[2][0]), (bl[1], bl[2][1])]
            ]
        )
    )
    if gains is None:
        gains = {}
    else:
        gains = deepcopy(gains)
    for ant in ants:
        gains[ant] = gains.get(ant, 1 + gain_scatter * noise((1,))) * np.ones(
            shape, dtype=np.complex
        )
    for bls in reds:
        true_vis[bls[0]] = noise(shape)
        for (i, j, pol) in bls:
            data[(i, j, pol)] = (
                true_vis[bls[0]] * gains[(i, pol[0])] * gains[(j, pol[1])].conj()
            )
    return gains, true_vis, data
