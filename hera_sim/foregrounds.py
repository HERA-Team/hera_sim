"""A module for generating realistic foregrounds."""

import numpy as np
from scipy.interpolate import RectBivariateSpline
import aipy
from . import noise
from . import utils


def diffuse_foreground(Tsky, lsts, fqs, bl_len_ns, bm_poly=noise.HERA_BEAM_POLY, scalar=30.,
                       fr_width=None, fr_max_mult=2.0):
    """
    Need a doc string...
    """
    fr_max = np.max(utils.calc_max_fringe_rate(fqs, bl_len_ns))
    dt = 1.0 / (fr_max_mult * fr_max)  # over-resolve by fr_mult factor
    ntimes = int(np.around(aipy.const.sidereal_day / dt))
    lst_grid = np.linspace(0, 2 * np.pi, ntimes, endpoint=False)
    nos = Tsky(lst_grid, fqs) * noise.white_noise((ntimes, fqs.size))
    nos, ff, frs = utils.rough_fringe_filter(
        nos, lst_grid, fqs, bl_len_ns, fr_width=fr_width
    )
    nos = utils.rough_delay_filter(nos, fqs, bl_len_ns)
    nos /= noise.jy2T(fqs, bm_poly=bm_poly)
    mdl_real = RectBivariateSpline(lst_grid, fqs, scalar * nos.real)
    mdl_imag = RectBivariateSpline(lst_grid, fqs, scalar * nos.imag)
    return mdl_real(lsts, fqs) + 1j * mdl_imag(lsts, fqs)


def pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=1000):
    """
    Need a doc string...
    """
    ras = np.random.uniform(0, 2 * np.pi, nsrcs)
    indices = np.random.normal(-1, 0.5, size=nsrcs)
    mfreq = 0.15
    beam_width = (
        (40 * 60.0) * (mfreq / fqs) / aipy.const.sidereal_day * 2 * np.pi
    )  # XXX hardcoded HERA
    x0, x1, n = 0.3, 300, -1.5
    # Draw flux densities from a power law between 1 and 1000 w/ index of -1.5
    flux_densities = (
        (x1 ** (n + 1) - x0 ** (n + 1)) * np.random.uniform(size=nsrcs) + x0 ** (n + 1)
    ) ** (1.0 / (n + 1))
    vis = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    for ra, flux, index in zip(ras, flux_densities, indices):
        t = np.argmin(np.abs(utils.compute_ha(lsts, ra)))
        dtau = np.random.uniform(
            -0.1 * bl_len_ns, 0.1 * bl_len_ns
        )  # XXX adds a bit to total delay, increasing bl_len_ns
        vis[t, :] += flux * (fqs / mfreq) ** index * np.exp(2j * np.pi * fqs * dtau)
    ha = utils.compute_ha(lsts, 0)
    for fi in xrange(fqs.size):
        bm = np.exp(-ha ** 2 / (2 * beam_width[fi] ** 2))
        bm = np.where(np.abs(ha) > np.pi / 2, 0, bm)
        w = (
            0.9 * bl_len_ns * np.sin(ha) * fqs[fi]
        )  # XXX .9 to offset increase from dtau above
        phs = np.exp(2j * np.pi * w)
        kernel = bm * phs
        vis[:, fi] = np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(vis[:, fi]))
    return vis
