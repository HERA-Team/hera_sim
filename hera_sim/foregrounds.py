"""
A module with functions for generating foregrounds signals.

Each function may take arbitrary parameters, but should return a 2D array of visibilities for the requested baseline
at the requested lsts and frequencies.
"""

import aipy
import numpy as np

from . import noise
from . import utils


def diffuse_foreground(Tsky_mdl, lsts, fqs, bl_vec, bm_poly=noise.HERA_BEAM_POLY,
                       standoff=0.0, delay_filter_type='tophat',
                       fringe_filter_type='tophat', **fringe_filter_kwargs):
    """
    Model diffuse foreground visibility.

    Args:
        Tsky_mdl (RectBivariateSpline): Model of total autocorrelation power, see noise.HERA_Tsky_mdl
        lsts (ndarray): LSTs [radians]
        fqs (ndarray): Frequencies [GHz]
        bl_vec (ndarray): East-North-Up (i.e. Topocentric) baseline vector in nanoseconds [East, North, Up]
        bm_poly (ndarray): beam scalar polynomial object, see noise.HERA_BEAM_POLY
        standoff (float): baseline horizon buffer [ns] for modeling suprahorizon emission
        delay_filter_type (str): type of delay filter to use, see utils.gen_delay_filter()
        fringe_filter_type (str): type of fringe-rate filter, see utils.gen_fringe_filter()
        fringe_filter_kwargs: kwargs given fringe_filter_type, see utils.gen_fringe_filter()

    Returns:
        data (ndarray): diffuse foreground visibility
    """
    # generate a Tsky visibility in time and freq space, convert from K to Jy
    Tsky = Tsky_mdl(lsts, fqs)
    data = np.asarray(Tsky * 1e3 / noise.jy2T(fqs, bm_poly=bm_poly), np.complex)

    # multiply by white noise
    data *= noise.white_noise((len(lsts), len(fqs)))

    # fringe rate filter across time using projected East-West distance
    data, fringe_filter = utils.rough_fringe_filter(data, lsts, fqs, np.abs(bl_vec[0]), filter_type=fringe_filter_type, **fringe_filter_kwargs)

    # delay filter across freq
    data, delay_filter = utils.rough_delay_filter(data, fqs, np.linalg.norm(bl_vec), standoff=standoff, filter_type=delay_filter_type)

    return data


def pntsrc_foreground(lsts, fqs, bl_vec, nsrcs=1000, Smin=0.3, Smax=300,
                      beta=-1.5, spectral_index_mean=-1, spectral_index_std=0.5,
                      reference_freq=0.15):
    """
    Generate visibilities from randomly placed point sources.

    Point sources drawn from a power-law source count distribution from 0.3 to 300 Jy, with index -1.5

    Args:
        lsts (ndarray): LSTs [radians]
        fqs (ndarray): frequencies [GHz]
        bl_vec (ndarray): East-North-Up (i.e. Topocentric) baseline vector in nanoseconds [East, North, Up]
        nsrcs (int): number of sources to place in the sky

    Returns:
        2D ndarray : visibilities at each lst, fq pair.
    """
    bl_len_ns = bl_vec[0]
    ras = np.random.uniform(0, 2 * np.pi, nsrcs)
    indices = np.random.normal(spectral_index_mean, spectral_index_std, size=nsrcs)
    mfreq = reference_freq
    beam_width = (40 * 60.) * (mfreq / fqs) / aipy.const.sidereal_day * 2 * np.pi  # XXX hardcoded HERA

    # Draw flux densities from a power law between Smin and Smax with a slope of beta.
    flux_densities = ((Smax ** (beta + 1) - Smin ** (beta + 1)) * np.random.uniform(size=nsrcs) + Smin ** (
                beta + 1)) ** (1. / (beta + 1))

    vis = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    for ra, flux, index in zip(ras, flux_densities, indices):
        t = np.argmin(np.abs(utils.compute_ha(lsts, ra)))
        dtau = np.random.uniform(-.1 * bl_len_ns, .1 * bl_len_ns)  # XXX adds a bit to total delay, increasing bl_len_ns
        vis[t, :] += flux * (fqs / mfreq) ** index * np.exp(2j * np.pi * fqs * dtau)
    ha = utils.compute_ha(lsts, 0)
    for fi in xrange(fqs.size):
        bm = np.exp(-ha ** 2 / (2 * beam_width[fi] ** 2))
        bm = np.where(np.abs(ha) > np.pi / 2, 0, bm)
        w = .9 * bl_len_ns * np.sin(ha) * fqs[fi]  # XXX .9 to offset increase from dtau above

        phs = np.exp(2j * np.pi * w)
        kernel = bm * phs
        vis[:, fi] = np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(vis[:, fi]))

    return vis
