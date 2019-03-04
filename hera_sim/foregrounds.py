"""
A module with functions for generating foregrounds signals.

Each function may take arbitrary parameters, but should return a 2D array of visibilities for the requested baseline
at the requested lsts and frequencies.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from aipy.const import sidereal_day

from . import noise
from . import utils


def diffuse_foreground(Tsky_mdl, lsts, fqs, bl_vec, bm_poly=noise.HERA_BEAM_POLY,
                       standoff=0.0, delay_filter_type='tophat', delay_filter_normalize=None,
                       fringe_filter_type='tophat', **fringe_filter_kwargs):
    """
    Produce a (NTIMES,NFREQS) mock-up of what diffuse foregrounds could look
    like on a baseline of a provided geometric length.

    Args:
        Tsky_mdl (callable): interpolation object
            an interpolation object that returns the sky temperature as a
            function of (lst, freqs).  Called as Tsky_mdl(lsts, fqs).
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the observation to be generated.
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the observation to be generated.
        bl_vec (array-like): shape=(3,), nanosec
            East-North-Up (i.e. Topocentric) baseline vector [East, North, Up]
        bm_poly (polynomial): default=noise.HERA_BEAM_POLY
            a polynomial fit to the solid-angle beam size of the observation
            as a function of frequency.  Used to convert temperatures to Jy.
        standoff (float): baseline horizon buffer [ns] for modeling suprahorizon emission
        delay_filter_type (str): type of delay filter to use, see utils.gen_delay_filter
        delay_filter_normalize (float): delay filter normalization, see utils.gen_delay_filter
        fringe_filter_type (str): type of fringe-rate filter, see utils.gen_fringe_filter
        fringe_filter_kwargs: kwargs given fringe_filter_type, see utils.gen_fringe_filter

    Returns:
        mdl (array-like): shape=(NTIMES,NFREQS)
            mock diffuse foreground visibility spectra vs. time
    """
    # generate a Tsky visibility in time and freq space, convert from K to Jy
    Tsky = Tsky_mdl(lsts, fqs)
    mdl = np.asarray(Tsky * 1e3 / noise.jy2T(fqs, bm_poly=bm_poly), np.complex)

    # If an auto-correlation, return the beam-weighted integrated sky.
    if np.isclose(np.linalg.norm(bl_vec), 0):
        return mdl

    # multiply by white noise
    mdl *= noise.white_noise((len(lsts), len(fqs)))

    # fringe rate filter across time using projected East-West distance
    mdl, fringe_filter = utils.rough_fringe_filter(mdl, lsts, fqs, np.abs(bl_vec[0]), filter_type=fringe_filter_type, **fringe_filter_kwargs)

    # delay filter across freq
    mdl, delay_filter = utils.rough_delay_filter(mdl, fqs, np.linalg.norm(bl_vec), standoff=standoff, filter_type=delay_filter_type, normalize=delay_filter_normalize)

    return mdl


def pntsrc_foreground(lsts, fqs, bl_vec, nsrcs=1000, Smin=0.3, Smax=300,
                      beta=-1.5, spectral_index_mean=-1, spectral_index_std=0.5,
                      reference_freq=0.15):
    """
    Produce a (NTIMES,NFREQS) mock-up of what point-source foregrounds
    could look like on a baseline of a provided geometric length.  Results
    have phase coherence within an observation but not between repeated
    calls of this function (i.e. no phase coherence between baselines).
    Beam width is currently hardcoded for HERA.
    Args:
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the observation to be generated.
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the observation to be generated.
        bl_vec (array-like): shape=(3,), nanosec
            East-North-Up (i.e. Topocentric) baseline vector [East, North, Up]
        nsrcs (float): default=1000.
            the number of mock sources to put on the sky, drawn from a power-law
            of flux-densities with an index of beta. between Smin and Smax
        Smin (float): [Jy], default=0.3
            the minimum flux density to sample
        Smax (float): [Jy], default=300
            the maximum flux density to sample
        beta (float): default=-1.5
            the power-law index of point-source counts versus flux density
        spectral_index_mean (float): default=-1
            the mean spectral index of point sources drawn
        spectral_index_std (float): default=0.5
            the standard deviation of the spectral index of point sources drawn
        reference_freq (float): [GHz], default=0.15
            the frequency from which spectral indices extrapolate

    Returns:
        vis (array-like): shape=(NTIMES,NFREQS)
            mock point-source foreground visibility spectra vs. time'''
    """
    bl_len_ns = np.linalg.norm(bl_vec)
    ras = np.random.uniform(0, 2 * np.pi, nsrcs)
    indices = np.random.normal(spectral_index_mean, spectral_index_std, size=nsrcs)
    mfreq = reference_freq
    beam_width = (40 * 60.) * (mfreq / fqs) / sidereal_day * 2 * np.pi  # XXX hardcoded HERA

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
