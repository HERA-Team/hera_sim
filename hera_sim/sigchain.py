"""A module for modeling HERA signal chains."""

from . import noise
import numpy as np
import aipy

HERA_NRAO_BANDPASS = np.array(
    [
        -2.04689451e06,
        1.90683718e06,
        -7.41348361e05,
        1.53930807e05,
        -1.79976473e04,
        1.12270390e03,
        -2.91166102e01,
    ]
)  # See "HERA's Passband to First Order"


def gen_bandpass(fqs, ants, gain_spread=0.1):
    """
    Produce a set of mock bandpass gains with variation based around the
    HERA_NRAO_BANDPASS model.

    Args:
        fqs: array-like, shape=(NFREQS,), GHz
            the spectral frequencies of the bandpasses
        ants: list
            the indices/names of the antennas
        gain_spread: float
            the fractional variation in gain harmonics
    Returns:
        g: dictionary
            a dictionary of ant:bandpass pairs where keys are elements
            of ants and bandpasses are complex arrays with shape (NFREQS,)

    See Also:
        :meth:`~gen_gains`: uses this function to generate full gains.
    """
    bp_base = np.polyval(HERA_NRAO_BANDPASS, fqs)
    window = aipy.dsp.gen_window(fqs.size, 'blackman-harris')
    _modes = np.abs(np.fft.fft(window * bp_base))
    g = {}
    for ai in ants:
        delta_bp = np.fft.ifft(noise.white_noise(fqs.size) * _modes * gain_spread)
        g[ai] = bp_base + delta_bp
    return g


def gen_delay_phs(fqs, ants, dly_rng=(-20, 20)):
    """
    Produce a set of mock complex phasors corresponding to cables delays.

    Args:
        fqs: array-like, shape=(NFREQS,), GHz
            the spectral frequencies of the bandpasses
        ants: list
            the indices/names of the antennas
        dly_range: 2-tuple, ns
            the range of the delay
    Returns:
        g: dictionary
            a dictionary of ant:exp(2pi*i*tau*fqs) pairs where keys are elements
            of ants and values are complex arrays with shape (NFREQS,)

    See Also:
        :meth:`~gen_gains`: uses this function to generate full gains.
    """
    phs = {}
    for ai in ants:
        dly = np.random.uniform(dly_rng[0], dly_rng[1])
        phs[ai] = np.exp(2j * np.pi * dly * fqs)
    return phs


def gen_gains(fqs, ants, gain_spread=0.1, dly_rng=(-20, 20)):
    """
    Produce a set of mock bandpasses perturbed around a HERA_NRAO_BANDPASS model
    and complex phasors corresponding to cables delays.

    Args:
        fqs: array-like, shape=(NFREQS,), GHz
            the spectral frequencies of the bandpasses
        ants: list
            the indices/names of the antennas
        gain_spread: float
            the fractional variation in gain harmonics
        dly_range: 2-tuple, ns
            the range of the delay
    Returns:
        g: dictionary
            a dictionary of ant:bandpass * exp(2pi*i*tau*fqs) pairs where
            keys are elements of ants and bandpasses are complex arrays
            with shape (NFREQS,)

    See Also:
        :meth:`~apply_gains`: apply gains from this function to a visibility
    """
    bp = gen_bandpass(fqs, ants, gain_spread)
    phs = gen_delay_phs(fqs, ants, dly_rng)
    return {ai: bp[ai] * phs[ai] for ai in ants}


def apply_gains(vis, gains, bl):
    """
    Apply to a (NTIMES,NFREQS) visibility waterfall the bandpass functions
    for its constituent antennas.

    Args:
        vis: array-like, shape=(NTIMES,NFREQS), Jy
            the visibility waterfall to which gains will be applied
        gains: dictionary
            a dictionary of ant:bandpass * exp(2pi*i*tau*fqs) pairs where
            keys are ants and bandpasses are complex arrays
            with shape (NFREQS,) (e.g. output of :meth:`~gen_gains`)
        bl: 2-tuple
            a (i, j) tuple representing the baseline corresponding to
            this visibility.  g_i * g_j.conj() will be multiplied into vis.
    Returns:
        vis: array-like, shape=(NTIMES,NFREQS)
            the visibility waterfall with gains applied
    """
    gij = gains[bl[0]] * gains[bl[1]].conj()
    gij.shape = (1, -1)
    return vis * gij

def gen_xtalk(fqs, amplitude=3.0):
    """
    Generate a random, crosstalk-like signal as a function of frequency.

    Args:
        fqs: array-like, shape=(NFREQS,), GHz
            the spectral frequencies of the bandpasses
        amplitude: float, default=3.
            a multiplicative scalar to the xtalk amplitude
    Returns:
        xtalk: array-like, shape=(NFREQS,)
            the crosstalk signal

    See Also:
        :meth:`~apply_xtalk`: apply the output of this function to a visibility.
    """
    xtalk = np.convolve(
        noise.white_noise(fqs.size),
        np.ones(50 if fqs.size > 50 else int(fqs.size/2)),
        'same'
    )
    return amplitude * xtalk


def apply_xtalk(vis, xtalk):
    """
    Apply to a (NTIMES,NFREQS) visibility waterfall a crosstalk signal

    Args:
        vis: array-like, shape=(NTIMES,NFREQS), Jy
            the visibility waterfall to which gains will be applied
        xtalk: array-like, shape=(NFREQS,), Jy
            the crosstalk signal to be applied.
    Returns:
        vis: array-like, shape=(NTIMES,NFREQS)
            the visibility waterfall with crosstalk injected'''
    """
    xtalk = np.reshape(xtalk, (1, -1))
    return vis + xtalk
