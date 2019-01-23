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


def gen_bandpass(freqs, ants, gain_spread=0.1):
    """
    Generate mock bandpasses.

    Args:
        freqs (ndarray): frequencies of observation [GHz]
        ants (sequence of ints): sequence of antenna numbers.
        gain_spread (float): controls the amplitude of random offsets of the gain from mean bandpass.

    Returns:
        dict: keys are antenna numbers, values are arrays of gain per frequency.

    See Also:
        :meth:`~gen_gains`: uses this function to generate full gains.
    """
    bp_base = np.polyval(HERA_NRAO_BANDPASS, freqs)
    window = aipy.dsp.gen_window(freqs.size, "blackman-harris")
    _modes = np.abs(np.fft.fft(window * bp_base))
    g = {}
    for ai in ants:
        delta_bp = np.fft.ifft(noise.white_noise(freqs.size) * _modes * gain_spread)
        g[ai] = bp_base + delta_bp
    return g


def gen_delay_phs(freqs, ants, dly_rng=(-20, 20)):
    """
    Generate mock delay phases.

    Args:
        freqs (ndarray): frequencies of observation [GHz]
        ants (sequence of ints): sequence of antenna numbers.
        dly_rng (2-tuple): range of the delay.

    Returns:
        dict: keys are antenna numbers, values are arrays of phase per frequency.

    See Also:
        :meth:`~gen_gains`: uses this function to generate full gains.
    """
    phs = {}
    for ai in ants:
        dly = np.random.uniform(dly_rng[0], dly_rng[1])
        phs[ai] = np.exp(2j * np.pi * dly * freqs)
    return phs


def gen_gains(freqs, ants, gain_spread=0.1, dly_rng=(-20, 20)):
    """
    Generate mock gains.

    Args:
        freqs (ndarray): frequencies of observation [GHz]
        ants (sequence of ints): sequence of antenna numbers.
        gain_spread (float): controls the amplitude of random offsets of the gain from mean bandpass.
        dly_rng (2-tuple): range of the delay.

    Returns:
        dict: keys are antenna numbers, values are arrays of gain per frequency.

    See Also:
        :meth:`~apply_gains`: apply gains from this function to a visibility
    """
    bp = gen_bandpass(freqs, ants, gain_spread)
    phs = gen_delay_phs(freqs, ants, dly_rng)
    return {ai: bp[ai] * phs[ai] for ai in ants}


def apply_gains(vis, gains, bl):
    """
    Apply a set of gains per frequency to a visibility.

    Args:
        vis (2D array): visibilities per (LST, frequency) [Jy]
        gains (dict): keys are antenna numbers, vals are arrays of gain per frequency (eg. output of :meth:`~gen_gains`
        bl (2-tuple): pair of integers specifying the antenna numbers in a particular baseline.

    Returns:
        2D array: visibilities per (LST, frequency) with gains applied [Jy]
    """
    gij = gains[bl[0]] * gains[bl[1]].conj()
    gij.shape = (1, -1)
    return vis * gij


def gen_xtalk(freqs, amplitude=3.0):
    """
    Generate cross-talk for a range of frequencies.

    Args:
        freqs (ndarray): frequencies of observation [GHz]
        amplitude (float): amplitude of cross-talk [Jy]

    Returns:
        complex ndarray: cross-talk per frequency [Jy].

    See Also:
        :meth:`~apply_xtalk`: apply the output of this function to a visibility.
    """
    # TODO: freqs really should just be n_freqs.
    xtalk = np.convolve(noise.white_noise(freqs.size), np.ones(50), "same")
    return amplitude * xtalk


def apply_xtalk(vis, xtalk):
    """
    Apply cross-talk to a visibility.

    Args:
        vis (complex 2D array): visibilities per (LST, frequency) [Jy]
        xtalk (complex ndarray): cross-talk per frequency [Jy]

    Returns:
        complex 2D array: visibilities after cross-talk is applied.
    """
    xtalk = np.reshape(xtalk, (1, -1))
    return vis + xtalk
