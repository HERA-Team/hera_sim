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
    Generate mock instrumental gains.

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


def gen_reflection_coefficient(freqs, amp, dly, phs, conj=False):
    """
    Generate a reflection coefficient.

    The reflection coefficient is described as

    .. math:: \epsilon = A * \exp(2i\pi\tau\nu + i\phi)

    Args:
        freqs (1D ndarray): frequencies [GHz]
        amp (float or ndarray): reflection amplitude
        dly (float or ndarray): reflection delay [nanosec]
        phs (float or ndarray): reflection phase [radian]
        conj (bool, optional): if True, conjugate the reflection coefficient

    Returns:
        complex ndarray: complex reflection gain

    Notes:
        Reflection terms can be fed as floats, in which case output coefficient
        is a 1D array of shape (Nfreqs,) or they can be fed as (Ntimes, 1) ndarrays,
        in which case output coefficient is a 2D narray of shape (Ntimes, Nfreqs)
    """
    # form reflection coefficient
    eps = amp * np.exp(2j * np.pi * freqs * dly + 1j * phs)

    # conjugate
    if conj:
        eps = eps.conj()

    return eps


def gen_reflection_gains(freqs, ants, amp=None, dly=None, phs=None, conj=False):
    """
    Generate a signal chain reflection as an antenna gain.

    A signal chain reflection is a copy of an antenna
    voltage stream at a boosted delay, and can be incorporated
    via a gain term

    .. math::   g_1 = (1 + \epsilon_{11})

    where :math:`\epsilon_{11}` is antenna 1's reflection coefficient
    which can be constructed as

    .. math:: \epsilon_{11} = A_{11} * \exp(2i\pi\tau_{11}\nu + i\phi_{11})

    Args:
        freqs (1D ndarray): frequencies [GHz]
        ants (list of integers): antenna numbers
        amp (list, optional): antenna reflection amplitudes for each antenna. Default is 1.0
        dly (list, optional): antenna reflection delays [nanosec]. Default is 0.0
        phs (lists, optional): antenna reflection phases [radian]. Default is 0.0
        conj (bool, optional): if True, conjugate the reflection coefficients

    Returns:
        dictionary: keys are antenna numbers and values are complex reflection gains

    Notes:
        Reflection terms for each antenna can be fed as a list of floats, in which case 
        the output coefficients are 1D arrays of shape (Nfreqs,) or they can be fed as
        a list of ndarrays of shape (Ntimes, 1), in which case output coefficients are
        2D narrays of shape (Ntimes, Nfreqs)
    """
    # fill in missing kwargs
    if amp is None:
        amp = [0.0 for ai in ants]
    if dly is None:
        dly = [0.0 for ai in ants]
    if phs is None:
        phs = [0.0 for ai in ants]

    # iterate over antennas
    gains = {}
    for i, ai in enumerate(ants):
        # form reflection coefficient
        eps = gen_reflection_coefficient(freqs, amp[i], dly[i], phs[i], conj=conj)
        gains[ai] = (1 + eps)

    return gains


def apply_gains(vis, gains, bl):
    """
    Apply a set of gains to a visibility.

    Args:
        vis (2D complex ndarray): visibility of shape (Ntimes, Nfreqs)
        gains (dict): keys are antenna numbers, vals are complex gain ndarrays (eg. output of :meth:`~gen_gains`)
        bl (2-tuple): antenna-integer pair for the input vis baseline

    Returns:
        2D array: input vis ndarray with gains applied, unless antennas in bl
            doesn't exist in gains, then vis is returned
    """
    # if an antenna doesn't exist, set to one
    if bl[0] not in gains:
        gi = 1.0
    else:
        gi = gains[bl[0]]
    if bl[1] not in gains:
        gj = 1.0
    else:
        gj = gains[bl[1]]

    # return vis if both antennas don't exist in gains
    if (bl[0] not in gains) and (bl[1] not in gains):
        return vis

    # form gain term for bl
    gij = gi * np.conj(gj)

    # reshape if necessary    
    if gij.ndim == 1:
        gij.shape = (1, -1)

    return vis * gij


def gen_whitenoise_xtalk(freqs, amplitude=3.0):
    """
    Generate a white-noise cross-talk model for specified bls.

    Args:
        freqs (ndarray): frequencies of observation [GHz]
        amplitude (float): amplitude of cross-talk in visibility units

    Returns:
        1D ndarray: xtalk model across frequencies

    See Also:
        :meth:`~apply_xtalk`: apply the output of this function to a visibility.
    """
    xtalk = np.convolve(
        noise.white_noise(freqs.size),
        np.ones(50 if freqs.size > 50 else int(freqs.size/2)),
        'same'
    )

    return xtalk * amplitude


def gen_cross_coupling_xtalk(freqs, autovis, amp=None, dly=None, phs=None, conj=False):
    """
    Generate a cross coupling systematic (e.g. crosstalk).

    A cross coupling systematic is the auto-correlation visibility multiplied by a coupling coefficient.
    If :math:`V_{11}` is the auto-correlation visibility of antenna 1, and :math:`\epsilon_{12}`
    is the coupling coefficient, then the cross coupling term in the cross correlation visibility
    takes the form

    .. math::   V_{12} = v_1 v_2^\ast + V_{11}\epsilon_{12}^\ast

    where :math:`\epsilon_{12}` is modeled as a reflection coefficient constructed as

    .. math::   \epsilon_{12} = A_{12} * \exp(2i\pi\tau_{12}\nu + i\phi_{12})

     Args:
        freqs (1D ndarray): frequencies [GHz]
        autovis (2D ndarray): auto-correlation visibility ndarray of shape (Ntimes, Nfreqs)
        amp (float): coupling amplitude
        dly (float): coupling delay [nanosec]
        phs (float): coupling phase [radian]
        conj (bool, optional): if True, conjugate the coupling coefficients

    Returns:
        2D ndarray: xtalk model of shape (Ntimes, Nfreqs)
    """
    # fill in missing kwargs
    if amp is None:
        amp = 0.0
    if dly is None:
        dly = 0.0
    if phs is None:
        phs = 0.0

    # generate coupling coefficient
    eps = gen_reflection_coefficient(freqs, amp, dly, phs, conj=conj)
    if eps.ndim == 1:
        eps = np.reshape(eps, (1, -1))

    return autovis * eps


def apply_xtalk(vis, xtalk):
    """
    Add cross-talk to a visibility.

    Args:
        vis (complex 2D array): visibilities per (LST, frequency)
        xtalk (complex 1D or 2D ndarray): cross-talk model

    Returns:
        complex 2D array: visibilities after cross-talk is applied.
    """
    # if xtalk is a single spectrum, reshape it into 2D array
    if xtalk.ndim == 1:
        xtalk = np.reshape(xtalk, (1, -1))

    return vis + xtalk

