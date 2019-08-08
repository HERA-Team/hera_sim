"""A module for modeling HERA signal chains."""

import numpy as np
import aipy
import warnings

from . import noise
from .interpolators import _read_npy
from .defaults import _defaults


@_defaults
def _get_hera_bandpass(datafile="HERA_H1C_BANDPASS.npy"):
    return _read_npy(datafile)

# turns out this will fix HERA_NRAO_BANDPASS as the H1C bandpass
# see "HERA's Passband to First Order" for info on how the
# bandpass was modeled for H1C
HERA_NRAO_BANDPASS = _get_hera_bandpass()

def gen_bandpass(fqs, ants, gain_spread=0.1):
    """
    Produce a set of mock bandpass gains with variation based around the
    HERA_NRAO_BANDPASS model.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the bandpasses
        ants (iterable):
            the indices/names of the antennas
        gain_spread (float): default=0.1
            the fractional variation in gain harmonics
    Returns:
        g (dictionary): 
            a dictionary of ant:bandpass pairs where keys are elements
            of ants and bandpasses are complex arrays with shape (NFREQS,)

    See Also:
        :meth:`~gen_gains`: uses this function to generate full gains.
    """
    bp_base = np.polyval(_get_hera_bandpass(), fqs)
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
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the bandpasses
        ants (iterable):
            the indices/names of the antennas
        dly_range (2-tuple): ns
            the range of the delay
    Returns:
        g (dictionary):
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
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the bandpasses
        ants (iterable): 
            the indices/names of the antennas
        gain_spread (float): default=0.1
            the fractional variation in gain harmonics
        dly_range (2-tuple): ns
            the range of the delay

    Returns:
        g (dictionary):
            a dictionary of ant:bandpass * exp(2pi*i*tau*fqs) pairs where
            keys are elements of ants and bandpasses are complex arrays
            with shape (NFREQS,)

    See Also:
        :meth:`~apply_gains`: apply gains from this function to a visibility
    """
    bp = gen_bandpass(fqs, ants, gain_spread)
    phs = gen_delay_phs(fqs, ants, dly_rng)
    return {ai: bp[ai] * phs[ai] for ai in ants}


def gen_reflection_coefficient(fqs, amp, dly, phs, conj=False):
    r"""
    Generate a reflection coefficient.

    The reflection coefficient is described as

    .. math:: \epsilon = A * \exp(2i\pi\tau\nu + i\phi)

    Args:
        fqs (1D ndarray): frequencies [GHz]
        amp (float or ndarray): reflection amplitude
        dly (float or ndarray): reflection delay [nanosec]
        phs (float or ndarray): reflection phase [radian]
        conj (bool, optional): if True, conjugate the reflection coefficient

    Returns:
        complex ndarray: complex reflection gain

    Notes:
        If reflection terms (amp, dly, phs) are fed as a float they are assumed to be
        frequency and time independent. If they are an ndarray, they can take the following
        shapes: (1,) or (Ntimes,) or (1, Nfreqs) or (Ntimes, Nfreqs).
    """
    # type and shape check
    def _type_check(arr):
        if isinstance(arr, np.ndarray):
            if arr.ndim == 1 and arr.size > 1:
                # resize into (Ntimes, 1)
                arr = arr.reshape(-1, 1)
                # if this happens to be of len Nfreqs, raise a warning
                if arr.shape[0] == Nfreqs:
                    warnings.warn("Warning: the input array had len Nfreqs, " \
                                  "but we are reshaping it as (Ntimes, 1)")
            elif arr.ndim > 1:
                assert arr.shape[1] in [1, Nfreqs], "frequency-dependent reflection coefficients" \
                "must match input fqs size"
        return arr
    Nfreqs = fqs.size
    amp = _type_check(amp)
    dly = _type_check(dly)
    phs = _type_check(phs)

    # form reflection coefficient
    eps = amp * np.exp(2j * np.pi * fqs * dly + 1j * phs)

    # conjugate
    if conj:
        eps = eps.conj()

    return eps


def gen_reflection_gains(fqs, ants, amp=None, dly=None, phs=None, conj=False):
    r"""
    Generate a signal chain reflection as an antenna gain.

    A signal chain reflection is a copy of an antenna
    voltage stream at a boosted delay, and can be incorporated
    via a gain term

    .. math::   g_1 = (1 + \epsilon_{11})

    where :math:`\epsilon_{11}` is antenna 1's reflection coefficient
    which can be constructed as

    .. math:: \epsilon_{11} = A_{11} * \exp(2i\pi\tau_{11}\nu + i\phi_{11})

    Args:
        fqs (1D ndarray): frequencies [GHz]
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
        eps = gen_reflection_coefficient(fqs, amp[i], dly[i], phs[i], conj=conj)
        gains[ai] = (1 + eps)

    return gains


def apply_gains(vis, gains, bl):
    """
    Apply to a (NTIMES,NFREQS) visibility waterfall the bandpass functions
    for its constituent antennas.

    Args:
        vis (array-like): shape=(NTIMES,NFREQS)
            the visibility waterfall to which gains will be applied
        gains (dictionary):
            a dictionary of antenna numbers as keys and
            complex gain ndarrays as values (e.g. output of :meth:`~gen_gains`)
            with shape as either (NTIMES,NFREQS) or (NFREQS,)
        bl (2-tuple):
            a (i, j) tuple representing the baseline corresponding to
            this visibility.  g_i * g_j.conj() will be multiplied into vis.
    Returns:
        vis (array-like): shape=(NTIMES,NFREQS)
            the visibility waterfall with gains applied, unless antennas in bl
            don't exist in gains, then input vis is returned
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


def gen_whitenoise_xtalk(fqs, amplitude=3.0):
    """
    Generate a white-noise cross-talk model for specified bls.

    Args:
        fqs (ndarray): frequencies of observation [GHz]
        amplitude (float): amplitude of cross-talk in visibility units

    Returns:
        1D ndarray: xtalk model across frequencies

    See Also:
        :meth:`~apply_xtalk`: apply the output of this function to a visibility.
    """
    xtalk = np.convolve(
        noise.white_noise(fqs.size),
        np.ones(50 if fqs.size > 50 else int(fqs.size/2)),
        'same'
    )

    return xtalk * amplitude


def gen_cross_coupling_xtalk(fqs, autovis, amp=None, dly=None, phs=None, conj=False):
    r"""
    Generate a cross coupling systematic (e.g. crosstalk).

    A cross coupling systematic is the auto-correlation visibility multiplied by a
    coupling coefficient. If :math:`V_{11}` is the auto-correlation visibility of
    antenna 1, and :math:`\epsilon_{12}` is the coupling coefficient, then cross
    correlation visibility takes the form

    .. math::   V_{12} = v_1 v_2^\ast + V_{11}\epsilon_{12}^\ast

    where :math:`\epsilon_{12}` is modeled as a reflection coefficient constructed as

    .. math::   \epsilon_{12} = A_{12} * \exp(2i\pi\tau_{12}\nu + i\phi_{12})

     Args:
        fqs (1D ndarray): frequencies [GHz]
        autovis (2D ndarray): auto-correlation visibility ndarray of shape (Ntimes, Nfreqs)
        amp (float): coupling amplitude
        dly (float): coupling delay [nanosec]
        phs (float): coupling phase [radian]
        conj (bool, optional): if True, conjugate the coupling coefficient

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
    eps = gen_reflection_coefficient(fqs, amp, dly, phs, conj=conj)
    if eps.ndim == 1:
        eps = np.reshape(eps, (1, -1))

    return autovis * eps


def apply_xtalk(vis, xtalk):
    """
    Apply to a (NTIMES,NFREQS) visibility waterfall a crosstalk signal

    Args:
        vis (array-like): shape=(NTIMES,NFREQS)
            the visibility waterfall to which gains will be applied
        xtalk (array-like): shape=(NTIMES,NFREQS) or (NFREQS,)
            the crosstalk signal to be applied.

    Returns:
        vis (array-like): shape=(NTIMES,NFREQS)
            the visibility waterfall with crosstalk injected
    """
    # if xtalk is a single spectrum, reshape it into 2D array
    if xtalk.ndim == 1:
        xtalk = np.reshape(xtalk, (1, -1))

    return vis + xtalk

