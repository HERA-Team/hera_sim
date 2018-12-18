"""A module for generating reflections"""

import numpy as np
import aipy

def auto_reflection(vis, freqs, amp, dly, phs, conj=False, Norder=1):
    """
    Insert an auto reflection into vis data. An auto reflection
    is one that couples an antenna's voltage stream with itself.
    Examples include cable reflections and dish-to-feed reflections.
    
    If v_1 is antenna 1's voltage spectrum, and V_12 is the
    cross correlation visibility between antenna 1 and 2,
    then an auto reflection in the visibility can be written as

        V_12 = v_1 (1 + eps_1) v_2^*

    where eps_1 is antenna 1's reflection coefficient which
    can be constructed as

        eps = amp * exp(2j pi dly nu + 1j phs)

    If Norder is set to larger than 1, the reflection term
    becomes

        (1 + eps_1 + eps_1^2 + eps_1^3 + ...)
    
    where the default Norder = 1 is just (1 + eps_1).

    Args:
        vis : 2D complex ndarray, shape=(Ntimes, Nfreqs)
        freqs : 1D ndarray, holds frequencies in GHz
        amp : float, reflection amplitude
        dly : float, reflection delay [nanosec]
        phs : float, reflection phase [radian]
        conj : bool, if True, conjugate reflection coefficient
        Norder : int, number of harmonics to simulate: must be >= 1.

    Returns:
        out_vis : contains input vis with the auto reflection
    """
    # form reflection coefficient
    eps = amp * np.exp(2j * np.pi * freqs * dly + 1j * phs)
    if conj:
        eps = eps.conj()

    # get higher order terms
    if Norder < 1:
        raise ValueError("Norder must be >= 1")
    reflection = 1
    for i in np.arange(1., Norder + 1):
        reflection += eps ** i

    # multiply into data
    out_vis = vis * reflection

    return out_vis


def cross_reflection(vis, freqs, autocorr, amp, dly, phs, conj=False, Norder=1):
    """
    Insert a cross reflection (e.g. crosstalk) into vis data,
    which is assumed to be a cross correlation visibility.
    A cross reflection introduces an autocorrelation term into the
    crosscorrelation at a specific delay with suppressed amplitude.

    If v_1 is antenna 1's voltage spectrum, and V_12 is the 
    cross-correlation visibility between antenna 1 and 2,
    then a cross reflection in the visibility can be written as

        V_12 = v_1 (v_2 + eps_12 v_1)^*
             = v_1 v_2^* + v_1 v_1^* eps_12^*

    where eps_12 is the coupling of antenna 1's voltage into
    antenna 2's signal chain, and is a standard reflection
    coefficient as described in auto_reflection().

    Args:
        vis : 2D complex ndarray, with shape=(Ntimes, Nfreqs)
        freqs : 1D ndarray, holding frequenices [GHz]
        autocorr : 2D float ndarray, autocorrelation waterfall
            matching vis in shape and units
        amp : float or 1D ndarray, reflection amplitude(s)
        dly : float or 1D ndarray, reflection delay(s)
        phs : float or 1D ndarray, reflection phase(s)
        conj : bool, if True, conjugate reflection coefficient
        Norder : int, number of harmonics to simulate: must be >= 1.

    Returns:
        out_vis : 2D complex ndarray, holding input vis data
            with cross reflections added in.
    """
    # generate empty reflection coefficient across freq
    eps = np.zeros(len(freqs), dtype=np.complex)

    if isinstance(amp, (float, np.float, np.int, int)):
        amp = [amp]
    if isinstance(dly, (float, np.float, np.int, int)):
        dly = [dly]
    if isinstance(phs, (float, np.float, np.int, int)):
        phs = [phs]

    # iterate over reflection parameters
    for a, d, p in zip(amp, dly, phs):
        eps += a * np.exp(2j * np.pi * freqs * d + 1j * p)
    if conj:
        eps = eps.conj()

    # get higher order reflection terms
    if Norder < 1:
        raise ValueError("Norder must be >= 1")
    reflection = 0
    for i in np.arange(1., Norder + 1):
        reflection += eps ** i

    # generate xtalk term
    X = autocorr * reflection
    
    # add into vis
    out_vis = vis + X

    return out_vis





