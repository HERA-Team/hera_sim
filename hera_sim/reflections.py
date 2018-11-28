"""A module for generating reflections"""

import numpy as np
import aipy

def auto_reflection(vis, freqs, amp, dly, phs, conj=False):
    """
    Insert an auto reflection into vis data. An auto reflection
    is one that couples an antenna's voltage stream with itself.
    Examples include cable reflections and dish-to-feed reflections.

    Args:
        vis : 2D complex ndarray, shape=(Ntimes, Nfreqs)
        freqs : 1D ndarray, holds frequencies in GHz
        amp : float, reflection amplitude
        dly : float, reflection delay [nanosec]
        phs : float, reflection phase [radian]
        conj : bool, if True, conjugate reflection coefficient

    Returns:
        out_vis : contains input vis with the auto reflection
    """
    # form reflection coefficient
    eps = amp * np.exp(2j * np.pi * freqs * dly + 1j * phs)
    if conj:
        eps = eps.conj()

    # multiply into data
    out_vis = vis * (1 + eps)

    return out_vis


def cross_reflection(vis, freqs, autocorr, amp, dly, phs, conj=False):
    """
    Insert a cross reflection (e.g. crosstalk) into vis data,
    which is assumed to be a cross correlation visibility.
    A cross reflection introduces an autocorrelation term into the
    crosscorrelation at a specific delay with suppressed amplitude.

    Args:
        vis : 2D complex ndarray, with shape=(Ntimes, Nfreqs)
        freqs : 1D ndarray, holding frequenices [GHz]
        autocorr : 2D float ndarray, autocorrelation waterfall
            matching vis in shape and units
        amp : float or 1D ndarray, reflection amplitude(s)
        dly : float or 1D ndarray, reflection delay(s)
        phs : float or 1D ndarray, reflection phase(s)
        conj : bool, if True, conjugate reflection coefficient

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

    # generate reflection term
    X = eps * autocorr

    # add into vis
    out_vis = vis + X

    return out_vis





