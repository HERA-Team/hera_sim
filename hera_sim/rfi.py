"""A module for generating realistic HERA RFI."""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

from future import standard_library
standard_library.install_aliases()
# from builtins import *
from builtins import zip
from astropy.units import sday
import numpy as np
from os import path
from hera_sim.data import DATA_PATH
from hera_sim.interpolators import _read_npy
from .defaults import _defaults
import warnings

# XXX the below is repeated code. figure out which module to store the general
# XXX code in

@_defaults
def _get_hera_stations(rfi_stations="HERA_H1C_RFI_STATIONS.npy"):
    """
    Accept a .npy file and return an array of HERA RFI station parameters.
    """
    return _read_npy(rfi_stations)

# this will just return the RFI Station parameters relevant for the H1C
# observing season.
HERA_RFI_STATIONS = _get_hera_stations()

class RfiStation:
    """
    Class for representing an RFI transmitter.

    Args:
        fq0 (float): GHz
            center frequency of the RFI transmitter
        duty_cycle (float): default=1.
            fraction of times that RFI transmitter is on
        strength (float): Jy, default=100
            amplitude of RFI transmitter
        std (float): default=10.
            standard deviation of transmission amplitude
        timescale (scalar): seconds, default=100.
            timescale for sinusoidal variation in transmission amplitude'''
    """

    def __init__(self, fq0, duty_cycle=1.0, strength=100.0, std=10.0, timescale=100.0):
        self.fq0 = fq0
        self.duty_cycle = duty_cycle
        self.std = std
        self.strength = strength
        self.timescale = timescale

    def gen_rfi(self, fqs, lsts, rfi=None):
        """
        Generate an (NTIMES,NFREQS) waterfall containing RFI.

        Args:
            lsts (array-like): shape=(NTIMES,), radians
                local sidereal times of the waterfall to be generated.
            fqs (array-like): shape=(NFREQS,), GHz
                the spectral frequencies of the waterfall to be generated.
            rfi (array-like): shape=(NTIMES,NFREQS), default=None
                an array to which the RFI will be added.  If None, a new array
                is generated.
        Returns:
            rfi (array-like): shape=(NTIMES,NFREQS)
                a waterfall containing RFI
        """
        sdf = np.average(fqs[1:] - fqs[:-1])
        if rfi is None:
            rfi = np.zeros((lsts.size, fqs.size), dtype=np.complex)
        assert rfi.shape == (lsts.size, fqs.size), "rfi is not shape (lsts.size, fqs.size)"

        try:
            ch1 = np.argwhere(np.abs(fqs - self.fq0) < sdf)[0, 0]
        except IndexError:
            return rfi
        if self.fq0 > fqs[ch1]:
            ch2 = ch1 + 1
        else:
            ch2 = ch1 - 1
        phs1, phs2 = np.random.uniform(0, 2 * np.pi, size=2)
        signal = 0.999 * np.cos(
            lsts * sday.to("s") /self.timescale + phs1
        ) + 2 * (self.duty_cycle - 0.5)
        signal = np.where(
            signal > 0, np.random.normal(self.strength, self.std) * np.exp(1j * phs2), 0
        )
        rfi[:, ch1] += signal * (1 - np.abs(fqs[ch1] - self.fq0) / sdf).clip(0, 1)
        rfi[:, ch2] += signal * (1 - np.abs(fqs[ch2] - self.fq0) / sdf).clip(0, 1)
        return rfi


# XXX reverse lsts and fqs?
def rfi_stations(fqs, lsts, stations=None, rfi=None):
    """
    Generate an (NTIMES,NFREQS) waterfall containing RFI stations that
    are localized in frequency.

    Args:
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the waterfall to be generated.
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the waterfall to be generated.
        stations (iterable): list of 5-tuples, default=HERA_RFI_STATIONS
            a list of (FREQ, DUTY_CYCLE, STRENGTH, STD, TIMESCALE) tuples
            for RfiStations that will be injected into waterfall. Instead
            of a tuple, an instance of :class:`RfiStation` may be given.
        rfi (array-like): shape=(NTIMES,NFREQS), default=None
            an array to which the RFI will be added.  If None, a new array
            is generated.
    Returns:
        rfi (array-like): shape=(NTIMES,NFREQS)
            a waterfall containing RFI
    """
    if stations is None:
        stations = _get_hera_stations()
    elif isinstance(stations, str):
        stations = _get_hera_stations(stations)
    for s in stations:
        if not isinstance(s, RfiStation):
            if len(s) != 5:
                raise ValueError("Each station must be a 5-tuple")

            s = RfiStation(*s)
        rfi = s.gen_rfi(fqs, lsts, rfi=rfi)
    return rfi


# XXX reverse lsts and fqs?
def rfi_impulse(fqs, lsts, rfi=None, chance=0.001, strength=20.0):
    """
    Generate an (NTIMES,NFREQS) waterfall containing RFI impulses that
    are localized in time but span the frequency band.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the waterfall to be generated.
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the waterfall to be generated.
        rfi (array-like): shape=(NTIMES,NFREQS), default=None
            an array to which the RFI will be added.  If None, a new array
            is generated.
        chance (float):
            the probability that a time bin will be assigned an RFI impulse
        strength (float): Jy
            the strength of the impulse generated in each time/freq bin
    Returns:
        rfi (array-like): shape=(NTIMES,NFREQS)
            a waterfall containing RFI'''
    """
    if rfi is None:
        rfi = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    assert rfi.shape == (lsts.size, fqs.size), "rfi is not shape (lsts.size, fqs.size)"

    impulse_times = np.where(np.random.uniform(size=lsts.size) <= chance)[0]
    dlys = np.random.uniform(-300, 300, size=impulse_times.size)  # ns
    impulses = strength * np.array([np.exp(2j * np.pi * dly * fqs) for dly in dlys])
    if impulses.size > 0:
        rfi[impulse_times] += impulses
    return rfi


# XXX reverse lsts and fqs?
def rfi_scatter(fqs, lsts, rfi=None, chance=0.0001, strength=10, std=10):
    """
    Generate an (NTIMES,NFREQS) waterfall containing RFI impulses that
    are localized in time but span the frequency band.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the waterfall to be generated.
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the waterfall to be generated.
        rfi (array-like): shape=(NTIMES,NFREQS), default=None
            an array to which the RFI will be added.  If None, a new array
            is generated.
        chance (float): default=0.0001
            the probability that a time/freq bin will be assigned an RFI impulse
        strength (float): Jy, default=10
            the average amplitude of the spike generated in each time/freq bin
        std (float): Jy, default = 10
            the standard deviation of the amplitudes drawn for each time/freq bin
    Returns:
        rfi (array-like): shape=(NTIMES,NFREQS)
            a waterfall containing RFI
    """
    if rfi is None:
        rfi = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    assert rfi.shape == (lsts.size, fqs.size), "rfi shape is not (lsts.size, fqs.size)"

    rfis = np.where(np.random.uniform(size=rfi.size) <= chance)[0]
    rfi.flat[rfis] += np.random.normal(strength, std) * np.exp(
        2 * np.pi * 1j * np.random.uniform(size=rfis.size)
    )
    return rfi


def rfi_dtv(fqs, lsts, rfi=None, freq_min=.174, freq_max=.214, width=0.008,
            chance=0.0001, strength=10, strength_std=10):
    """
    Generate an (NTIMES, NFREQS) waterfall containing Digital TV RFI.

    DTV RFI is expected to be of uniform bandwidth (eg. 8MHz), in contiguous
    bands, in a nominal frequency range. Furthermore, it is expected to be
    short-duration, and so is implemented as randomly affecting discrete LSTS.

    There may be evidence that close times are correlated in having DTV RFI,
    and this is *not currently implemented*.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the waterfall to be generated.
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the waterfall to be generated.
        rfi (array-like): shape=(NTIMES,NFREQS), default=None
            an array to which the RFI will be added.  If None, a new array
            is generated.
        freq_min, freq_max (float):
            the min and max frequencies of the full DTV band [GHz]
        width (float):
            Width of individual DTV bands [GHz]
        chance (float): default=0.0001
            the probability that a time/freq bin will be assigned an RFI impulse
        strength (float): Jy, default=10
            the average amplitude of the spike generated in each time/freq bin
        strength_std (float): Jy, default = 10
            the standard deviation of the amplitudes drawn for each time/freq bin
    Returns:
        rfi (array-like): shape=(NTIMES,NFREQS)
            a waterfall containing RFI
    """
    if rfi is None:
        rfi = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    assert rfi.shape == (lsts.size, fqs.size), "rfi shape is not (lst.size, fqs.size)"

    bands = np.arange(freq_min, freq_max, width)  # lower freq of each potential DTV band

    # ensure that only the DTV bands which overlap with the passed frequencies are kept
    bands = bands[np.logical_and(bands >= fqs.min() - width, bands <= fqs.max())]

    if len(bands) is 0:
        warnings.warn("You are attempting to add DTV RFI to a visibility array whose " \
                      "frequencies do not overlap with any DTV band. Please ensure " \
                      "that you are using the correct frequencies.")

    delta_f = fqs[1] - fqs[0]

    chance = _listify(chance)
    strength_std = _listify(strength_std)
    strength = _listify(strength)

    if len(chance) == 1:
        chance *= len(bands)
    if len(strength) == 1:
        strength *= len(bands)
    if len(strength_std) == 1:
        strength_std *= len(bands)

    if len(chance) != len(bands):
        raise ValueError("chance must be float or list with len equal to number of bands")
    if len(strength) != len(bands):
        raise ValueError("strength must be float or list with len equal to number of bands")
    if len(strength_std) != len(bands):
        raise ValueError("strength_std must be float or list with len equal to number of bands")

    for band, chnc, strngth, str_std in zip(bands, chance, strength, strength_std):
        fq_ind_min = np.argwhere(band <= fqs)[0][0]
        try:
            fq_ind_max = np.argwhere(band + width <= fqs)[0][0]
        except IndexError:
            fq_ind_max = fqs.size
        this_rfi = rfi[:, fq_ind_min:min(fq_ind_max, fqs.size)]

        rfis = np.random.uniform(size=lsts.size) <= chnc
        this_rfi[rfis] += np.atleast_2d(np.random.normal(strngth, str_std, size=np.sum(rfis)) 
                          * np.exp(2 * np.pi * 1j * np.random.uniform(size=np.sum(rfis)))).T

    return rfi


def _listify(x):
    """
    Ensure a scalar/list is returned as a list.

    Gotten from https://stackoverflow.com/a/1416677/1467820
    """
    try:
        basestring
    except NameError:
        basestring = (str, bytes)

    if isinstance(x, basestring):
        return [x]
    else:
        try:
            iter(x)
        except TypeError:
            return [x]
        else:
            return list(x)

