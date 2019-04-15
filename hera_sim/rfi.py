"""A module for generating realistic HERA RFI."""

import numpy as np
import aipy


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
        assert rfi.shape == (lsts.size, fqs.size)
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
            lsts * aipy.const.sidereal_day / self.timescale + phs1
        ) + 2 * (self.duty_cycle - 0.5)
        signal = np.where(
            signal > 0, np.random.normal(self.strength, self.std) * np.exp(1j * phs2), 0
        )
        rfi[:, ch1] += signal * (1 - np.abs(fqs[ch1] - self.fq0) / sdf).clip(0, 1)
        rfi[:, ch2] += signal * (1 - np.abs(fqs[ch2] - self.fq0) / sdf).clip(0, 1)
        return rfi


HERA_RFI_STATIONS = [
    # FM Stations
    (0.1007, 1.0, 1000 * 100.0, 10.0, 100.0),
    (0.1016, 1.0, 1000 * 100.0, 10.0, 100.0),
    (0.1024, 1.0, 1000 * 100.0, 10.0, 100.0),
    (0.1028, 1.0, 1000 * 3.0, 1.0, 100.0),
    (0.1043, 1.0, 1000 * 100.0, 10.0, 100.0),
    (0.1050, 1.0, 1000 * 10.0, 3.0, 100.0),
    (0.1052, 1.0, 1000 * 100.0, 10.0, 100.0),
    (0.1061, 1.0, 1000 * 100.0, 10.0, 100.0),
    (0.1064, 1.0, 1000 * 10.0, 3.0, 100.0),

    # Orbcomm
    (0.1371, 0.2, 1000 * 100.0, 3.0, 600.0),
    (0.1372, 0.2, 1000 * 100.0, 3.0, 600.0),
    (0.1373, 0.2, 1000 * 100.0, 3.0, 600.0),
    (0.1374, 0.2, 1000 * 100.0, 3.0, 600.0),
    (0.1375, 0.2, 1000 * 100.0, 3.0, 600.0),
    # Other
    (0.1831, 1.0, 1000 * 100.0, 30.0, 1000),
    (0.1891, 1.0, 1000 * 2.0, 1.0, 1000),
    (0.1911, 1.0, 1000 * 100.0, 30.0, 1000),
    (0.1972, 1.0, 1000 * 100.0, 30.0, 1000),
    # DC Offset from ADCs
    # (.2000, 1., 100., 0., 10000),
]

# XXX reverse lsts and fqs?
def rfi_stations(fqs, lsts, stations=HERA_RFI_STATIONS, rfi=None):
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
    assert rfi.shape == (lsts.size, fqs.size)
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
    assert rfi.shape == (lsts.size, fqs.size)
    rfis = np.where(np.random.uniform(size=rfi.size) <= chance)[0]
    rfi.flat[rfis] += np.random.normal(strength, std) * np.exp(
        1j * np.random.uniform(size=rfis.size)
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

    bands = np.arange(freq_min, freq_max, width) # lower freq of each potential DTV band

    delta_f = fqs[1] - fqs[0]

    for band in bands:
        fq_ind_min = np.argwhere(band >= fqs)[0][0]
        fq_ind_max = fq_ind_min + int(width/delta_f)
        this_rfi = rfi[:, fq_ind_min:fq_ind_max]

        rfis = np.where(np.random.uniform(size=this_rfi) <= chance)[0]
        this_rfi.flat[rfis] += np.random.normal(strength, strength_std, size=rfis.size)*np.exp(
            2 * np.pi * 1j*np.random.uniform(size=rfis.size)
        )

    return rfi
