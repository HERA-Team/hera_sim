"""A module for generating realistic HERA RFI."""

import numpy as np
import aipy


class RfiStation:
    """
    An object representing an RFI station (or source).

    It is defined by attributes such `duty_cycle`, `strength` and `timescale`, and contains methods for producing
    mock RFI.
    """
    def __init__(self, fq0, duty_cycle=1.0, strength=100.0, std=10.0, timescale=100.0):
        """
        Initializer.

        Args:
            fq0 (float): reference frequency [GHz]
            duty_cycle (float): affects how much of the day is plagued by RFI.
            strength (float): mean strength of RFI [Jy]
            std (float): std deviation of RFI strength [Jy]
            timescale (float): timescale of recurring RFI [units?].
        """
        self.fq0 = fq0
        self.duty_cycle = duty_cycle
        self.std = std
        self.strength = strength
        self.timescale = timescale

    def gen_rfi(self, fqs, lsts, rfi=None):
        """
        Generate mock RFI over a given set of frequencies and times.

        Args:
            fqs (ndarray): frequencies [Ghz]
            lsts (ndarray): LSTs [radians]
            rfi (complex 2D ndarray, optional): an RFI array to which to add the generated RFI.

        Returns:
            complex 2D ndarray: RFI at each LST and frequency.
        """
        sdf = np.average(fqs[1:] - fqs[:-1])
        if rfi is None:
            rfi = np.zeros((lsts.size, fqs.size), dtype=np.complex)
        assert rfi.shape == (lsts.size, fqs.size)
        try:
            ch1 = np.argwhere(np.abs(fqs - self.fq0) < sdf)[0, 0]
        except (IndexError):
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


def rfi_stations(fqs, lsts, stations=HERA_RFI_STATIONS, rfi=None):
    """
    Generate mock RFI for a list of stations.

    Args:
        fqs (ndarray): frequencies [Ghz]
        lsts (ndarray): LSTs [radians]
        stations (list of 5-tuples): a list of tuples used to initialize :class:`RfiStation` instances.
            The tuple is of the form (fq0, duty_cycle, strength, std, timescale). Instead of a tuple, an instance
            of :class:`RfiStation` may be given.
        rfi (complex 2D ndarray, optional): an RFI array to which to add the generated RFI.

    Returns:
        complex 2D ndarray: RFI at each LST and frequency.
    """
    for s in stations:
        if not isinstance(s, RfiStation):
            if len(s) != 5:
                raise ValueError("Each station must be a 5-tuple")

            s = RfiStation(*s)
        rfi = s.gen_rfi(fqs, lsts, rfi=rfi)
    return rfi


def rfi_impulse(fqs, lsts, rfi=None, chance=0.001, strength=20.0):
    """
    Generate RFI impulses.

    Args:
        fqs (ndarray): frequencies [Ghz]
        lsts (ndarray): LSTs [radians]
        rfi (complex 2D ndarray, optional): an RFI array to which to add the generated RFI.
        chance (float): probability of RFI occuring in a given time bin.
        strength (float): mean strength of the resulting RFI [Jy]

    Returns:
        complex 2D ndarray: RFI at each LST and frequency.
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

def rfi_scatter(fqs, lsts, rfi=None, chance=0.0001, strength=10, std=10):
    """
    Generate scattered RFI over times and frequencies.

    Args:
        fqs (ndarray): frequencies [Ghz]
        lsts (ndarray): LSTs [radians]
        rfi (complex 2D ndarray, optional): an RFI array to which to add the generated RFI.
        chance (float): probability of RFI occuring in a given time/frequency bin.
        strength (float): mean strength of the resulting RFI [Jy]
        std (float): std deviation of the RFI srength.

    Returns:
        complex 2D ndarray: RFI at each LST and frequency.
    """
    if rfi is None:
        rfi = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    assert rfi.shape == (lsts.size, fqs.size)
    rfis = np.where(np.random.uniform(size=rfi.size) <= chance)[0]
    rfi.flat[rfis] += np.random.normal(strength, std) * np.exp(
        1j * np.random.uniform(size=rfis.size)
    )
    return rfi
