"""v1 implementation of RFI. Consult w/ Steven for new RFI module."""

import astropy.units as u
from .components import registry

@registry
class RFI:
    pass

class RfiStation:
    # TODO: docstring
    """
    """
    def __init__(self, f0, duty_cycle=1.0, strength=100.0, 
                 std=10.0, timescale=100.0):
        self.f0 = f0
        self.duty_cycle = duty_cycle
        self.strength = strength
        self.std = std
        self.timescale = timescale

    def __call__(self, lsts, freqs):
        # TODO: docstring
        """
        """
        # initialize an array for storing the rfi
        rfi = np.zeros((lsts.size, freqs.size), dtype=np.complex)
        
        # get the mean channel width
        channel_width = np.mean(np.diff(freqs))

        # find out if the station is in the observing band
        try:
            ch1 = np.argwhere(np.abs(freqs - self.f0) < channel_width)[0,0]
        except IndexError:
            # station is not observed
            return rfi

        # find out whether to use the channel above or below... why?
        # I would think that the only time we care about neighboring
        # channels is when the station bandwidth causes the signal to 
        # spill over into neighboring channels
        ch2 = ch1 + 1 if self.f0 > freqs[ch1] else ch1 - 1

        # generate some random phases
        phs1, phs2 = np.random.uniform(0, 2 * np.pi, size=2)
        
        # find out when the station is broadcasting
        is_on = 0.999 * np.cos(lsts * u.sday.to("s") / self.timescale + phs1)
        is_on = np.where(is_on > 1 - 2 * self.duty_cycle, True, False)

        # generate a signal and filter it according to when it's on
        signal = np.random.normal(self.strength, self.std, lsts.size)
        signal = np.where(is_on, signal, 0) * np.exp(1j * phs2)

        # now add the signal to the rfi array
        for ch in (ch1, ch2):
            # note: this assumes that the signal is completely contained
            # within the two channels ch1 and ch2; for very fine freq 
            # resolution, this will usually not be the case
            df = np.abs(freqs[ch] - self.f0)
            taper = (1 - df / channel_width).clip(0, 1)
            rfi[:, ch] += signal * taper

        return rfi

class RfiStations(RFI):
    # TODO: docstring
    """
    """
    def __init__(self, stations=None):
        # TODO: docstring
        """
        """
        super().__init__(stations=stations)
