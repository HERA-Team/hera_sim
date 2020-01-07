"""v1 implementation of RFI. Consult w/ Steven for new RFI module."""
import warnings
import numpy as np
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

class Stations(RFI):
    # TODO: docstring
    """
    """
    _alias = ("rfi_stations", )

    def __init__(self, stations=None):
        # TODO: docstring
        """
        """
        super().__init__(stations=stations)

    def __call__(self, lsts, freqs, **kwargs):
        # TODO: docstring
        """
        """
        # kind of silly to use **kwargs with just one optional parameter...
        self._check_kwargs(**kwargs)

        # but this is where the magic comes in (thanks to defaults)
        stations, = self._extract_kwarg_values(**kwargs)

        # initialize an array to store the rfi in
        rfi = np.zeros((lsts.size, freqs.size), dtype=np.complex)

        if stations is None:
            warnings.warn(
                "You did not specify any stations to simulate."
            )
            return rfi
        elif isinstance(stations, str):
            # assume that it's a path to a npy file
            stations = np.load(stations)

        for station in stations:
            if not isinstance(station, RfiStation):
                if len(station) != 5:
                    raise ValueError(
                        "Stations are specified by 5-tuples. Please "
                        "check the format of your stations."
                    )
                
                # make an RfiStation if it isn't one
                station = RfiStation(*station)

            # add the effect
            rfi += station(lsts, freqs)

        return rfi

class Impulse(RFI):
    # TODO: docstring
    """
    """
    _alias = ("rfi_impulse", )

    def __init__(self, impulse_chance=0.001, impulse_strength=20.0):
        # TODO: docstring
        """
        """
        super().__init__(
            impulse_chance=impulse_chance,
            impulse_strength=impulse_strength
        )

    def __call__(self, lsts, freqs, **kwargs):
        # TODO: docstring
        """
        """
        # check that the kwargs are okay
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        chance, strength = self._extract_kwarg_values(**kwargs)

        # initialize the rfi array
        rfi = np.zeros((lsts.size, freqs.size), dtype=np.complex)

        # find times when an impulse occurs
        impulses = np.where(np.random.uniform(size=lsts.size) <= chance)[0]

        # only do something if there are impulses
        if impulses.size > 0:
            # randomly generate some delays for each impulse
            dlys = np.random.uniform(-300, 300, impulses.size) # ns

            # generate the signals
            signals = strength * np.asarray(
                [np.exp(2j * np.pi * dly * freqs) for dly in dlys]
            )

            rfi[impulses] += signals

        return rfi

class Scatter(RFI):
    # TODO: docstring
    """
    """
    _alias = ("rfi_scatter", )

    def __init__(self, scatter_chance=0.0001, scatter_strength=10.0,
                 scatter_std=10.0):
        # TODO: docstring
        """
        """
        super().__init__(
            scatter_chance=scatter_chance,
            scatter_strength=scatter_strength,
            scatter_std=scatter_std
        )

    def __call__(self, lsts, freqs, **kwargs):
        # TODO: docstring
        """
        """
        # validate the kwargs
        self._check_kwargs(**kwargs)

        # now unpack them
        chance, strength, std = self._extract_kwarg_values(**kwargs)

        # make an empty rfi array
        rfi = np.zeros((lsts.size, freqs.size), dtype=np.complex)

        # find out where to put the rfi
        rfis = np.where(np.random.uniform(size=rfi.size) <= chance)[0]

        # simulate the rfi; one random amplitude, all random phases
        signal = np.random.normal(strength, std) * np.exp(
            2j * np.pi * np.random.uniform(size=rfis.size)
        )

        # add the signal to the rfi
        rfi.flat[rfis] += signal

        return rfi

class DTV(RFI):
    # TODO: docstring
    """
    """
    _alias = ("rfi_dtv", )

    def __init__(self, dtv_band=(0.174, 0.214), 
                 dtv_channel_width=0.008, dtv_chance=0.0001, 
                 dtv_strength=10.0, dtv_std=10.0):
        # TODO: docstring
        """
        """
        super().__init__(
            dtv_band=dtv_band,
            dtv_channel_width=dtv_channel_width,
            dtv_chance=dtv_chance,
            dtv_strength=dtv_strength,
            dtv_std=dtv_std
        )

    def __call__(self, lsts, freqs, **kwargs):
        # TODO: docstring
        """
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # unpack them
        (dtv_band, width, chance, strength, 
            std) = self._extract_kwarg_values(**kwargs)

        # make an empty rfi array
        rfi = np.zeros((lsts.size, freqs.size), dtype=np.complex)

        # get the lower and upper frequencies of the DTV band
        freq_min, freq_max = dtv_band

        # get the lower frequencies of each subband
        bands = np.arange(freq_min, freq_max, width)

        # only keep bands which overlap with the passed frequencies
        overlap = np.logical_and(
            bands >= freqs.min() - width, bands <= freqs.max()
        )
        bands = bands[overlap]

        # raise a warning if there are no remaining bands
        if len(bands) == 0:
            warnings.warn(
                "The DTV band does not overlap with any of the passed "
                "frequencies. Please ensure that you are passing the "
                "correct set of parameters."
            )

        # get the instrument channel width
        channel_width = np.mean(np.diff(freqs))

        # loop over the DTV bands, generating rfi where appropriate
        for band in bands:
            # get the channels affected
            ch1 = np.argwhere(band <= freqs)[0][0]
            try:
                ch2 = np.argwhere(band + width <= freqs)[0][0]
            except IndexError:
                # in case the upper edge of the DTV band is outside
                # the range of observed frequencies
                ch2 = freqs.size

            # pick out just the channels affected
            this_rfi = rfi[:, ch1:ch2]

            # find out which times are affected
            rfis = np.random.uniform(size=lsts.size) <= chance

            # calculate the signal... whomever did this first, tsk tsk...
            signal = np.atleast_2d(
                np.random.normal(strength, std, size=rfis.sum())
                * np.exp(2j * np.pi * np.random.uniform(size=rfis.sum()))
            ).T # don't ask me why we're taking the transpose...

            # add the signal to the rfi array
            this_rfi[rfis] += signal

        return rfi

rfi_stations = Stations()
rfi_impulse = Impulse()
rfi_scatter = Scatter()
rfi_dtv = DTV()
