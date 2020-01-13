"""EoR from an object-oriented approach."""

from .components import registry 
from abc import abstractmethod
from cached_property import cached_property
from . import utils

@registry
class EoR:
    pass

class NoiselikeEoR(EoR):
    # TODO: docstring
    _alias = ("noiselike_eor",) 

    def __init__(self, eor_amp=1e-5, min_delay=None, max_delay=None, 
                 fringe_filter_type="tophat", fringe_filter_kwargs={}):
        # TODO: docstring
        """
        """
        super().__init__(
            eor_amp=eor_amp, 
            min_delay=min_delay, max_delay=max_delay, 
            fringe_filter_type=fringe_filter_type,
            fringe_filter_kwargs=fringe_filter_kwargs
        )

    def __call__(self, lsts, freqs, bl_vec, **kwargs):
        # validate the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (eor_amp, min_delay, max_delay, fringe_filter_type, 
            fringe_filter_kwargs) = self._extract_kwarg_values(**kwargs)

        # make white noise in freq/time
        # XXX: original says in frate/freq, not sure why
        data = utils.gen_white_noise(size=(len(lsts), len(freqs)))

        # scale data by EoR amplitude
        data *= eor_amp

        # apply delay filter; default does nothing
        # XXX find out why bl_len_ns is hardcoded as 1e10
        # XXX also find out why a tophat filter is hardcoded
        data = utils.rough_delay_filter(data, freqs, 1e10, 
                                        delay_filter_type="tophat",
                                        min_delay=min_delay,
                                        max_delay=max_delay)

        # apply fringe-rate filter
        data = utils.rough_fringe_filter(data, lsts, freqs, bl_vec[0], 
                                         fringe_filter_type=fringe_filter_type, 
                                         **fringe_filter_kwargs)

        # dirty trick to make autocorrelations real-valued
        if np.all(np.isclose(bl_vec, 0)):
            data = data.real + np.zeros(data.shape, dtype=np.complex)

        return data


noiselike_eor = NoiselikeEoR()
