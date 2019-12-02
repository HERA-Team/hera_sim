"""EoR from an object-oriented approach."""

from .components import registry 
from abc import abstractmethod
from cached_property import cached_property
from . import utils

@registry
class EoR:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class NoiselikeEoR(EoR):
    def __init__(self, eor_amp=1e-5, 
                 min_delay=None, max_delay=None, 
                 fringe_filter_type="tophat",
                 fringe_filter_kwargs={}, **kwargs):
        super().__init__(**kwargs)
    # TODO: docstrings

    def __call__(self, lsts, freqs, bl_vec, **kwargs):
        # make white noise in freq/time
        # XXX: original says in frate/freq, not sure why
        data = utils.gen_white_noise(size=(len(lsts), len(freqs)))

        # scale data by EoR amplitude
        data *= eor_amp

        # apply delay filter; default does nothing
        # XXX find out why bl_len_ns is hardcoded as 1e10
        # XXX also find out why a tophat filter is hardcoded
        data = utils.rough_delay_filter(data, freqs, 1e10, 
                                        filter_type="tophat",
                                        min_delay=min_delay,
                                        max_delay=max_delay)

        # apply fringe-rate filter
        data = utils.rough_fringe_filter(data, lsts, freqs, bl_vec[0], 
                                         filter_type=fringe_filter_type, 
                                         **fringe_filter_kwargs)

        return data


noiselike_eor = NoiselikeEoR()
