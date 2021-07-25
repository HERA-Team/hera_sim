"""A module for simulating EoR-like visibilities.

EoR models should require lsts, frequencies, and a baseline vector as
arguments, and may have arbitrary optional parameters. Models should
return complex-valued arrays with shape (Nlsts, Nfreqs) that represent
a visibility appropriate for the given baseline.
"""

import numpy as np
from .components import component
from . import utils
from typing import Optional


@component
class EoR:
    """Base class for fast EoR simualtors."""

    pass


class NoiselikeEoR(EoR):
    """Generate a noiselike, fringe-filtered EoR visibility.

    Parameters
    ----------
    eor_amp
        The amplitude of the EoR power spectrum.
    min_delay
        Minimum delay to allow through the delay filter.
        Default is -inf.
    max_delay
        Maximum delay to allow through the delay filter.
        Default is +inf
    fringe_filter_type
        The kind of filter to apply in fringe-space.
    fringe_filter_kwargs
        Arguments to pass to the fringe filter. See :func:`utils.rough_fringe_filter`
        for possible arguments.

    Notes
    -----
    This algorithm produces visibilities as a function of time/frequency
    that have white noise structure, filtered over the delay and fringe-rate
    axes. The fringe-rate filter makes the data look more like EoR by constraining
    it to moving with the sky (given the baseline vector).
    """

    _alias = ("noiselike_eor",)
    is_smooth_in_freq = False

    def __init__(
        self,
        eor_amp: float = 1e-5,
        min_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        fringe_filter_type: str = "tophat",
        fringe_filter_kwargs: Optional[dict] = None,
    ):
        fringe_filter_kwargs = fringe_filter_kwargs or {}

        super().__init__(
            eor_amp=eor_amp,
            min_delay=min_delay,
            max_delay=max_delay,
            fringe_filter_type=fringe_filter_type,
            fringe_filter_kwargs=fringe_filter_kwargs,
        )

    def __call__(self, lsts, freqs, bl_vec, **kwargs):
        """Compute the noise-like EoR model."""
        # validate the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (
            eor_amp,
            min_delay,
            max_delay,
            fringe_filter_type,
            fringe_filter_kwargs,
        ) = self._extract_kwarg_values(**kwargs)

        # make white noise in freq/time (original says in frate/freq, not sure why)
        data = utils.gen_white_noise(size=(len(lsts), len(freqs)))

        # scale data by EoR amplitude
        data *= eor_amp

        # apply delay filter; default does nothing
        # TODO: find out why bl_len_ns is hardcoded as 1e10, also
        # why a tophat filter is hardcoded; isn't this the same as just
        # using no filter but setting min/max delay?
        data = utils.rough_delay_filter(
            data,
            freqs,
            1e10,
            delay_filter_type="tophat",
            min_delay=min_delay,
            max_delay=max_delay,
        )

        # apply fringe-rate filter
        data = utils.rough_fringe_filter(
            data,
            lsts,
            freqs,
            bl_vec[0],
            fringe_filter_type=fringe_filter_type,
            **fringe_filter_kwargs
        )

        # dirty trick to make autocorrelations real-valued
        # TODO: Figure out the statistically correct way to handle autos.
        # Handling autos this way makes the covariance look like it has
        # no structure... which is wrong.
        if np.all(np.isclose(bl_vec, 0)):
            data = data.real.astype(complex)

        return data


noiselike_eor = NoiselikeEoR()
