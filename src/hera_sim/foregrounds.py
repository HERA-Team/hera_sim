"""Visibility-space foreground models.

This module defines several cheap foreground models evaluated in visibility space.
"""

import numpy as np
from astropy import units

from . import utils
from .components import component


@component
class Foreground:
    """Base class for foreground models."""

    pass


class DiffuseForeground(Foreground):
    """
    Produce a rough simulation of diffuse foreground-like structure.

    Parameters
    ----------
    Tsky_mdl : interpolation object
        Sky temperature model, in units of Kelvin. Must be callable
        with signature Tsky_mdl(lsts, freqs), formatted so that lsts
        are in radians and freqs are in GHz.
    omega_p : interpolation object or array-like of float
        Beam size model, in units of steradian. If passing an array,
        then it must be the same shape as the frequency array passed
        to the ``freqs`` parameter.
    delay_filter_kwargs : dict, optional
        Keyword arguments and associated values to be passed to
        :func:`~hera_sim.utils.rough_delay_filter`. Default is to use the
        following settings: ``standoff : 0.0``, ``delay_filter_type : tophat``.
    fringe_filter_kwargs : dict, optional
        Keyword arguments and associated values to be passed to
        :func:`~hera_sim.utils.rough_fringe_filter`. Default is to use the
        following settings: ``fringe_filter_type : tophat``.
    rng: np.random.Generator, optional
        Random number generator.

    Notes
    -----
    This algorithm provides a rough simulation of visibilities from
    diffuse foregrounds by using a sky temperature model. The sky
    temperature models provided in this package are appropriate for
    the HERA H1C observing season, and are only valid for frequencies
    between 100 MHz and 200 MHz; anything beyond this range is just a
    copy of the value at the nearest edge. Simulated autocorrelations
    (i.e. zero magnitude ``bl_vec``) are returned as complex arrays,
    but have zero imaginary component everywhere. For cross-correlations,
    the sky model is convolved with white noise (in delay/fringe-rate
    space), and rough delay and fringe filters are applied to the
    visibility. As a standalone component model, this is does not
    produce consistent simulated visibilities for baselines within a
    redundant group (except for autocorrelations); however, the
    :class:`~hera_sim.simulate.Simulator` class provides the functionality to ensure
    that redundant baselines see the same sky. Additionally, visibilities
    simulated with this model are not invariant under complex conjugation
    and baseline conjugation, since the delay filter applied is symmetric;
    however, the :class:`~.simulate.Simulator`  class is aware of this and ensures
    invariance under complex conjugation and baseline conjugation.
    """

    _alias = ("diffuse_foreground",)
    is_smooth_in_freq = True
    is_randomized = True
    return_type = "per_baseline"
    attrs_to_pull = dict(bl_vec=None)

    def __init__(
        self,
        Tsky_mdl=None,
        omega_p=None,
        delay_filter_kwargs=None,
        fringe_filter_kwargs=None,
        rng=None,
    ):
        if delay_filter_kwargs is None:
            delay_filter_kwargs = {
                "standoff": 0.0,
                "delay_filter_type": "tophat",
                "normalize": None,
            }
        if fringe_filter_kwargs is None:
            fringe_filter_kwargs = {"fringe_filter_type": "tophat"}

        super().__init__(
            Tsky_mdl=Tsky_mdl,
            omega_p=omega_p,
            delay_filter_kwargs=delay_filter_kwargs,
            fringe_filter_kwargs=fringe_filter_kwargs,
            rng=rng,
        )

    def __call__(self, lsts, freqs, bl_vec, **kwargs):
        """Compute the foregrounds.

        Parameters
        ----------
        lsts : array-like of float
            Array of LST values in units of radians.
        freqs : array-like of float
            Array of frequency values in units of GHz.
        bl_vec : array-like of float
            Length-3 array specifying the baseline vector in units of ns.

        Returns
        -------
        vis : ndarray of complex
            Array of visibilities at each LST and frequency appropriate
            for the given sky temperature model, beam size model, and
            baseline vector. Returned in units of Jy with shape
            (lsts.size, freqs.size).
        """
        # validate the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (Tsky_mdl, omega_p, delay_filter_kwargs, fringe_filter_kwargs, rng) = (
            self._extract_kwarg_values(**kwargs)
        )

        if Tsky_mdl is None:
            raise ValueError(
                "A sky temperature model must be specified in "
                "order to use this function."
            )

        if omega_p is None:
            raise ValueError(
                "A beam area array or interpolation object is "
                "required to use this function."
            )

        # support passing beam as an interpolator
        if callable(omega_p):
            omega_p = omega_p(freqs)

        # resample the sky temperature model
        Tsky = Tsky_mdl(lsts=lsts, freqs=freqs)  # K
        vis = np.asarray(Tsky / utils.jansky_to_kelvin(freqs, omega_p), complex)

        if np.isclose(np.linalg.norm(bl_vec), 0):
            return vis

        vis *= utils.gen_white_noise(size=vis.shape, rng=rng)

        vis = utils.rough_fringe_filter(
            vis, lsts, freqs, bl_vec[0], **fringe_filter_kwargs
        )

        vis = utils.rough_delay_filter(
            vis, freqs, np.linalg.norm(bl_vec), **delay_filter_kwargs
        )

        return vis


class PointSourceForeground(Foreground):
    """
    Produce a uniformly-random point-source sky observed with a truncated Gaussian beam.

    Parameters
    ----------
    nsrcs : int, optional
        Number of sources to place on the sky. Point sources are
        simulated to have a flux-density drawn from a power-law
        distribution specified by the ``Smin``, ``Smax``, and
        ``beta`` parameters. Additionally, each source has a chromatic
        flux-density given by a power law; the spectral index is drawn
        from a normal distribution with mean ``spectral_index_mean`` and
        standard deviation ``spectral_index_std``.
    Smin : float, optional
        Lower bound of the power-law distribution to draw flux-densities
        from, in units of Jy.
    Smax : float, optional
        Upper bound of the power-law distribution to draw flux-densities
        from, in units of Jy.
    beta : float, optional
        Power law index for the source counts versus flux-density.
    spectral_index_mean : float, optional
        The mean of the normal distribution to draw source spectral indices
        from.
    spectral_index_std : float, optional
        The standard deviation of the normal distribution to draw source
        spectral indices from.
    reference_freq : float, optional
        Reference frequency used to make the point source flux densities
        chromatic, in units of GHz.
    rng: np.random.Generator, optional
        Random number generator.
    """

    _alias = ("pntsrc_foreground",)
    is_randomized = True
    return_type = "per_baseline"
    attrs_to_pull = dict(bl_vec=None)

    def __init__(
        self,
        nsrcs=1000,
        Smin=0.3,
        Smax=300,
        beta=-1.5,
        spectral_index_mean=-1,
        spectral_index_std=0.5,
        reference_freq=0.15,
        rng=None,
    ):
        super().__init__(
            nsrcs=nsrcs,
            Smin=Smin,
            Smax=Smax,
            beta=beta,
            spectral_index_mean=spectral_index_mean,
            spectral_index_std=spectral_index_std,
            reference_freq=reference_freq,
            rng=rng,
        )

    def __call__(self, lsts, freqs, bl_vec, **kwargs):
        """Compute the point source foregrounds.

        Parameters
        ----------
        lsts : array-like of float
            Local Sidereal Times for the simulated observation, in units
            of radians.
        freqs : array-like of float
            Frequency array for the simulated observation, in units of GHz.
        bl_vec : array-like of float
            Baseline vector for the simulated observation, given in
            East-North-Up coordinates in units of nanoseconds. Must have
            length 3.

        Returns
        -------
        vis : np.ndarray of complex
            Simulated observed visibilities for the specified LSTs, frequencies,
            and baseline. Complex-valued with shape (lsts.size, freqs.size).

        Notes
        -----
        The beam used here is a Gaussian with width hard-coded to HERA's width,
        and truncated at the horizon.

        This is a *very* rough simulator, use at your own risk.
        """
        # validate the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (nsrcs, Smin, Smax, beta, spectral_index_mean, spectral_index_std, f0, rng) = (
            self._extract_kwarg_values(**kwargs)
        )

        # get baseline length (it should already be in ns)
        bl_len_ns = np.linalg.norm(bl_vec)

        # Randomly generate source positions and spectral indices.
        rng = rng or np.random.default_rng()
        ras = rng.uniform(0, 2 * np.pi, nsrcs)
        spec_indices = rng.normal(
            loc=spectral_index_mean, scale=spectral_index_std, size=nsrcs
        )

        # calculate beam width, hardcoded for HERA
        beam_width = (40 * 60) * (f0 / freqs) / units.sday.to("s") * 2 * np.pi

        # draw flux densities from a power law
        alpha = beta + 1
        flux_densities = (
            Smax**alpha + Smin**alpha * (1 - rng.uniform(size=nsrcs))
        ) ** (1 / alpha)

        # initialize the visibility array
        vis = np.zeros((lsts.size, freqs.size), dtype=complex)

        # Compute the visibility source-by-source.
        for ra, flux, index in zip(ras, flux_densities, spec_indices):
            # Figure out when the source crosses the meridian.
            lst_ind = np.argmin(np.abs(utils.compute_ha(lsts, ra)))

            # This is effectively baking in that up to 10% of the baseline
            # length is oriented along the North-South direction, since this
            # is the delay measured when the source transits the meridian.
            # (Still need to think more carefully about this, but this seems
            # like the right explanation for this, as well as the factor of
            # 0.9 in the calculation of w further down.)
            dtau = rng.uniform(-1, 1) * 0.1 * bl_len_ns

            # Add the contribution from the source as it transits the meridian.
            vis[lst_ind, :] += flux * (freqs / f0) ** index
            vis[lst_ind, :] *= np.exp(2j * np.pi * freqs * dtau)

        # Figure out the hour angles to use for computing the beam kernel.
        has = utils.compute_ha(lsts, 0)

        # convolve vis with beam at each frequency
        for j, freq in enumerate(freqs):
            # Treat the beam as if it's a Gaussian with a sharp horizon.
            beam = np.exp(-(has**2) / (2 * beam_width[j] ** 2))
            beam = np.where(np.abs(has) > np.pi / 2, 0, beam)

            # Compute the phase evolution as the source transits the sky.
            w = 0.9 * bl_len_ns * np.sin(has) * freq
            phase = np.exp(2j * np.pi * w)

            # Now actually apply the mock source transit.
            kernel = beam * phase
            vis[:, j] = np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(vis[:, j]))

        return vis


diffuse_foreground = DiffuseForeground()
pntsrc_foreground = PointSourceForeground()
