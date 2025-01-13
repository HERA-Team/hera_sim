"""Models of signal-chain systematics.

This module defines several models of systematics that arise in the signal chain, for
example bandpass gains, reflections and cross-talk.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import astropy_healpix as aph
import numpy as np
from astropy import constants, units
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AnalyticBeam
from pyuvdata.beam_interface import BeamInterface
from scipy.signal.windows import blackmanharris

from . import DATA_PATH, interpolators, utils
from .components import component
from .defaults import _defaults

try:
    from uvtools.dspec import gen_window

    HAVE_UVTOOLS = True
except ModuleNotFoundError:
    HAVE_UVTOOLS = False


@component
class Gain:
    """Base class for systematic gains."""

    pass


class Bandpass(Gain):
    """Generate bandpass gains.

    Parameters
    ----------
    gain_spread
        Standard deviation of random gains. Default is about 10% variation across
        antennas.
    dly_rng
        Lower and upper range of delays which are uniformly sampled, in nanoseconds.
        Default is -20 ns to +20 ns.
    bp_poly
        Either an array of polynomial coefficients, a callable object that provides
        the bandpass amplitude as a function of frequency (in GHz), or a string
        providing a path to a file that can be read into an interpolation object.
        By default, the HERA Phase One bandpass is used.
    taper
        Taper to apply to the simulated gains. Default is to not apply a taper.
    taper_kwds
        Keyword arguments used in generating the taper.
    rng
        Random number generator.
    """

    _alias = ("gains", "bandpass_gain")
    is_multiplicative = True
    is_randomized = True
    return_type = "per_antenna"
    attrs_to_pull = dict(ants="antpos")

    def __init__(
        self,
        gain_spread: float | np.ndarray = 0.1,
        dly_rng: tuple = (-20, 20),
        bp_poly: str | callable | np.ndarray | None = None,
        taper: str | callable | np.ndarray | None = None,
        taper_kwds: dict | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            gain_spread=gain_spread,
            dly_rng=dly_rng,
            bp_poly=bp_poly,
            taper=taper,
            taper_kwds=taper_kwds,
            rng=rng,
        )

    def __call__(self, freqs, ants, **kwargs):
        """Generate the bandpass.

        Parameters
        ----------
        freqs : array_like of float
            Frequencies in GHz.
        ants : array_like of int
            Antenna numbers for which to produce gains.

        Returns
        -------
        dict
            Keys are antenna numbers and values are arrays of bandpass
            gains as a function of frequency.
        """
        # validate kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (gain_spread, dly_rng, bp_poly, taper, taper_kwds, rng) = (
            self._extract_kwarg_values(**kwargs)
        )
        rng = rng or np.random.default_rng()

        # get the bandpass gains
        bandpass = self._gen_bandpass(freqs, ants, gain_spread, bp_poly, rng=rng)

        # get the delay phases
        phase = self._gen_delay_phase(freqs, ants, dly_rng, rng=rng)

        if taper is None:
            taper = np.ones(freqs.size)
        elif isinstance(taper, str):
            if taper_kwds is None:
                taper_kwds = {}
            if taper == "tanh":
                taper = utils.tanh_window(freqs, **taper_kwds)
            elif HAVE_UVTOOLS:
                taper = gen_window(taper, freqs.size, **taper_kwds)
            else:  # pragma: no cover
                taper = np.ones(freqs.size)
                warnings.warn(
                    "uvtools is not installed, so you must provide the taper.",
                    stacklevel=1,
                )
        elif callable(taper):
            if taper_kwds is None:
                taper_kwds = {}
            taper = taper(freqs, **taper_kwds)
        elif not isinstance(taper, np.ndarray):
            raise ValueError("Unsupported choice of taper.")

        return {ant: bandpass[ant] * phase[ant] * taper for ant in ants}

    @_defaults
    def _gen_bandpass(self, freqs, ants, gain_spread=0.1, bp_poly=None, rng=None):
        if bp_poly is None:
            # default to the H1C bandpass
            bp_poly = np.load(DATA_PATH / "HERA_H1C_BANDPASS.npy")
        elif isinstance(bp_poly, str):
            # make an interpolation object, assume it's a polyfit
            bp_poly = interpolators.Bandpass(bp_poly)
        if callable(bp_poly):
            # support for interpolation objects
            bp_base = bp_poly(freqs)
        else:
            bp_base = np.polyval(bp_poly, freqs)
        window = blackmanharris(freqs.size)
        modes = np.abs(np.fft.fft(window * bp_base))
        gains = {}
        for ant in ants:
            delta_bp = np.fft.ifft(
                utils.gen_white_noise(freqs.size, rng=rng) * modes * gain_spread
            )
            gains[ant] = bp_base + delta_bp
        return gains

    def _gen_delay_phase(self, freqs, ants, dly_rng=(-20, 20), rng=None):
        phases = {}
        rng = rng or np.random.default_rng()
        for ant in ants:
            delay = rng.uniform(*dly_rng)
            phases[ant] = np.exp(2j * np.pi * delay * freqs)
        return phases


class Reflections(Gain):
    """Produce multiplicative reflection gains.

    Parameters
    ----------
    amp : float, optional
        Mean Amplitude of the reflection gains.
    dly : float, optional
        Mean delay of the reflection gains.
    phs : float, optional
        Phase of the reflection gains.
    conj : bool, optional
        Whether to conjugate the gain.
    amp_jitter : float, optional
        Final amplitudes are multiplied by a normal variable with mean one, and
        with standard deviation of ``amp_jitter``.
    dly_jitter : float, optional
        Final delays are offset by a normal variable with mean
        zero and standard deviation ``dly_jitter``.
    rng: np.random.Generator, optional
        Random number generator.
    """

    _alias = ("reflection_gains", "sigchain_reflections")
    is_multiplicative = True
    is_randomized = True
    return_type = "per_antenna"
    attrs_to_pull = dict(ants="antpos")

    def __init__(
        self,
        amp=None,
        dly=None,
        phs=None,
        conj=False,
        amp_jitter=0,
        dly_jitter=0,
        rng=None,
    ):
        super().__init__(
            amp=amp,
            dly=dly,
            phs=phs,
            conj=conj,
            amp_jitter=amp_jitter,
            dly_jitter=dly_jitter,
            rng=rng,
        )

    def __call__(self, freqs, ants, **kwargs):
        """Generate the bandpass.

        Parameters
        ----------
        freqs : array_like of float
            Frequencies in units inverse to :attr:`dly`.
        ants : array_like of int
            Antenna numbers for which to produce gains.

        Returns
        -------
        dict
            Keys are antenna numbers and values are arrays of bandpass
            gains.
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        amp, dly, phs, conj, amp_jitter, dly_jitter, rng = self._extract_kwarg_values(
            **kwargs
        )
        rng = rng or np.random.default_rng()

        # fill in missing kwargs
        amp, dly, phs = self._complete_params(
            ants, amp, dly, phs, amp_jitter, dly_jitter, rng=rng
        )

        # determine gains iteratively
        gains = {}
        for j, ant in enumerate(ants):
            # calculate the reflection coefficient
            eps = self.gen_reflection_coefficient(
                freqs, amp[j], dly[j], phs[j], conj=conj
            )
            gains[ant] = 1 + eps

        return gains

    @staticmethod
    def gen_reflection_coefficient(freqs, amp, dly, phs, conj=False):
        """Randomly generate reflection coefficients.

        Parameters
        ----------
        freqs : array_like of float
            Frequencies, units are arbitrary but must be the inverse of ``dly``.
        amp : array_like of float
            Either a scalar amplitude, or 1D with size Nfreqs, or 2D
            with shape (Ntimes, Nfreqs).
        dly : [type]
            Either a scalar delay, or 1D with size Nfreqs, or 2D
            with shape (Ntimes, Nfreqs). Units are inverse of ``freqs``.
        phs : [type]
            Either a scalar phase, or 1D with size Nfreqs, or 2D
            with shape (Ntimes, Nfreqs). Units radians.
        conj : bool, optional
            Whether to conjugate the gain.

        Returns
        -------
        array_like
            The reflection gains as a 2D array of (Ntimes, Nfreqs).
        """
        # this is copied directly from the old sigchain module
        # TODO: make this cleaner

        # helper function for checking type/shape
        def _type_check(arr):
            if isinstance(arr, np.ndarray):
                if arr.ndim == 1 and arr.size > 1:
                    # reshape the array to (Ntimes, 1)
                    arr = arr.reshape(-1, 1)
                    # raise a warning if it's the same length as freqs
                    if arr.shape[0] == Nfreqs:
                        warnings.warn(
                            "The input array had lengths Nfreqs "
                            "and is being reshaped as (Ntimes,1).",
                            stacklevel=1,
                        )
                elif arr.ndim > 1:
                    assert arr.shape[1] in (1, Nfreqs), (
                        "Frequency-dependent reflection coefficients must "
                        "match the input frequency array size."
                    )
            return arr

        Nfreqs = freqs.size
        amp = _type_check(amp)
        dly = _type_check(dly)
        phs = _type_check(phs)

        # actually make the reflection coefficient
        eps = amp * np.exp(1j * (2 * np.pi * freqs * dly + phs))

        # conjugate if desired
        return np.conj(eps) if conj else eps

    @staticmethod
    def _complete_params(
        ants, amp=None, dly=None, phs=None, amp_jitter=0, dly_jitter=0, rng=None
    ):
        # TODO: docstring isn't exactly accurate, should be updated
        """
        Generate parameters to calculate a reflection coefficient.

        Parameters
        ----------
        ants: iterable
            Iterable providing information about antenna numbers. Only used to
            determine how many entries each parameter needs to have.
        amp: float or length-2 array-like of float, optional
            If a single number is provided, then every antenna is assigned that
            number as the amplitude of the reflection. Otherwise, it should
            specify the lower and upper bounds, respectively, of the uniform
            distribution from which to randomly assign an amplitude for each
            antenna. Default is to randomly choose a number between 0 and 1.
        dly: float or length-2 array-like of float
            If a single number provided, then the reflection shows up at that
            delay for every antenna. Otherwise, it should specify the lower and
            upper bounds, respectively, of the uniform distribution from which
            to randomly assign delays. This should be specified in units of ns.
            Default is to randomly choose a delay between -20 and 20 ns.
        phs: float or length-2 array-like of float
            The phase of the reflection, or the bounds to use for assigning
            random phases. Default is to randomly choose a phase on [-pi, pi).
        amp_jitter: float, optional
            Standard deviation of multiplicative jitter to apply to amplitudes.
            For example, setting this to 1e-4 will introduce, on average, 0.01%
            deviations to each amplitude. Default is to not add any jitter.
        dly_jitter: float, optional
            Standard deviation of additive jitter to apply to delays, in ns.
            For example, setting this to 10 will introduce, on average, delay
            deviations up to 10 ns. (This is drawn from a normal distribution, so
            it is possible that delays will exceed the value provided.)
        rng: np.random.Generator, optional
            Random number generator.

        Returns
        -------
        amps: array-like of float
            Amplitude of reflection coefficient for each antenna.
        dlys: array-like of float
            Delay of each reflection coefficient, in ns, for each antenna.
        phases: array-like of float
            Phase of each reflection coefficient for each antenna.
        """

        rng = rng or np.random.default_rng()

        def broadcast_param(param, lower_bound, upper_bound, size):
            if param is None:
                return rng.uniform(lower_bound, upper_bound, size)
            elif np.isscalar(param):
                return np.ones(size, dtype=float) * param
            else:
                if len(param) == size:
                    return np.array(param, dtype=float)
                else:
                    return rng.uniform(*param, size)

        # Transform parameters into arrays.
        amps = broadcast_param(amp, 0, 1, len(ants))
        dlys = broadcast_param(dly, -20, 20, len(ants))
        phases = broadcast_param(phs, -np.pi, np.pi, len(ants))

        # Apply jitter.
        amps *= rng.normal(1, amp_jitter, len(ants))
        dlys += rng.normal(0, dly_jitter, len(ants))

        return amps, dlys, phases


class ReflectionSpectrum(Gain):
    """Generate many reflections between a range of delays.

    Amplitudes are distributed on a logarithmic grid, while delays are distributed
    on a linear grid. Effectively, this gives a reflection spectrum whose amplitude
    decreases exponentially over the range of delays specified.

    Parameters
    ----------
    n_copies
        Number of peaks in the reflection spectrum.
    amp_range
        Max/min of the amplitudes of the reflections in the spectrum. The
        spectrum amplitudes monotonically decrease (up to jitter).
    dly_range
        Min/max of the delays at which the reflections are injected, in ns.
    phs_range
        Bounds of the uniform distribution from which to draw reflection phases.
    amp_jitter
        Fractional jitter in amplitude across antennas for each of the reflections.
    dly_jitter
        Absolute jitter in delay across antennas for each of the reflections.
    amp_logbase
        Base of the logarithm to use for generating reflection amplitudes.
    rng
        Random number generator.

    Notes
    -----
    The generated amplitudes will be in the range
    ``amp_logbase ** amp_range[0]`` to ``amp_logbase ** amp_range[1]``.
    """

    _alias = ("reflection_spectrum",)
    is_multiplicative = True
    is_randomized = True
    return_type = "per_antenna"
    attrs_to_pull = dict(ants="antpos")

    def __init__(
        self,
        n_copies: int = 20,
        amp_range: tuple[float, float] = (-3, -4),
        dly_range: tuple[float, float] = (200, 1000),
        phs_range: tuple[float, float] = (-np.pi, np.pi),
        amp_jitter: float = 0.05,
        dly_jitter: float = 30,
        amp_logbase: float = 10,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            n_copies=n_copies,
            amp_range=amp_range,
            dly_range=dly_range,
            phs_range=phs_range,
            amp_jitter=amp_jitter,
            dly_jitter=dly_jitter,
            amp_logbase=amp_logbase,
            rng=rng,
        )

    def __call__(
        self, freqs: np.ndarray, ants: Sequence[int], **kwargs
    ) -> dict[int, np.ndarray]:
        """
        Generate a series of reflections.

        Parameters
        ----------
        freqs
            Frequencies at which to calculate the reflection coefficients.
            These should be provided in GHz.
        ants
            Antenna numbers for which to generate reflections.

        Returns
        -------
        reflection_gains
            Reflection gains for each antenna.
        """
        (
            n_copies,
            amp_range,
            dly_range,
            phs_range,
            amp_jitter,
            dly_jitter,
            amp_logbase,
            rng,
        ) = self._extract_kwarg_values(**kwargs)
        rng = rng or np.random.default_rng()

        amps = np.logspace(*amp_range, n_copies, base=amp_logbase)
        dlys = np.linspace(*dly_range, n_copies)
        phases = rng.uniform(*phs_range, n_copies)

        reflection_gains = {ant: np.ones(freqs.size, dtype=complex) for ant in ants}
        for amp, dly, phs in zip(amps, dlys, phases):
            reflections = Reflections(
                amp=amp,
                dly=dly,
                phs=phs,
                amp_jitter=amp_jitter,
                dly_jitter=dly_jitter,
                rng=rng,
            )
            reflections = reflections(freqs, ants)
            for ant, reflection in reflections.items():
                reflection_gains[ant] *= reflection

        return reflection_gains


@component
class Crosstalk:
    """Base class for cross-talk models."""

    pass


class CrossCouplingCrosstalk(Crosstalk, Reflections):
    """Generate cross-coupling xtalk.

    Parameters
    ----------
    amp : float, optional
        Mean Amplitude of the reflection gains.
    dly : float, optional
        Mean delay of the reflection gains.
    phs : float, optional
        Phase of the reflection gains.
    conj : bool, optional
        Whether to conjugate the gain.
    amp_jitter : float, optional
        Final amplitudes are multiplied by a normal variable with mean one, and
        with standard deviation of ``amp_jitter``.
    dly_jitter : float, optional
        Final delays are offset by a normal variable with mean
        zero and standard deviation ``dly_jitter``.
    rng : np.random.Generator, optional
        Random number generator.
    """

    _alias = ("cross_coupling_xtalk",)
    is_multiplicative = False
    is_randomized = True
    return_type = "per_baseline"
    attrs_to_pull = dict(autovis=None)

    def __init__(
        self,
        amp=None,
        dly=None,
        phs=None,
        conj=False,
        amp_jitter=0,
        dly_jitter=0,
        rng=None,
    ):
        super().__init__(
            amp=amp,
            dly=dly,
            phs=phs,
            conj=conj,
            amp_jitter=amp_jitter,
            dly_jitter=dly_jitter,
            rng=rng,
        )

    def __call__(self, freqs, autovis, **kwargs):
        """Copute the cross-correlations.

        Parameters
        ----------
        freqs : array_like of float
            Frequencies in units inverse to :attr:`dly`.
        autovis : array_like of float
            The autocorrelations as a function of frequency.

        Return
        ------
        array
            The cross-coupling contribution to the visibility,
            same shape as ``freqs``.
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # now unpack them
        amp, dly, phs, conj, amp_jitter, dly_jitter, rng = self._extract_kwarg_values(
            **kwargs
        )
        rng = rng or np.random.default_rng()

        # handle the amplitude, phase, and delay
        amp, dly, phs = self._complete_params(
            [1], amp, dly, phs, amp_jitter, dly_jitter, rng=rng
        )

        # Make reflection coefficient.
        eps = self.gen_reflection_coefficient(freqs, amp, dly, phs, conj=conj)

        # reshape if necessary
        if eps.ndim == 1:
            eps = eps.reshape((1, -1))

        # scale it by the autocorrelation and return the result
        return autovis * eps


class CrossCouplingSpectrum(Crosstalk):
    """Generate a cross-coupling spectrum.

    This generates multiple copies of :class:`CrossCouplingCrosstalk`
    into the visibilities.

    Parameters
    ----------
    n_copies : int, optional
        Number of random cross-talk models to add.
    amp_range : tuple, optional
        Two-tuple of floats specifying the range of amplitudes
        to be sampled regularly in log-space.
    dly_range : tuple, optional
        Two-tuple of floats specifying the range of delays to be
        sampled at regular intervals.
    phs_range : tuple, optional
        Range of uniformly random phases.
    amp_jitter : int, optional
        Standard deviation of random jitter to be applied to the
        regular amplitudes.
    dly_jitter : int, optional
        Standard deviation of the random jitter to be applied to
        the regular delays.
    amp_logbase: float, optional
        Base of the logarithm to use for generating amplitudes.
    symmetrize : bool, optional
        Whether to also produce statistically equivalent cross-talk at
        negative delays. Note that while the statistics are equivalent,
        both amplitudes and delays will be different random realizations.
    rng : np.random.Generator, optional
        Random number generator.

    Notes
    -----
    The generated amplitudes will be in the range
    ``amp_logbase ** amp_range[0]`` to ``amp_logbase ** amp_range[1]``.
    """

    _alias = ("cross_coupling_spectrum", "xtalk_spectrum")
    is_randomized = True
    return_type = "per_baseline"
    attrs_to_pull = dict(autovis=None)

    def __init__(
        self,
        n_copies=10,
        amp_range=(-4, -6),
        dly_range=(1000, 1200),
        phs_range=(-np.pi, np.pi),
        amp_jitter=0,
        dly_jitter=0,
        amp_logbase=10,
        symmetrize=True,
        rng=None,
    ):
        super().__init__(
            n_copies=n_copies,
            amp_range=amp_range,
            dly_range=dly_range,
            phs_range=phs_range,
            amp_jitter=amp_jitter,
            dly_jitter=dly_jitter,
            amp_logbase=amp_logbase,
            symmetrize=symmetrize,
            rng=rng,
        )

    def __call__(self, freqs, autovis, **kwargs):
        """Compute the cross-correlations.

        Parameters
        ----------
        freqs : array_like of float
            Frequencies in units inverse to :attr:`dly`.
        autovis : array_like of float
            The autocorrelations as a function of frequency.

        Return
        ------
        array
            The cross-coupling contribution to the visibility,
            same shape as ``freqs``.
        """
        self._check_kwargs(**kwargs)

        (
            n_copies,
            amp_range,
            dly_range,
            phs_range,
            amp_jitter,
            dly_jitter,
            amp_logbase,
            symmetrize,
            rng,
        ) = self._extract_kwarg_values(**kwargs)

        # Construct the arrays of amplitudes and delays.
        amps = np.logspace(*amp_range, n_copies, base=amp_logbase)
        dlys = np.linspace(*dly_range, n_copies)

        # Construct the spectrum of crosstalk.
        crosstalk_spectrum = np.zeros(autovis.shape, dtype=complex)
        for amp, dly in zip(amps, dlys):
            gen_xtalk = CrossCouplingCrosstalk(
                amp=amp,
                dly=dly,
                phs=phs_range,
                amp_jitter=amp_jitter,
                dly_jitter=dly_jitter,
                rng=rng,
            )

            crosstalk_spectrum += gen_xtalk(freqs, autovis)
            if symmetrize:
                # Note: this will have neither the same jitter realization nor
                # the same phase as the first crosstalk spectrum.
                crosstalk_spectrum += gen_xtalk(freqs, autovis, dly=-dly)

        return crosstalk_spectrum


class MutualCoupling(Crosstalk):
    r"""Simulate mutual coupling according to Josaitis+ 2022.

    This class simulates the "first-order coupling" between visibilities in an
    array. The model assumes that coupling is induced via re-radiation of
    incident astrophysical radiation due to an impedance mismatch at the
    antenna feed, and that the re-radiated signal is in the far-field of every
    other antenna in the array. Full details can be found here:

        `MNRAS <https://doi.org/10.1093/mnras/stac916>`_

        `arXiv <https://arxiv.org/abs/2110.10879>`_

    The essential equations from the paper are Equations 9 and 19. The
    implementation here effectively calculates Equation 19 for every
    visibility in the provided data. The original publication contains an
    error in Equation 9 (the effective height in transmission should have a
    complex conjugation applied), which we correct for in our implementation.
    In addition to this, we assume that every antenna feed has the same
    impedance, reflection coefficient, and effective height. Applying the
    correct conjugation, and enforcing these assumptions, the first-order
    correction to the visibility :math:`{\bf V}_{ij}` can be written as:

    .. math::

        {\bf V}_{ij}^{\rm xt} = \sum_k \Bigl[ (1-\delta_{kj}) {\bf V}_{ik}^0
        {\bf X}_{jk}^\dagger + (1-\delta_{ik}) {\bf X}_{ik} {\bf V}_{kj}^0
        \Bigr],

    where the "xt" superscript is shorthand for "crosstalk", the "0"
    superscript refers to the "zeroth-order" visibilities, :math:`\delta_{ij}`
    is the Kronecker delta, and :math:`{\bf X}_{ij}` is a "coupling matrix"
    that describes how radiation emitted from antenna :math:`j` is received by
    antenna :math:`i`. The coupling matrix can be written as

    .. math::

        {\bf X}_{jk} \equiv \frac{i\eta_0}{4\lambda} \frac{\Gamma_k}{R_k}
        \frac{e^{i2\pi\nu\tau_{jk}}}{b_{jk}} {\bf J}_j (\hat{\bf b}_{jk})
        {\bf J}_k(\hat{\bf b}_{kj})^\dagger h_0^2,

    where :math:`\Gamma` is the reflection coefficient, :math:`R` is the real
    part of the impedance, :math:`\eta_0` is the impedance of free space,
    :math:`\lambda` is the wavelength of the radiation, :math:`\nu` is the
    frequency of the radiation, :math:`\tau=b/c` is the delay of the baseline,
    :math:`b` is the baseline length, :math:`\hat{\bf b}_{ij}` is a unit
    vector pointing from antenna :math:`i` to antenna :math:`j`, :math:`{\bf J}`
    is the Jones matrix describing the antenna's peak-normalized far-field
    radiation pattern, and :math:`h_0` is the amplitude of the antenna's
    effective height.

    The boldfaced variables without any overhead decorations indicate 2x2
    matrices:

    .. math::

        {\bf V} = \begin{pmatrix}
            V_{XX} & V_{XY} \\ V_{YX} & V_{YY}
        \end{pmatrix},
        \quad
        {\bf J} = \frac{1}{h_0} \begin{pmatrix}
            h_{X\theta} & h_{X\phi} \\ h_{Y\theta} & h_{Y\phi}
        \end{pmatrix}

    The effective height can be rewritten as

    .. math::

        h_0^2 = \frac{4\lambda^2 R}{\eta_0 \Omega_p}

    where :math:`\Omega_p` is the beam area (i.e. integral of the peak-normalized
    power beam). Substituting this in to the previous expression for the coupling
    coefficient and taking antennas to be identical gives

    .. math::

        {\bf X}_{jk} = \frac{i\Gamma}{\Omega_p} \frac{e^{i2\pi\nu\tau_{jk}}}
        {b_{jk}/\lambda} {\bf J}(\hat{\bf b}_{jk}) {\bf J}(\hat{\bf b}_{kj})^\dagger.

    In order to efficiently simulate the mutual coupling, the antenna and
    polarization axes of the visibilities and coupling matrix are combined
    into a single "antenna-polarization" axis, and the problem is recast as a
    simple matrix multiplication.

    Parameters
    ----------
    uvbeam
        The beam (i.e. Jones matrix) to be used for calculating the coupling
        matrix. This may either be a :class:`pyuvdata.UVBeam` object, a path
        to a file that may be read into a :class:`pyuvdata.UVBeam` object, or
        a string identifying which :class:`pyuvdata.analytic_beam.AnalyticBeam` to use.
        Not required if providing a pre-calculated coupling matrix.
    reflection
        The reflection coefficient to use for calculating the coupling matrix.
        Should be either a :class:`np.ndarray` or an interpolation object that
        gives the reflection coefficient as a function of frequency (in GHz).
        Not required if providing a pre-calculated coupling matrix.
    omega_p
        The integral of the peak-normalized power beam as a function of frequency
        (in GHz). Not required if providing a pre-calculated coupling matrix.
    ant_1_array
        Array of integers specifying the number of the first antenna in each
        visibility. Required for calculating the coupling matrix and the
        coupled visibilities.
    ant_2_array
        Array of integers specifying the number of the second antenna in each
        visibility.
    pol_array
        Array of integers representing polarization numbers, following the
        convention used for :class:`pyuvdata.UVData` objects. Required for
        calculating the coupled visibilities.
    array_layout
        Dictionary mapping antenna numbers to their positions in local East-
        North-Up coordinates, expressed in meters. Not required if providing
        a pre-calculated coupling matrix.
    coupling_matrix
        Matrix describing how radiation is coupled between antennas in the
        array. Should have shape `(1, n_freqs, 2*n_ants, 2*n_ants)`. The even
        elements along the "antenna-polarization" axes correspond to the "X"
        polarization; the odd elements correspond to the "Y" polarization.
    pixel_interp
        The name of the spatial interpolation method used for the beam. Not
        required if using an analytic beam or if providing a pre-computed
        coupling matrix.
    freq_interp
        The order of the spline to be used for interpolating the beam in
        frequency. Not required if using an analytic beam or if providing a
        pre-computed coupling matrix.
    beam_kwargs
        Additional keywords used for either reading in a beam or creating an
        analytic beam.
    use_numba
        Whether to use ``numba`` for accelerating the simulation. Default is
        to use ``numba`` if it is installed.
    """

    _alias = ("mutual_coupling", "first_order_coupling")
    return_type = "full_array"
    attrs_to_pull = dict(
        ant_1_array="ant_1_array",
        ant_2_array="ant_2_array",
        pol_array="polarization_array",
        array_layout="antpos",
        visibilities="data_array",
    )

    def __init__(
        self,
        uvbeam: UVBeam | str | Path | None = None,
        reflection: np.ndarray | Callable | None = None,
        omega_p: np.ndarray | Callable | None = None,
        ant_1_array: np.ndarray | None = None,
        ant_2_array: np.ndarray | None = None,
        pol_array: np.ndarray | None = None,
        array_layout: dict | None = None,
        coupling_matrix: np.ndarray | None = None,
        pixel_interp: str = "az_za_simple",
        freq_interp: str = "cubic",
        beam_kwargs: dict | None = None,
        use_numba: bool = True,
    ):
        super().__init__(
            uvbeam=uvbeam,
            reflection=reflection,
            omega_p=omega_p,
            ant_1_array=ant_1_array,
            ant_2_array=ant_2_array,
            pol_array=pol_array,
            array_layout=array_layout,
            coupling_matrix=coupling_matrix,
            pixel_interp=pixel_interp,
            freq_interp=freq_interp,
            beam_kwargs=beam_kwargs or {},
            use_numba=use_numba,
        )

    def __call__(
        self, freqs: np.ndarray, visibilities: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Calculate the first-order coupled visibilities.

        Parameters
        ----------
        freqs
            The observed frequencies, in GHz.
        visibilities
            The full set of visibilities for the array. Should have shape
            `(n_bls*n_times, n_freqs, [1,] n_pols)`.
        kwargs
            Additional parameters to use instead of the current attribute
            values for the class instance. See the class docstring for details.

        Returns
        -------
        xt_vis
            The first-order correction to the visibilities due to mutual
            coupling between array elements. Has the same shape as the provided
            visibilities.

        Notes
        -----
        This method is somewhat memory hungry, as it produces two arrays which
        are each twice as large as the input visibility array in intermediate
        steps of the calculation.
        """
        self._check_kwargs(**kwargs)
        (
            beam,
            reflection,
            omega_p,
            ant_1_array,
            ant_2_array,
            pol_array,
            array_layout,
            coupling_matrix,
            pixel_interp,
            freq_interp,
            beam_kwargs,
            use_numba,
        ) = self._extract_kwarg_values(**kwargs)

        # Do all our sanity checks up front. First, check the array.
        data_ants = set(ant_1_array).union(ant_2_array)
        antpos_ants = set(array_layout.keys())
        if antpos_ants.issubset(data_ants) and antpos_ants != data_ants:
            raise ValueError("Full array layout not provided.")

        # Now, check that the input beam is OK in case we need to use it.
        if coupling_matrix is None:
            beam = MutualCoupling._handle_beam(beam, **beam_kwargs)

            # This already happens in build_coupling_matrix, but the reshape
            # step is not a trivial amount of time, so it's better to do it
            # again here.
            self._check_beam_is_ok(beam)

        # Let's make sure that we're only using antennas that are in the data.
        antpos_ants = antpos_ants.intersection(data_ants)
        array_layout = {ant: array_layout[ant] for ant in sorted(antpos_ants)}
        antenna_numbers = np.array(list(array_layout.keys()))

        # Figure out how to reshape the visibility array
        n_bls = np.unique(np.vstack([ant_1_array, ant_2_array]), axis=1).shape[1]
        n_ants = antenna_numbers.size
        n_times = ant_1_array.size // n_bls
        n_freqs = visibilities.shape[1]
        n_pols = visibilities.shape[-1]
        visibilities = utils.reshape_vis(
            vis=visibilities,
            ant_1_array=ant_1_array,
            ant_2_array=ant_2_array,
            pol_array=pol_array,
            antenna_numbers=antenna_numbers,
            n_times=n_times,
            n_freqs=n_freqs,
            n_ants=n_ants,
            n_pols=n_pols,
            invert=False,
            use_numba=use_numba,
        )

        if coupling_matrix is None:
            coupling_matrix = self.build_coupling_matrix(
                freqs=freqs,
                ant_1_array=ant_1_array,
                ant_2_array=ant_2_array,
                array_layout=array_layout,
                uvbeam=beam,
                reflection=reflection,
                omega_p=omega_p,
                pixel_interp=pixel_interp,
                freq_interp=freq_interp,
                **beam_kwargs,
            )

        # Now actually calculate the mutual coupling.
        xt_vis = utils.matmul(coupling_matrix, visibilities, use_numba=use_numba)
        xt_vis += xt_vis.conj().transpose(0, 1, 3, 2)

        # Return something with the same shape as the input data array.
        return utils.reshape_vis(
            vis=xt_vis,
            ant_1_array=ant_1_array,
            ant_2_array=ant_2_array,
            pol_array=pol_array,
            antenna_numbers=antenna_numbers,
            n_times=n_times,
            n_freqs=n_freqs,
            n_ants=n_ants,
            n_pols=n_pols,
            invert=True,
            use_numba=use_numba,
        )

    @staticmethod
    def build_coupling_matrix(
        freqs: np.ndarray,
        array_layout: dict,
        uvbeam: UVBeam | str,
        reflection: np.ndarray | Callable | None = None,
        omega_p: np.ndarray | Callable | None = None,
        pixel_interp: str | None = "az_za_simple",
        freq_interp: str | None = "cubic",
        **beam_kwargs,
    ) -> np.ndarray:
        """Calculate the coupling matrix used for mutual coupling simulation.

        See the :class:`MutualCoupling` class docstring for a description of
        the coupling matrix.

        Parameters
        ----------
        freqs
            The observed frequencies, in GHz.
        array_layout
            Dictionary mapping antenna numbers to their positions in local East-
            North-Up coordinates, expressed in meters. Not required if providing
            a pre-calculated coupling matrix.
        uvbeam
            The beam (i.e. Jones matrix) to be used for calculating the coupling
            matrix. This may either be a :class:`pyuvdata.UVBeam` object, a path
            to a file that may be read into a :class:`pyuvdata.UVBeam` object, or
            an :class:`pyuvdata.analytic_beam.AnalyticBeam`. Not
            required if providing a pre-calculated coupling matrix.
        reflection
            The reflection coefficient to use for calculating the coupling matrix.
            Should be either a :class:`np.ndarray` or an interpolation object that
            gives the reflection coefficient as a function of frequency (in GHz).
        omega_p
            The integral of the peak-normalized power beam as a function of frequency
            (in GHz). If this is not provided, then it will be calculated from the
            provided beam model.
        pixel_interp
            The name of the spatial interpolation method used for the beam. Not
            required if using an analytic beam or if providing a pre-computed
            coupling matrix.
        freq_interp
            The order of the spline to be used for interpolating the beam in
            frequency. Not required if using an analytic beam or if providing a
            pre-computed coupling matrix.
        beam_kwargs
            Additional keywords used for either reading in a beam or creating an
            analytic beam.
        """
        n_ants = len(array_layout)
        antenna_numbers = np.array(sorted(array_layout.keys()))
        enu_antpos = np.array([array_layout[ant] for ant in antenna_numbers])
        antpair2angle = utils.find_baseline_orientations(
            antenna_numbers=antenna_numbers, enu_antpos=enu_antpos
        )
        antpair2angle = {
            antpair: np.round(angle, 2) for antpair, angle in antpair2angle.items()
        }
        unique_angles = np.array(list(set(antpair2angle.values())))

        # Make sure the reflection coefficients and resistances make sense.
        if reflection is None:
            reflection = np.ones_like(freqs)
        elif callable(reflection):
            reflection = reflection(freqs)
        if reflection.size != freqs.size:
            raise ValueError("Reflection coefficients have the wrong shape.")

        if omega_p is None:
            warnings.warn(
                "Calculating the power beam integral; this may take a while.",
                stacklevel=1,
            )
            if isinstance(uvbeam, AnalyticBeam):
                power_beam = uvbeam.to_uvbeam(
                    freq_array=freqs * units.GHz.to("Hz"),
                    beam_type='power',
                    pixel_coordinate_system='healpix',
                    nside=128
                )
            else:
                power_beam = uvbeam.copy()
                power_beam.efield_to_power()
                power_beam = power_beam.interp(
                    freq_array=freqs * units.GHz.to("Hz"),
                    new_object=True,
                    interpolation_function=pixel_interp,
                    freq_interp_kind=freq_interp,
                )  # Interpolate to the desired frequencies
                power_beam.to_healpix()

            power_beam.peak_normalize()
            omega_p = power_beam.get_beam_area(pol="xx").real
            del power_beam
        elif callable(omega_p):
            omega_p = omega_p(freqs)
        if omega_p.size != freqs.size:
            raise ValueError("Beam integral has the wrong shape.")

        # Check the beam is OK and make it smaller if it's too big.
        uvbeam = MutualCoupling._handle_beam(uvbeam, **beam_kwargs)
        MutualCoupling._check_beam_is_ok(uvbeam)
        if isinstance(uvbeam, UVBeam):
            uvbeam = uvbeam.copy()
            uvbeam.peak_normalize()
            if uvbeam.Naxes2 > 5:
                # We only need two points on either side of the horizon.
                za_array = uvbeam.axis2_array
                horizon_ind = np.argmin(np.abs(za_array - np.pi / 2))
                horizon_select = np.arange(horizon_ind - 2, horizon_ind + 3)
                # Do it this way to not overwrite uvbeam in memory.
                uvbeam = uvbeam.select(
                    axis2_inds=horizon_select, inplace=False, run_check=False
                )

        # Now we'll actually interpolate the beam.
        # The end shape is (n_az, n_freq, 2, 2).
        uvbeam = BeamInterface(uvbeam)

        jones_matrices = uvbeam.compute_response(
            az_array=unique_angles,
            za_array=np.ones_like(unique_angles) * np.pi / 2,
            freq_array=freqs * units.GHz.to("Hz"),
        ).transpose(3, 2, 1, 0)

        jones_matrices = {
            angle: jones_matrices[i] for i, angle in enumerate(unique_angles)
        }

        # Now let's actually make the coupling matrix.
        coupling_matrix = np.zeros(
            (1, freqs.size, 2 * n_ants, 2 * n_ants), dtype=complex
        )
        for i, ai in enumerate(antenna_numbers):
            for j, aj in enumerate(antenna_numbers[i + 1 :]):
                j += i + 1
                # Calculate J(b_ij)J(b_ji)^\dag
                jones_ij = jones_matrices[antpair2angle[ai, aj]]
                jones_ji = jones_matrices[antpair2angle[aj, ai]]
                jones_prod = jones_ij @ jones_ji.conj().transpose(0, 2, 1)

                # If we wanted to add a baseline orientation/length cut,
                # then this is where we would do it.
                bl_len = np.linalg.norm(enu_antpos[j] - enu_antpos[i])
                delay = np.exp(
                    2j * np.pi * freqs * bl_len / constants.c.to("m/ns").value
                ).reshape(-1, 1, 1)
                coupling = delay * jones_prod / bl_len

                # Fill in the upper-triangular part
                # Even indices are "X" feed; odd are "Y" feed
                coupling_matrix[0, :, ::2, ::2][:, i, j] = coupling[:, 0, 0]
                coupling_matrix[0, :, 1::2, ::2][:, i, j] = coupling[:, 0, 1]
                coupling_matrix[0, :, ::2, 1::2][:, i, j] = coupling[:, 1, 0]
                coupling_matrix[0, :, 1::2, 1::2][:, i, j] = coupling[:, 1, 1]

                # Now fill in the lower-triangular part
                # Remember we're assuming identical antennas
                coupling_matrix[0, :, ::2, ::2][:, j, i] = coupling[:, 0, 0]
                coupling_matrix[0, :, 1::2, ::2][:, j, i] = coupling[:, 0, 1]
                coupling_matrix[0, :, ::2, 1::2][:, j, i] = coupling[:, 1, 0]
                coupling_matrix[0, :, 1::2, 1::2][:, j, i] = coupling[:, 1, 1]

        # Now let's tack on the prefactor
        wavelengths = constants.c.si.value / (freqs * units.GHz.to("Hz"))
        coupling_matrix *= (1j * reflection * wavelengths / omega_p).reshape(
            1, -1, 1, 1
        )
        return coupling_matrix

    @staticmethod
    def _check_beam_is_ok(beam):
        if isinstance(beam, AnalyticBeam):
            return
        if getattr(beam, "pixel_coordinate_system", "") != "az_za":
            raise ValueError("Beam must be given in az/za coordinates.")
        if beam.beam_type != "efield":
            raise NotImplementedError("Only E-field beams are supported.")

    @staticmethod
    def _handle_beam(beam, **beam_kwargs):
        if isinstance(beam, (AnalyticBeam, UVBeam)):
            return beam
        if Path(beam).exists():
            return UVBeam.from_file(beam, **beam_kwargs)
        raise ValueError("uvbeam has incorrect format")


class OverAirCrossCoupling(Crosstalk):
    r"""Crosstalk model based on the mechanism described in HERA Memo 104.

    This model describes first-order coupling between a visibility :math:`V_{ij}`
    and the autocorrelations for each antenna involved. Physically, it is modeled
    as the signal from one antenna traveling to the receiverator, then being
    broadcast to the other antenna. Under this model, the cross-coupling component
    :math:`V_{ij}^{\rm cc}` can be described via

    .. math::

        V_{ij}^{\rm cc} = \epsilon_{ij}^* V_{ii} + \epsilon_{ji} V_{jj},

    where the reflection coefficient :math:`\epsilon_{ij}` is modeled as

    .. math::

        \epsilon_{ij} = A_i \exp \bigl[2\pi i\nu(\tau_{i,{\rm cable}} +
        \tau_{X \rightarrow j} ) \bigr].

    Here, :math:`X` denotes the position of the receiverator (or rather, where the
    excess signal is radiated from), and the indices :math:`i,j` refer to antennas.
    So, :math:`\tau_{i,{\rm cable}}` is the delay from the signal traveling down
    the cable from antenna :math:`i` to the receiverator, and :math:`\tau_{X
    \rightarrow j}` denotes the delay from the signal traveling over-the-air from
    the receiverator to antenna :math:`j`. As usual, :math:`A_i` is the amplitude
    of the reflection coefficient. Here, the amplitude is described by three free
    parameters, :math:`a, \vec{r}_X, \beta`:

    .. math::
       A_i = a |\vec{r}_i - \vec{r}_X|^\beta.

    :math:`a` is a base amplitude, :math:`\vec{r}_X` is the receiverator position,
    and :math:`\beta` describes how quickly the amplitude falls off with distance
    from the receiverator, and is typically taken to be negative. For more details,
    refer to HERA Memo 104 for more details:

    http://reionization.org/manual_uploads/HERA104_Crosstalk_Physical_Model.html

    Parameters
    ----------
    emitter_pos
        Receiverator position, in meters, in local ENU coordinates.
    cable_delays
        Mapping from antenna numbers to cable delays, in nanoseconds.
    base_amp
        Base amplitude of reflection coefficient. If `amp_slope` is set to 0, then
        this is the amplitude of all of the reflection coefficients.
    amp_norm
        Distance from the receiverator, in meteres, at which the cross-coupling
        amplitude is equal to ``base_amp``.
    amp_slope
        Power-law index describing how rapidly the reflection coefficient decays
        with distance from the receiverator.
    amp_decay_base
        Logarithmic base to use when generating the additional peaks in the
        cross-coupling spectrum.
    n_copies
        Number of peaks in the cross-coupling spectrum at positive and negative
        delays, separately.
    amp_jitter
        Fractional jitter to apply to the amplitudes of the peaks in the
        cross-coupling spectrum.
    dly_jitter
        Absolute jitter to apply to the delays of the peaks in the cross-coupling
        spectrum, in nanoseconds.
    max_delay
        Magnitude of the maximum delay to which the cross-coupling spectrum extends,
        in nanoseconds.
    amp_decay_fac
        Ratio of the amplitude of the last peak in the cross-coupling spectrum to
        the first peak. In other words, how much the cross-coupling spectrum decays
        over the full range of delays it covers.
    rng
        Random number generator.

    See Also
    --------
    :class:`CrossCouplingSpectrum`
    """

    is_randomized = True
    return_type = "per_baseline"
    attrs_to_pull = dict(antpair=None, autovis_i=None, autovis_j=None)

    def __init__(
        self,
        emitter_pos: np.ndarray | Sequence | None = None,
        cable_delays: dict[int, float] | None = None,
        base_amp: float = 2e-5,
        amp_norm: float = 100,
        amp_slope: float = -1,
        amp_decay_base: float = 10,
        n_copies: int = 10,
        amp_jitter: float = 0,
        dly_jitter: float = 0,
        max_delay: float = 2000,
        amp_decay_fac: float = 1e-2,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            emitter_pos=emitter_pos,
            cable_delays=cable_delays or {},
            base_amp=base_amp,
            amp_norm=amp_norm,
            amp_slope=amp_slope,
            amp_decay_base=amp_decay_base,
            n_copies=n_copies,
            amp_jitter=amp_jitter,
            dly_jitter=dly_jitter,
            max_delay=max_delay,
            amp_decay_fac=amp_decay_fac,
            rng=rng,
        )

    def __call__(
        self,
        freqs: np.ndarray,
        antpair: tuple[int, int],
        antpos: dict[int, np.ndarray],
        autovis_i: np.ndarray,
        autovis_j: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Generate a cross-coupling spectrum modeled via HERA Memo 104.

        Parameters
        ----------
        freqs
            Frequencies at which to evaluate the reflection coefficients, in GHz.
        antpair
            The two antennas involved in forming the visibility.
        antpos
            Mapping from antenna numbers to positions in meters, in local ENU
            coordinates.
        autovis_i
            Autocorrelation for the first antenna in the pair.
        autovis_j
            Autocorrelation for the second antenna in the pair.

        Returns
        -------
        xtalk_vis
            Array with the cross-coupling visibility. Has the same shape as the input
            autocorrelations. This systematic is not applied to the auto-correlations.
        """
        self._check_kwargs(**kwargs)
        (
            emitter_pos,
            cable_delay,
            base_amp,
            amp_norm,
            amp_slope,
            amp_decay_base,
            n_copies,
            amp_jitter,
            dly_jitter,
            max_delay,
            amp_decay_fac,
            rng,
        ) = self._extract_kwarg_values(**kwargs)

        ai, aj = antpair
        if ai == aj:
            return np.zeros_like(autovis_i)

        if emitter_pos is None:
            emitter_pos = np.zeros(3, dtype=float)
        xi = np.linalg.norm(antpos[ai] - np.asarray(emitter_pos))
        xj = np.linalg.norm(antpos[aj] - np.asarray(emitter_pos))

        log_scale = np.log(amp_decay_base)

        def log(x):
            return np.log(x) / log_scale

        amp_i = base_amp * (xi / amp_norm) ** amp_slope
        amp_j = base_amp * (xj / amp_norm) ** amp_slope
        dly_i = xi / constants.c.to("m/ns").value
        dly_j = xj / constants.c.to("m/ns").value
        dly_ij = cable_delay[ai] + dly_j
        dly_ji = cable_delay[aj] + dly_i

        xt_ij = CrossCouplingSpectrum(
            n_copies=n_copies,
            amp_range=(log(amp_i), log(amp_i * amp_decay_fac)),
            dly_range=(-dly_ij, -max_delay),
            amp_jitter=amp_jitter,
            dly_jitter=dly_jitter,
            amp_logbase=amp_decay_base,
            symmetrize=False,
            rng=rng,
        )
        xt_ji = CrossCouplingSpectrum(
            n_copies=n_copies,
            amp_range=(log(amp_j), log(amp_j * amp_decay_fac)),
            dly_range=(dly_ji, max_delay),
            amp_jitter=amp_jitter,
            dly_jitter=dly_jitter,
            amp_logbase=amp_decay_base,
            symmetrize=False,
            rng=rng,
        )

        return xt_ij(freqs, autovis_i) + xt_ji(freqs, autovis_j)


class WhiteNoiseCrosstalk(Crosstalk):
    """Generate cross-talk that is simply white noise.

    Parameters
    ----------
    amplitude : float, optional
        The amplitude of the white noise spectrum (i.e. its standard deviation).
    rng : np.random.Generator, optional
        Random number generator.
    """

    _alias = ("whitenoise_xtalk", "white_noise_xtalk")
    is_randomized = True
    return_type = "per_baseline"

    def __init__(self, amplitude=3.0, rng=None):
        super().__init__(amplitude=amplitude, rng=rng)

    def __call__(self, freqs, **kwargs):
        """Compute the cross-correlations.

        Parameters
        ----------
        freqs : array_like of float
            Frequencies in units inverse to :attr:`dly`.

        Return
        ------
        array
            The cross-coupling contribution to the visibility,
            same shape as ``freqs``.
        """
        # check the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (amplitude, rng) = self._extract_kwarg_values(**kwargs)

        # why choose this size for the convolving kernel?
        kernel = np.ones(50 if freqs.size > 50 else int(freqs.size / 2))

        # generate the crosstalk
        xtalk = np.convolve(utils.gen_white_noise(freqs.size, rng=rng), kernel, "same")

        # scale the result and return
        return amplitude * xtalk


def apply_gains(
    vis: float | np.ndarray, gains: dict[int, float | np.ndarray], bl: tuple[int, int]
) -> np.ndarray:
    """Apply antenna-based gains to a visibility.

    Parameters
    ----------
    vis
        The visibilities of the given baseline as a function of frequency.
    gains
        Dictionary where keys are antenna numbers and values are arrays of
        gains as a function of frequency.
    bl
        2-tuple of integers specifying the antenna numbers in the particular
        baseline.

    Returns
    -------
    vis
        The visibilities with gains applied.
    """
    # get the gains for each antenna in the baseline
    # don't apply a gain if the antenna isn't found
    gi = 1.0 if bl[0] not in gains else gains[bl[0]]
    gj = 1.0 if bl[1] not in gains else gains[bl[1]]

    # if neither antenna is in the gains dict, do nothing
    if bl[0] not in gains and bl[1] not in gains:
        return vis

    # form the gain term for the given baseline
    gain = gi * np.conj(gj)

    # reshape if need be
    if gain.ndim == 1:
        gain.shape = (1, -1)

    return vis * gain


def vary_gains_in_time(
    gains,
    times,
    freqs=None,
    delays=None,
    parameter="amp",
    variation_ref_time=None,
    variation_timescale=None,
    variation_amp=0.05,
    variation_mode="linear",
    rng=None,
):
    r"""
    Vary gain amplitudes, phases, or delays in time.

    Notes
    -----
    If the gains initially have the form

    .. math:: g(\nu) = g_0(\nu)\exp(i 2\pi\nu\tau + i\phi)

    then the output gains have the form

    .. math:: g(\nu,t) = g_0(\nu,t)\exp \bigl( i2\pi\nu\tau(t) + i\phi(t)\bigr).


    Parameters
    ----------
    gains: dict
        Dictionary mapping antenna numbers to gain spectra/waterfalls.
    times: array-like of float
        Times at which to simulate time variation. Should be the same length as
        the data to which the gains will be applied. Should also be in the same
        units as ``variation_ref_time`` and ``variation_timescale``.
    freqs: array-like of float, optional
        Frequencies at which the gains are evaluated, in GHz. Only needs to be
        specified for adding time variation to the delays.
    delays: dict, optional
        Dictionary mapping antenna numbers to gain delays, in ns.
    parameter: str, optional
        Which gain parameter to vary; must be one of ("amp", "phs", "dly").
    variation_ref_time: float or array-like of float, optional
        Reference time(s) used for generating time variation. For linear and
        sinusoidal variation, this is the time where the gains are equal to their
        original, time-independent values. Should be in the same units as the
        ``times`` array. Default is to use the center of the ``times`` provided.
    variation_timescale: float or array-like of float, optional
        Timescale(s) for one cycle of the variation(s), in the same units as
        the provided ``times``. Default is to use the duration of the entire
        ``times`` array.
    variation_amp: float or array-like of float, optional
        Amplitude(s) of the variation(s) introduced. This is *not* the peak-to-peak
        amplitude! This also does not have exactly the same interpretation for each
        type of variation mode. For amplitude and delay variation, this represents
        the amplitude of modulations--so it can be interpreted as a fractional
        variation. For phase variation, this represents an absolute, time-dependent
        phase offset to introduce to the gains; however, it is still *not* a
        peak-to-peak amplitude.
    variation_mode: str or array-like of str, optional
        Which type(s) of variation to simulate. Supported modes are "linear",
        "sinusoidal", and "noiselike". Default is "linear". Note that the "linear"
        mode produces a triangle wave variation with period twice the corresponding
        timescale; this ensures that the gains vary linearly over the entire set of
        provided times if the default variation timescale is used.
    rng: np.random.Generator, optional
        Random number generator.

    Returns
    -------
    time_varied_gains: dict
        Dictionary mapping antenna numbers to gain waterfalls.
    """
    # Parameter checking/preparation.
    if np.isscalar(times) or not np.isrealobj(times):
        raise TypeError("times must be an array of real numbers.")
    if not isinstance(gains, dict):
        raise TypeError("gains must be provided as a dictionary.")
    if parameter not in ("amp", "phs", "dly"):
        raise ValueError("parameter must be one of 'amp', 'phs', or 'dly'.")

    times = np.array(times)
    gain_shapes = [np.array(gain).shape for gain in gains.values()]
    if any(gain_shape != gain_shapes[0] for gain_shape in gain_shapes):
        raise ValueError("Gains must all have the same shape.")
    gain_shape = gain_shapes[0]

    if parameter == "dly":
        if freqs is None or delays is None:
            raise ValueError(
                "In order to vary delays, you must provide both the corresponding "
                "frequency array and a dictionary mapping antenna numbers to delays."
            )

        freqs = np.array(freqs)
        if set(delays.keys()) != set(gains.keys()):
            raise ValueError("Delays and gains must have the same keys.")
        if len(gain_shape) == 2:
            if gain_shape != (times.size, freqs.size):
                raise ValueError("Gain waterfalls must have shape (Ntimes, Nfreqs).")
        elif len(gain_shape) == 1:
            if gain_shape[0] != freqs.size:
                raise ValueError(
                    "Gain spectra must be the same length as the provided frequencies."
                )
        else:
            raise ValueError("Gain dictionary values must be at most 2-dimensional.")

    # Setup for handling multiple modes of variation.
    if variation_ref_time is None:
        variation_ref_time = (np.median(times),)
    if variation_timescale is None:
        variation_timescale = (times[-1] - times[0],)
        if utils._listify(variation_mode)[0] == "linear":
            variation_timescale = (variation_timescale[0] * 2,)
    variation_ref_time = utils._listify(variation_ref_time)
    variation_timescale = utils._listify(variation_timescale)
    variation_amp = utils._listify(variation_amp)
    variation_mode = utils._listify(variation_mode)
    variation_settings = (
        variation_mode,
        variation_amp,
        variation_ref_time,
        variation_timescale,
    )

    # Check that everything is the same length.
    Nmodes = len(variation_mode)
    if any(len(settings) != Nmodes for settings in variation_settings):
        raise ValueError(
            "At least one of the variation settings does not have the same "
            "number of entries as the number of variation modes specified."
        )

    # Now generate a multiplicative envelope to use for applying time variation.
    iterator = zip(
        variation_mode, variation_amp, variation_ref_time, variation_timescale
    )
    envelope = 1
    for mode, amp, ref_time, timescale in iterator:
        phases = ((times - ref_time) / timescale) % 1  # Map times to [0, 1)
        if mode == "linear":
            phases = (phases + 0.25) % 1  # Shift left a quarter period.
            # Map phases to [-1, 1].
            response = np.where(phases <= 0.5, 4 * phases - 1, 3 - 4 * phases)
            envelope *= 1 + amp * response
        elif mode == "sinusoidal":
            envelope *= 1 + amp * np.sin(2 * np.pi * phases)
        elif mode == "noiselike":
            rng = rng or np.random.default_rng()
            envelope *= rng.normal(1, amp, times.size)
        else:
            raise NotImplementedError(f"Variation mode {mode!r} not supported.")

    if parameter in ("amp", "phs"):
        envelope = np.outer(envelope, np.ones(gain_shape[-1]))
        if parameter == "phs":
            envelope = np.exp(1j * (envelope - 1))
        gains = {ant: np.atleast_2d(gain) * envelope for ant, gain in gains.items()}
    else:
        envelope = 2 * np.pi * np.outer(envelope - 1, freqs)
        gains = {
            ant: np.atleast_2d(gain) * np.exp(1j * delays[ant] * envelope)
            for ant, gain in gains.items()
        }

    return gains


# to minimize breaking changes
gen_gains = Bandpass()
gen_bandpass = gen_gains._gen_bandpass
gen_delay_phs = gen_gains._gen_delay_phase

gen_reflection_coefficient = Reflections.gen_reflection_coefficient
gen_reflection_gains = Reflections()

gen_whitenoise_xtalk = WhiteNoiseCrosstalk()
gen_cross_coupling_xtalk = CrossCouplingCrosstalk()
