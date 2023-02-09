"""Module defining analytic polynomial beams."""
import numpy as np
from numpy.polynomial.chebyshev import chebval
from pyuvsim import AnalyticBeam

from . import utils


def stokes_matrix(pol_index):
    """
    Calculate Pauli matrices for pseudo-Stokes conversion.

    Source code adapted from `pyuvdata`.

    Derived from https://arxiv.org/pdf/1401.2095.pdf, the Pauli
    indices are reordered from the quantum mechanical
    convention to an order which gives the ordering of the pseudo-Stokes vector
    ['pI', 'pQ', 'pU, 'pV'].

    Parameters
    ----------
    pol_index : int
        Polarization index for which the Pauli matrix is generated, the index
        must lie between 0 and 3 ('pI': 0, 'pQ': 1, 'pU': 2, 'pV':3).

    Returns
    -------
    pauli_mat: array of float
        Pauli matrix for pol_index. Shape: (2, 2)
    """
    if pol_index == 0:
        pauli_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    elif pol_index == 1:
        pauli_mat = np.array([[1.0, 0.0], [0.0, -1.0]])
    elif pol_index == 2:
        pauli_mat = np.array([[0.0, 1.0], [1.0, 0.0]])
    elif pol_index == 3:
        pauli_mat = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    else:
        raise ValueError("'pol_index' must be an integer between 0 and 3")

    return pauli_mat


def construct_mueller(jones, pol_index1, pol_index2):
    """
    Generate Mueller components. Source code adapted from `pyuvdata`.

    Following https://arxiv.org/pdf/1802.04151.pdf. Using equation:

            Mij = Tr(J sigma_i J^* sigma_j)

    where sigma_i and sigma_j are Pauli matrices.

    Parameters
    ----------
    jones: array of float
        Jones matrices containing the electric field for the dipole arms
        or linear polarizations. Shape: (Npixels, 2, 2) for Healpix beams or
        (Naxes1 * Naxes2, 2, 2) otherwise.

    pol_index1: int
        Polarization index referring to the first index of Mij (i).

    pol_index2: int
        Polarization index referring to the second index of Mij (j).

    Returns
    -------
    mueller: array of float
        Mueller array containing the Mij values, shape: (Npixels,) for Healpix beams
        or (Naxes1 * Naxes2,) otherwise.
    """
    pauli_mat1 = stokes_matrix(pol_index1)
    pauli_mat2 = stokes_matrix(pol_index2)

    mueller = 0.5 * np.einsum(
        "...ab,...bc,...cd,...ad", pauli_mat1, jones, pauli_mat2, np.conj(jones)
    )
    mueller = np.abs(mueller)

    return mueller


def efield_to_pstokes(efield_beam, npix, Nfreqs):
    """
    Convert E-field to pseudo-stokes power. Source code adapted from `pyuvdata`.

    Following https://arxiv.org/pdf/1802.04151.pdf, using the equation:

            M_ij = Tr(sigma_i J sigma_j J^*)

    where sigma_i and sigma_j are Pauli matrices.

    Parameters
    ----------
    efield_beam: array_like, complex
        The E-field to convert to pStokes power beam.
        Must have shape (2, 1, 2, Nfreq, npix).

    npix: int
        The npix number of the HEALPix maps of the efield_beam.

    Nfreqs: int
        The number of frequencies of the efield_beam.

    Returns
    -------
    power_data: array_like, complex
        The pseudo-Stokes power beam computed from efield_beam.
        Shape (1, 1, 4, Nfreq, npix)

    """
    # construct jones matrix containing the electric field

    pol_strings = ["pI", "pQ", "pU", "pV"]
    power_data = np.zeros((1, 1, 4, Nfreqs, npix), dtype=np.complex128)

    for fq_i in range(Nfreqs):
        jones = np.zeros((npix, 2, 2), dtype=np.complex128)
        pol_strings = ["pI", "pQ", "pU", "pV"]
        jones[:, 0, 0] = efield_beam[0, 0, 0, fq_i, :]
        jones[:, 0, 1] = efield_beam[0, 0, 1, fq_i, :]
        jones[:, 1, 0] = efield_beam[1, 0, 0, fq_i, :]
        jones[:, 1, 1] = efield_beam[1, 0, 1, fq_i, :]

        for pol_i in range(len(pol_strings)):
            power_data[:, :, pol_i, fq_i, :] = construct_mueller(jones, pol_i, pol_i)

    return power_data


def modulate_with_dipole(az, za, freqs, ref_freq, beam_vals, fscale):
    """
    Take a beam pattern and modulate it  to turn it into an approximate E-field beam.

    This is achieved by taking the beam pattern (assumed to be the square-root of a
    power beam) and multiplying it by an zenith, azimuth and frequency -dependent
    complex dipole matrix (a polarised dipole pattern), with elements:

    ```
    dipole = q(za_s) * (1. + p(za_s) * 1.j) * [[-sin(az), cos(az)], [cos(az), sin(az)]]
    ```

    where q and p are functions defined elsewhere in this file, and za_s is the
    zenith angle streched by a power law.

    Parameters
    ----------
    az: array_like
        Array of azimuth values, in radians. 1-dimensional, of same size 'Naz'
        than za.

    za: array_like
        Array of zenith-angle values, in radians. 1-dimensional, of same size 'Nza'
        than az.

    freqs: array_like
        Array of frequencies at which the beam pattern has been computed. Size 'Nfreqs'.

    ref_freq: float
        The reference frequency for the beam width scaling power law.

    beam_vals: array_like, complex
        Array of beam values, with shape (Nfreqs, Naz). This will normally be the
        square-root of a power beam.

    Returns
    -------
    pol_efield_beam : array_like, complex
        Array of polarized beam values, with shape (2, 1, 2, Nfreqs, Naz), where 2 =
        (phi, theta) directions, 1 = number of spectral windows, and 2 = N_feed is the
        number of linearly-polarised feeds, assumed to be the 'n' and 'e' directions.
    """
    # Form the beam.
    # initial dipole matrix, shape (2, 2, az.size)
    dipole = np.array([[-np.sin(az), np.cos(az)], [np.cos(az), np.sin(az)]])
    # stretched zenith angle, shape (Nfreq, za.size)
    za_scale = za[np.newaxis, :] / fscale[:, np.newaxis]
    # phase component, shape za_scale.shape = (Nfreq, za.size)
    ph = q(za_scale) * (1.0 + p(za_scale) * 1.0j)
    # shape (2, 2, 1, az.size)
    dipole_mod = ph[np.newaxis, np.newaxis, ...] * dipole[:, :, np.newaxis, :]
    # shape (2, 1, 2, Nfreq, az.size)
    pol_efield_beam = (
        dipole_mod[:, np.newaxis, ...]
        * beam_vals[np.newaxis, np.newaxis, np.newaxis, ...]
    )

    # Correct it for frequency dependency.
    # extract modulus and phase of the beams
    modulus = np.abs(pol_efield_beam)
    phase = np.angle(pol_efield_beam)
    # assume linear shift of phase along frequency
    shift = -np.pi / 18e6 * (freqs[:, np.newaxis] - ref_freq)  # shape (Nfreq, 1)
    # shift the phase
    phase += shift[np.newaxis, np.newaxis, np.newaxis, :, :]
    # upscale the modulus
    modulus = np.power(modulus, 0.6)  # ad-hoc
    # map the phase to [-pi; +pi]
    phase = utils.wrap2pipi(phase)
    # reconstruct
    pol_efield_beam = modulus * np.exp(1j * phase)

    return pol_efield_beam


def p(za):
    """
    Models the general behavior of the phase of the 'Fagnoni beam', and its first ring.

    (the first ring is the θ < π/11 region)

    Parameters
    ----------
    za: array_like
        Array of zenith-angle values, in radians.

    Returns
    -------
    res: aray_like
        Array of same size than za, in radians.

    """
    # "manual" fit on the 100 MHz Fagnoni beam
    res = np.pi * np.sin(np.pi * za)  # general behavior (kind of...)
    res[np.where(za < np.pi / 11)] = 0  # first ring

    return res


def q(za):
    """
    Models the 'second ring' of the phase of the 'Fagnoni beam'.

    (the second ring is the π/6 < θ < π/11 region)

    Parameters
    ----------
    za: array_like
        Array of zenith-angle values, in radians.

    Returns
    -------
    res: aray_like
        Array of same size than za, in radians.

    """
    # "manual" fit on the 100MHz beam
    res = np.ones(za.shape, dtype=np.complex128)
    res[np.where(np.logical_and(np.pi / 6 > za, za > np.pi / 11))] = 1j

    return res


class PolyBeam(AnalyticBeam):
    """
    Analytic, azimuthally-symmetric beam model based on Chebyshev polynomials.

    The frequency-dependence of the beam is implemented by scaling source zenith
    angles when the beam is interpolated, using a power law.

    See HERA memo:
    http://reionization.org/wp-content/uploads/2013/03/HERA081_HERA_Primary_Beam_Chebyshev_Apr2020.pdf

    Parameters
    ----------
    beam_coeffs : array_like
        Co-efficients of the Chebyshev polynomial.
    spectral_index : float, optional
        Spectral index of the frequency-dependent power law scaling to
        apply to the width of the beam. Default: 0.0.
    ref_freq : float, optional
        Reference frequency for the beam width scaling power law, in Hz.
        Default: 1e8.
    polarized : bool, optional
        Whether to multiply the axisymmetric beam model by a dipole
        modulation factor to emulate a polarized beam response. If False,
        the axisymmetric representation will be put in the (phi, n)
        and (theta, e) elements of the Jones matrix returned by the
        `interp()` method. Default: False.
    """

    def __init__(
        self,
        beam_coeffs=None,
        spectral_index=0.0,
        ref_freq=1e8,
        polarized=False,
    ):
        self.ref_freq = ref_freq
        self.spectral_index = spectral_index
        self.polarized = polarized
        self.data_normalization = "peak"
        self.freq_interp_kind = None
        self.Nspws = 1

        # Polarization conventions
        self.beam_type = "efield"
        self.Nfeeds = 2  # n and e feeds
        self.pixel_coordinate_system = "az_za"  # az runs from East to North
        self.feed_array = ["n", "e"]
        self.x_orientation = "east"

        # Beam data
        self.beam_coeffs = beam_coeffs

    def peak_normalize(self):
        """Normalize the beam to have peak of unity."""
        # Not required
        pass

    def select(self, **kwargs):
        """Dummy select method."""
        pass

    def interp(self, az_array, za_array, freq_array, **kwargs):
        """
        Evaluate the primary beam at given az, za locations (in radians).

        Parameters
        ----------
        az_array : array_like
            Azimuth values in radians (same length as za_array). The azimuth
            here has the UVBeam convention: North of East(East=0, North=pi/2)
        za_array : array_like
            Zenith angle values in radians (same length as az_array).
        freq_array : array_like
            Frequency values to evaluate at.


        Other Parameters
        ----------------
        All other parameters are ignored -- they are there for compatibility with
        :class:`pyuvdata.uvbeam.UVBeam`.

        Returns
        -------
        interp_data : array_like
            Array of beam values, shape (Naxes_vec, Nspws, Nfeeds or Npols,
            Nfreqs or freq_array.size if freq_array is passed,
            Npixels/(Naxis1, Naxis2) or az_array.size if az/za_arrays are passed)
        interp_basis_vector : array_like
            Array of interpolated basis vectors (or self.basis_vector_array
            if az/za_arrays are not passed), shape: (Naxes_vec, Ncomponents_vec,
            Npixels/(Naxis1, Naxis2) or az_array.size if az/za_arrays are passed)
        """
        # Check that coordinates have same length
        if az_array.size != za_array.size:
            raise ValueError(
                "Azimuth and zenith angle coordinate arrays must have same length."
            )

        # Empty data array
        interp_data = np.zeros(
            (2, 1, 2, freq_array.size, az_array.size), dtype=np.complex128
        )

        # Frequency scaling
        fscale = (freq_array / self.ref_freq) ** self.spectral_index

        # Transformed zenith angle, also scaled with frequency
        x = 2.0 * np.sin(za_array[np.newaxis, ...] / fscale[:, np.newaxis]) - 1.0

        # Primary beam values from Chebyshev polynomial
        beam_values = chebval(x, self.beam_coeffs)
        central_val = chebval(-1.0, self.beam_coeffs)
        beam_values /= central_val  # ensure normalized to 1 at za=0

        # Set beam Jones matrix values (see Eq. 5 of Kohn+ arXiv:1802.04151)
        # Axes: [phi, theta] (az and za) / Feeds: [n, e]
        # interp_data shape: (Naxes_vec, Nspws, Nfeeds or Npols, Nfreqs, Naz)
        if self.polarized:
            interp_data = modulate_with_dipole(
                az_array, za_array, freq_array, self.ref_freq, beam_values, fscale
            )
        else:
            interp_data[1, 0, 0, :, :] = beam_values  # (theta, n)
            interp_data[0, 0, 1, :, :] = beam_values  # (phi, e)

        interp_basis_vector = None

        if self.beam_type == "power":
            # Cross-multiplying feeds, adding vector components
            pairs = [(i, j) for i in range(2) for j in range(2)]
            power_data = np.zeros((1, 1, 4) + beam_values.shape, dtype=float)
            for pol_i, pair in enumerate(pairs):
                power_data[:, :, pol_i] = (
                    interp_data[0, :, pair[0]] * np.conj(interp_data[0, :, pair[1]])
                ) + (interp_data[1, :, pair[0]] * np.conj(interp_data[1, :, pair[1]]))
            interp_data = power_data

        return interp_data, interp_basis_vector

    def __eq__(self, other):
        """Evaluate equality with another object."""
        if not isinstance(other, self.__class__):
            return False
        return self.beam_coeffs == other.beam_coeffs


class PerturbedPolyBeam(PolyBeam):
    """A :class:`PolyBeam` in which the shape of the beam has been modified.

    The perturbations can be applied to the mainlobe, sidelobes, or
    the entire beam. While the underlying :class:`PolyBeam` depends on
    frequency via the `spectral_index` kwarg, the perturbations themselves do
    not have a frequency dependence unless explicitly stated.

    Mainlobe: A Gaussian of width FWHM is subtracted and then a new
    Gaussian with width `mainlobe_width` is added back in. This perturbs
    the width of the primary beam mainlobe, but leaves the sidelobes mostly
    unchanged.

    Sidelobes: The baseline primary beam model, PB, is moduled by a (sine)
    Fourier series at angles beyond some zenith angle. There is an angle-
    only modulation (set by `perturb_coeffs` and `perturb_scale`), and a
    frequency-only modulation (set by `freq_perturb_coeffs` and
    `freq_perturb_scale`).

    Entire beam: May be sheared, stretched, and rotated.

    Parameters
    ----------
    beam_coeffs : array_like
        Co-efficients of the baseline Chebyshev polynomial.
    perturb_coeffs : array_like, optional
        Array of floats with the coefficients of a (sine-only) Fourier
        series that will be used to modulate the base Chebyshev primary
        beam model.
    perturb_scale : float, optional
        Overall scale of the primary beam modulation. Must be less than 1,
        otherwise the primary beam can go negative.
    mainlobe_width : float
        Width of the mainlobe, in radians. This determines the width of the
        Gaussian mainlobe model that is subtracted, as well as the location
        of the transition between the mainlobe and sidelobe regimes.
    mainlobe_scale : float, optional
        Factor to apply to the FHWM of the Gaussian that is used to rescale
        the mainlobe.
    transition_width : float, optional
        Width of the smooth transition between the range of angles
        considered to be in the mainlobe vs in the sidelobes, in radians.
    xstretch, ystretch : float, optional
        Stretching factors to apply to the beam in the x and y directions,
        which introduces beam ellipticity, as well as an overall
        stretching/shrinking. Default: 1.0 (no ellipticity or stretching).
    rotation : float, optional
        Rotation of the beam in the x-y plane, in degrees. Only has an
        effect if xstretch != ystretch.
    freq_perturb_coeffs : array_like, optional
        Array of floats with the coefficients of a sine and cosine Fourier
        series that will be used to modulate the base Chebyshev primary
        beam model in the frequency direction. Default: None.
    freq_perturb_scale : float, optional
        Overall scale of the primary beam modulation in the frequency
        direction. Must be less than 1, otherwise the primary beam can go
        negative. Default: 0.
    perturb_zeropoint : float, optional
        If specified, override the automatical zero-point calculation for
        the angle-dependent sidelobe perturbation. Default: None (use the
        automatically-calculated zero-point).
    spectral_index : float, optional
        Spectral index of the frequency-dependent power law scaling to
        apply to the width of the beam.
    ref_freq : float, optional
        Reference frequency for the beam width scaling power law, in Hz.
    **kwargs
        Any other parameters are used to initialize superclass :class:`PolyBeam`.
    """

    def __init__(
        self,
        beam_coeffs=None,
        perturb_coeffs=None,
        perturb_scale=0.1,
        mainlobe_width=0.3,
        mainlobe_scale=1.0,
        transition_width=0.05,
        xstretch=1.0,
        ystretch=1.0,
        rotation=0.0,
        freq_perturb_coeffs=None,
        freq_perturb_scale=0.0,
        perturb_zeropoint=None,
        **kwargs
    ):
        # Initialize base class
        super().__init__(beam_coeffs=beam_coeffs, **kwargs)

        # Check for valid input parameters
        if mainlobe_width is None:
            raise ValueError("Must specify a value for 'mainlobe_width' kwarg")

        # Set sidelobe perturbation parameters
        if perturb_coeffs is None:
            perturb_coeffs = []
        if freq_perturb_coeffs is None:
            freq_perturb_coeffs = []
        self.perturb_coeffs = np.array(perturb_coeffs)
        self.freq_perturb_coeffs = np.array(freq_perturb_coeffs)

        # Set all other parameters
        self.perturb_scale = perturb_scale
        self.freq_perturb_scale = freq_perturb_scale
        self.mainlobe_width = mainlobe_width
        self.mainlobe_scale = mainlobe_scale
        self.transition_width = transition_width
        self.xstretch, self.ystretch = xstretch, ystretch
        self.rotation = rotation

        # Calculate normalization of sidelobe perturbation functions on
        # fixed grid (ensures rescaling is deterministic/independent of input
        # to the interp() method)
        za = np.linspace(0.0, np.pi / 2.0, 1000)  # rad
        freqs = np.linspace(100.0, 200.0, 1000) * 1e6  # Hz
        p_za = self._sidelobe_modulation_za(za, scale=1.0, zeropoint=0.0)
        p_freq = self._sidelobe_modulation_freq(freqs, scale=1.0, zeropoint=0.0)

        # Rescale p_za to the range [-0.5, +0.5]
        self._scale_pza, self._zeropoint_pza = 0.0, 0.0
        if self.perturb_coeffs.size > 0:
            self._scale_pza = 2.0 / (np.max(p_za) - np.min(p_za))
            self._zeropoint_pza = -0.5 - 2.0 * np.min(p_za) / (
                np.max(p_za) - np.min(p_za)
            )

            # Override calculated zeropoint with user-specified value
            if perturb_zeropoint is not None:
                self._zeropoint_pza = perturb_zeropoint

        # Rescale p_freq to the range [-0.5, +0.5]
        self._scale_pfreq, self._zeropoint_pfreq = 0.0, 0.0
        if self.freq_perturb_coeffs.size > 0:
            self._scale_pfreq = 2.0 / (np.max(p_freq) - np.min(p_freq))
            self._zeropoint_pfreq = -0.5 - 2.0 * np.min(p_freq) / (
                np.max(p_freq) - np.min(p_freq)
            )

        # Sanity checks
        if self.perturb_scale >= 1.0:
            raise ValueError(
                "'perturb_scale' must be less than 1; otherwise "
                "the beam can go negative."
            )
        if self.freq_perturb_scale >= 1.0:
            raise ValueError(
                "'freq_perturb_scale' must be less than 1; "
                "otherwise the beam can go negative."
            )

    def _sidelobe_modulation_za(self, za_array, scale=1.0, zeropoint=0.0):
        """Calculate sidelobe modulation factor for a set of zenith angle values.

        Parameters
        ----------
        za_array : array_like
            Array of zenith angles, in radians.

        scale : float, optional
            Multiplicative rescaling factor to be applied to the modulation
            function. Default: 1.

        zeropoint : float, optional
            Zero-point correction to be applied to the modulation function.
            Default: 0.
        """
        # Construct sidelobe perturbations (angle-dependent)
        p_za = 0
        if self.perturb_coeffs.size > 0:
            # Build Fourier (sine) series
            f_fac = 2.0 * np.pi / (np.pi / 2.0)  # Fourier series with period pi/2
            for n in range(self.perturb_coeffs.size):
                p_za += self.perturb_coeffs[n] * np.sin(f_fac * n * za_array)

        return p_za * scale + zeropoint

    def _sidelobe_modulation_freq(self, freq_array, scale=1.0, zeropoint=0.0):
        """Calculate sidelobe modulation factor for a set of frequency values.

        Parameters
        ----------
        freq_array : array_like
            Array of frequencies, in Hz.

        scale : float, optional
            Multiplicative rescaling factor to be applied to the modulation
            function. Default: 1.

        zeropoint : float, optional
            Zero-point correction to be applied to the modulation function.
            Default: 0.
        """
        # Construct sidelobe perturbations (frequency-dependent)
        p_freq = 0
        if self.freq_perturb_coeffs.size > 0:
            # Build Fourier series (sine + cosine)
            f_fac = 2.0 * np.pi / (100.0e6)  # Fourier series with period 100 MHz
            for n in range(self.freq_perturb_coeffs.size):
                if n == 0:
                    fn = 1.0 + 0.0 * freq_array
                elif n % 2 == 0:
                    fn = np.sin(f_fac * ((n + 1) // 2) * freq_array)
                else:
                    fn = np.cos(f_fac * ((n + 1) // 2) * freq_array)
                p_freq += self.freq_perturb_coeffs[n] * fn

        return p_freq * scale + zeropoint

    def interp(self, az_array, za_array, freq_array, reuse_spline=None):
        """Evaluate the primary beam after shearing/stretching/rotation."""
        # Apply shearing, stretching, or rotation
        if self.xstretch != 1.0 or self.ystretch != 1.0:
            # Convert sheared Cartesian coords to circular polar coords
            # mX stretches in x direction, mY in y direction, a is angle
            # Notation: phi = az, theta = za. Subscript 's' are transformed coords
            a = self.rotation * np.pi / 180.0
            X = za_array * np.cos(az_array)
            Y = za_array * np.sin(az_array)
            Xs = (X * np.cos(a) - Y * np.sin(a)) / self.xstretch
            Ys = (X * np.sin(a) + Y * np.cos(a)) / self.ystretch

            # Updated polar coordinates
            theta_s = np.sqrt(Xs**2.0 + Ys**2.0)
            phi_s = np.zeros_like(theta_s)
            mask = theta_s == 0.0
            phi_s[~mask] = np.arccos(Xs[~mask] / theta_s[~mask])
            phi_s[Ys < 0.0] *= -1.0

            # Fix coordinates below the horizon of the unstretched beam
            theta_s[np.where(theta_s < 0.0)] = 0.5 * np.pi
            theta_s[np.where(theta_s >= np.pi / 2.0)] = 0.5 * np.pi

            # Update za_array and az_array
            az_array, za_array = phi_s, theta_s

        # Call interp() method on parent class
        interp_data, interp_basis_vector = super().interp(
            az_array=az_array,
            za_array=za_array,
            freq_array=freq_array,
            reuse_spline=reuse_spline,
        )

        # Smooth step function
        step = 0.5 * (
            1.0 + np.tanh((za_array - self.mainlobe_width) / self.transition_width)
        )

        # Construct sidelobe perturbations (angle- and frequency-dependent)
        p_za = self._sidelobe_modulation_za(
            za_array, scale=self._scale_pza, zeropoint=self._zeropoint_pza
        )
        p_freq = self._sidelobe_modulation_freq(
            freq_array, scale=self._scale_pfreq, zeropoint=self._zeropoint_pfreq
        )
        p_za = np.atleast_1d(self.perturb_scale * p_za)
        p_freq = np.atleast_1d(self.freq_perturb_scale * p_freq)

        # Modulate primary beam by sidelobe perturbation function
        interp_data *= 1.0 + (step * p_za)[np.newaxis, :] * (
            1.0 + p_freq[:, np.newaxis]
        )

        # Add mainlobe stretch factor
        if self.mainlobe_scale != 1.0:
            # Subtract and re-add Gaussian normalized to 1 at za = 0
            w = self.mainlobe_width / 2.0
            mainlobe0 = np.exp(-0.5 * (za_array / w) ** 2.0)
            mainlobe_pert = np.exp(-0.5 * (za_array / (w * self.mainlobe_scale)) ** 2.0)
            interp_data += (1.0 - step) * (mainlobe_pert - mainlobe0)

        return interp_data, interp_basis_vector


class ZernikeBeam(AnalyticBeam):
    """
    Analytic beam model based on Zernike polynomials.

    Parameters
    ----------
    beam_coeffs : array_like
        Co-efficients of the Chebyshev polynomial.
    spectral_index : float, optional
        Spectral index of the frequency-dependent power law scaling to
        apply to the width of the beam.
    ref_freq : float, optional
        Reference frequency for the beam width scaling power law, in Hz.
    peak_normalized : bool, optional
        Whether the beam should be normalized to 1 at beam center.
    """

    def __init__(
        self, beam_coeffs, spectral_index=0.0, ref_freq=1e8, peak_normalized=True
    ):
        self.ref_freq = ref_freq
        self.spectral_index = spectral_index
        self.data_normalization = "peak"
        self.peak_normalized = peak_normalized
        self.freq_interp_kind = None
        self.beam_type = "efield"
        self.beam_coeffs = beam_coeffs

    def peak_normalize(self):
        """Normalize the beam to have peak of unity."""
        self.peak_normalized = True

    def interp(self, az_array, za_array, freq_array, reuse_spline=None):
        """
        Evaluate the primary beam at given az, za locations (in radians).

        Parameters
        ----------
        az_array : array_like
            Azimuth values in radians (same length as za_array). The azimuth
            here has the UVBeam convention: North of East(East=0, North=pi/2)
        za_array : array_like
            Zenith angle values in radians (same length as az_array).
        freq_array : array_like
            Frequency values to evaluate at.
        reuse_spline : bool, optional
            Does nothing for analytic beams. Here for compatibility with UVBeam.

        Returns
        -------
        interp_data : array_like
            Array of beam values, shape (Naxes_vec, Nspws, Nfeeds or Npols,
            Nfreqs or freq_array.size if freq_array is passed,
            Npixels/(Naxis1, Naxis2) or az_array.size if az/za_arrays are passed)
        interp_basis_vector : array_like
            Array of interpolated basis vectors (or self.basis_vector_array
            if az/za_arrays are not passed), shape: (Naxes_vec, Ncomponents_vec,
            Npixels/(Naxis1, Naxis2) or az_array.size if az/za_arrays are passed)
        """
        # Empty data array
        interp_data = np.zeros((2, 1, 2, freq_array.size, az_array.size), dtype=float)

        # Frequency scaling
        fscale = (freq_array / self.ref_freq) ** self.spectral_index
        radial_coord = za_array[np.newaxis, ...] / fscale[:, np.newaxis]
        axial_coord = az_array[np.newaxis, ...]

        # Primary beam values from Zernike polynomial
        values = self.zernike(
            coeffs=self.beam_coeffs,
            x=radial_coord * np.cos(axial_coord),
            y=radial_coord * np.sin(axial_coord),
        )
        central_val = self.zernike(coeffs=self.beam_coeffs, x=0.0, y=0.0)
        if self.peak_normalized:
            values /= central_val  # ensure normalized to 1 at za=0

        # Set values
        interp_data[1, 0, 0, :, :] = values
        interp_data[0, 0, 1, :, :] = values
        interp_basis_vector = None

        if self.beam_type == "power":
            # Cross-multiplying feeds, adding vector components
            pairs = [(i, j) for i in range(2) for j in range(2)]
            power_data = np.zeros((1, 1, 4) + values.shape, dtype=float)
            for pol_i, pair in enumerate(pairs):
                power_data[:, :, pol_i] = (
                    interp_data[0, :, pair[0]] * np.conj(interp_data[0, :, pair[1]])
                ) + (interp_data[1, :, pair[0]] * np.conj(interp_data[1, :, pair[1]]))
            interp_data = power_data

        return interp_data, interp_basis_vector

    def __eq__(self, other):
        """Evaluate equality with another object."""
        if not isinstance(other, self.__class__):
            return False
        return self.beam_coeffs == other.beam_coeffs

    @staticmethod
    def zernike(coeffs, x, y):
        """
        Zernike polynomials (up to degree 66) on the unit disc.

        This code was adapted from:
        https://gitlab.nrao.edu/pjaganna/zcpb/-/blob/master/zernikeAperture.py

        Parameters
        ----------
        coeffs : array_like
            Array of real coefficients of the Zernike polynomials, from 0..66.

        x, y : array_like
            Points on the unit disc.

        Returns
        -------
        zernike : array_like
            Values of the Zernike polynomial at the input x,y points.
        """
        # Coefficients
        assert len(coeffs) <= 66, "Max. number of coeffs is 66."
        c = np.zeros(66)
        c[: len(coeffs)] += coeffs

        # Precompute powers of x and y
        x2, x3, x4, x5, x6, x7, x8, x9, x10 = tuple(x**idx for idx in range(2, 11))
        y2, y3, y4, y5, y6, y7, y8, y9, y10 = tuple(y**idx for idx in range(2, 11))

        # Setting the equations for the Zernike polynomials
        # r = np.sqrt(powl(x,2) + powl(y,2))
        Z = {
            1: c[0] * 1,  # m = 0    n = 0
            2: c[1] * x,  # m = -1   n = 1
            3: c[2] * y,  # m = 1    n = 1
            4: c[3] * 2 * x * y,  # m = -2   n = 2
            5: c[4] * (2 * x2 + 2 * y2 - 1),  # m = 0  n = 2
            6: c[5] * (-1 * x2 + y2),  # m = 2  n = 2
            7: c[6] * (-1 * x3 + 3 * x * y2),  # m = -3     n = 3
            8: c[7] * (-2 * x + 3 * (x3) + 3 * x * (y2)),  # m = -1   n = 3
            9: c[8] * (-2 * y + 3 * y3 + 3 * (x2) * y),  # m = 1    n = 3
            10: c[9] * (y3 - 3 * (x2) * y),  # m = 3 n =3
            11: c[10] * (-4 * (x3) * y + 4 * x * (y3)),  # m = -4    n = 4
            12: c[11] * (-6 * x * y + 8 * (x3) * y + 8 * x * (y3)),  # m = -2   n = 4
            13: c[12]
            * (
                1 - 6 * x2 - 6 * y2 + 6 * x4 + 12 * (x2) * (y2) + 6 * y4
            ),  # m = 0  n = 4
            14: c[13] * (3 * x2 - 3 * y2 - 4 * x4 + 4 * y4),  # m = 2    n = 4
            15: c[14] * (x4 - 6 * (x2) * (y2) + y4),  # m = 4   n = 4
            16: c[15] * (x5 - 10 * (x3) * y2 + 5 * x * (y4)),  # m = -5   n = 5
            17: c[16]
            * (
                4 * x3 - 12 * x * (y2) - 5 * x5 + 10 * (x3) * (y2) + 15 * x * y4
            ),  # m =-3     n = 5
            18: c[17]
            * (
                3 * x
                - 12 * x3
                - 12 * x * (y2)
                + 10 * x5
                + 20 * (x3) * (y2)
                + 10 * x * (y4)
            ),  # m= -1  n = 5
            19: c[18]
            * (
                3 * y
                - 12 * y3
                - 12 * y * (x2)
                + 10 * y5
                + 20 * (y3) * (x2)
                + 10 * y * (x4)
            ),  # m = 1  n = 5
            20: c[19]
            * (
                -4 * y3 + 12 * y * (x2) + 5 * y5 - 10 * (y3) * (x2) - 15 * y * x4
            ),  # m = 3   n = 5
            21: c[20] * (y5 - 10 * (y3) * x2 + 5 * y * (x4)),  # m = 5 n = 5
            22: c[21]
            * (6 * (x5) * y - 20 * (x3) * (y3) + 6 * x * (y5)),  # m = -6 n = 6
            23: c[22]
            * (
                20 * (x3) * y - 20 * x * (y3) - 24 * (x5) * y + 24 * x * (y5)
            ),  # m = -4   n = 6
            24: c[23]
            * (
                12 * x * y
                + 40 * (x3) * y
                - 40 * x * (y3)
                + 30 * (x5) * y
                + 60 * (x3) * (y3)
                - 30 * x * (y5)
            ),  # m = -2   n = 6
            25: c[24]
            * (
                -1
                + 12 * (x2)
                + 12 * (y2)
                - 30 * (x4)
                - 60 * (x2) * (y2)
                - 30 * (y4)
                + 20 * (x6)
                + 60 * (x4) * y2
                + 60 * (x2) * (y4)
                + 20 * (y6)
            ),  # m = 0   n = 6
            26: c[25]
            * (
                -6 * (x2)
                + 6 * (y2)
                + 20 * (x4)
                - 20 * (y4)
                - 15 * (x6)
                - 15 * (x4) * (y2)
                + 15 * (x2) * (y4)
                + 15 * (y6)
            ),  # m = 2   n = 6
            27: c[26]
            * (
                -5 * (x4)
                + 30 * (x2) * (y2)
                - 5 * (y4)
                + 6 * (x6)
                - 30 * (x4) * y2
                - 30 * (x2) * (y4)
                + 6 * (y6)
            ),  # m = 4    n = 6
            28: c[27]
            * (-1 * (x6) + 15 * (x4) * (y2) - 15 * (x2) * (y4) + y6),  # m = 6   n = 6
            29: c[28]
            * (
                -1 * (x7) + 21 * (x5) * (y2) - 35 * (x3) * (y4) + 7 * x * (y6)
            ),  # m = -7    n = 7
            30: c[29]
            * (
                -6 * (x5)
                + 60 * (x3) * (y2)
                - 30 * x * (y4)
                + 7 * x7
                - 63 * (x5) * (y2)
                - 35 * (x3) * (y4)
                + 35 * x * (y6)
            ),  # m = -5    n = 7
            31: c[30]
            * (
                -10 * (x3)
                + 30 * x * (y2)
                + 30 * x5
                - 60 * (x3) * (y2)
                - 90 * x * (y4)
                - 21 * x7
                + 21 * (x5) * (y2)
                + 105 * (x3) * (y4)
                + 63 * x * (y6)
            ),  # m =-3       n = 7
            32: c[31]
            * (
                -4 * x
                + 30 * x3
                + 30 * x * (y2)
                - 60 * (x5)
                - 120 * (x3) * (y2)
                - 60 * x * (y4)
                + 35 * x7
                + 105 * (x5) * (y2)
                + 105 * (x3) * (y4)
                + 35 * x * (y6)
            ),  # m = -1  n = 7
            33: c[32]
            * (
                -4 * y
                + 30 * y3
                + 30 * y * (x2)
                - 60 * (y5)
                - 120 * (y3) * (x2)
                - 60 * y * (x4)
                + 35 * y7
                + 105 * (y5) * (x2)
                + 105 * (y3) * (x4)
                + 35 * y * (x6)
            ),  # m = 1   n = 7
            34: c[33]
            * (
                10 * (y3)
                - 30 * y * (x2)
                - 30 * y5
                + 60 * (y3) * (x2)
                + 90 * y * (x4)
                + 21 * y7
                - 21 * (y5) * (x2)
                - 105 * (y3) * (x4)
                - 63 * y * (x6)
            ),  # m =3     n = 7
            35: c[34]
            * (
                -6 * (y5)
                + 60 * (y3) * (x2)
                - 30 * y * (x4)
                + 7 * y7
                - 63 * (y5) * (x2)
                - 35 * (y3) * (x4)
                + 35 * y * (x6)
            ),  # m = 5  n = 7
            36: c[35]
            * (y7 - 21 * (y5) * (x2) + 35 * (y3) * (x4) - 7 * y * (x6)),  # m = 7  n = 7
            37: c[36]
            * (
                -8 * (x7) * y + 56 * (x5) * (y3) - 56 * (x3) * (y5) + 8 * x * (y7)
            ),  # m = -8  n = 8
            38: c[37]
            * (
                -42 * (x5) * y
                + 140 * (x3) * (y3)
                - 42 * x * (y5)
                + 48 * (x7) * y
                - 112 * (x5) * (y3)
                - 112 * (x3) * (y5)
                + 48 * x * (y7)
            ),  # m = -6  n = 8
            39: c[38]
            * (
                -60 * (x3) * y
                + 60 * x * (y3)
                + 168 * (x5) * y
                - 168 * x * (y5)
                - 112 * (x7) * y
                - 112 * (x5) * (y3)
                + 112 * (x3) * (y5)
                + 112 * x * (y7)
            ),  # m = -4   n = 8
            40: c[39]
            * (
                -20 * x * y
                + 120 * (x3) * y
                + 120 * x * (y3)
                - 210 * (x5) * y
                - 420 * (x3) * (y3)
                - 210 * x * (y5)
                - 112 * (x7) * y
                + 336 * (x5) * (y3)
                + 336 * (x3) * (y5)
                + 112 * x * (y7)
            ),  # m = -2   n = 8
            41: c[40]
            * (
                1
                - 20 * x2
                - 20 * y2
                + 90 * x4
                + 180 * (x2) * (y2)
                + 90 * y4
                - 140 * x6
                - 420 * (x4) * (y2)
                - 420 * (x2) * (y4)
                - 140 * (y6)
                + 70 * x8
                + 280 * (x6) * (y2)
                + 420 * (x4) * (y4)
                + 280 * (x2) * (y6)
                + 70 * y8
            ),  # m = 0    n = 8
            42: c[41]
            * (
                10 * x2
                - 10 * y2
                - 60 * x4
                + 105 * (x4) * (y2)
                - 105 * (x2) * (y4)
                + 60 * y4
                + 105 * x6
                - 105 * y6
                - 56 * x8
                - 112 * (x6) * (y2)
                + 112 * (x2) * (y6)
                + 56 * y8
            ),  # m = 2  n = 8
            43: c[42]
            * (
                15 * x4
                - 90 * (x2) * (y2)
                + 15 * y4
                - 42 * x6
                + 210 * (x4) * (y2)
                + 210 * (x2) * (y4)
                - 42 * y6
                + 28 * x8
                - 112 * (x6) * (y2)
                - 280 * (x4) * (y4)
                - 112 * (x2) * (y6)
                + 28 * y8
            ),  # m = 4     n = 8
            44: c[43]
            * (
                7 * x6
                - 105 * (x4) * (y2)
                + 105 * (x2) * (y4)
                - 7 * y6
                - 8 * x8
                + 112 * (x6) * (y2)
                - 112 * (x2) * (y6)
                + 8 * y8
            ),  # m = 6    n = 8
            45: c[44]
            * (
                x8 - 28 * (x6) * (y2) + 70 * (x4) * (y4) - 28 * (x2) * (y6) + y8
            ),  # m = 8     n = 9
            46: c[45]
            * (
                x9
                - 36 * (x7) * (y2)
                + 126 * (x5) * (y4)
                - 84 * (x3) * (y6)
                + 9 * x * (y8)
            ),  # m = -9     n = 9
            47: c[46]
            * (
                8 * x7
                - 168 * (x5) * (y2)
                + 280 * (x3) * (y4)
                - 56 * x * (y6)
                - 9 * x9
                + 180 * (x7) * (y2)
                - 126 * (x5) * (y4)
                - 252 * (x3) * (y6)
                + 63 * x * (y8)
            ),  # m = -7    n = 9
            48: c[47]
            * (
                21 * x5
                - 210 * (x3) * (y2)
                + 105 * x * (y4)
                - 56 * x7
                + 504 * (x5) * (y2)
                + 280 * (x3) * (y4)
                - 280 * x * (y6)
                + 36 * x9
                - 288 * (x7) * (y2)
                - 504 * (x5) * (y4)
                + 180 * x * (y8)
            ),  # m = -5    n = 9
            49: c[48]
            * (
                20 * x3
                - 60 * x * (y2)
                - 105 * x5
                + 210 * (x3) * (y2)
                + 315 * x * (y4)
                + 168 * x7
                - 168 * (x5) * (y2)
                - 840 * (x3) * (y4)
                - 504 * x * (y6)
                - 84 * x9
                + 504 * (x5) * (y4)
                + 672 * (x3) * (y6)
                + 252 * x * (y8)
            ),  # m = -3  n = 9
            50: c[49]
            * (
                5 * x
                - 60 * x3
                - 60 * x * (y2)
                + 210 * x5
                + 420 * (x3) * (y2)
                + 210 * x * (y4)
                - 280 * x7
                - 840 * (x5) * (y2)
                - 840 * (x3) * (y4)
                - 280 * x * (y6)
                + 126 * x9
                + 504 * (x7) * (y2)
                + 756 * (x5) * (y4)
                + 504 * (x3) * (y6)
                + 126 * x * (y8)
            ),  # m = -1   n = 9
            51: c[50]
            * (
                5 * y
                - 60 * y3
                - 60 * y * (x2)
                + 210 * y5
                + 420 * (y3) * (x2)
                + 210 * y * (x4)
                - 280 * y7
                - 840 * (y5) * (x2)
                - 840 * (y3) * (x4)
                - 280 * y * (x6)
                + 126 * y9
                + 504 * (y7) * (x2)
                + 756 * (y5) * (x4)
                + 504 * (y3) * (x6)
                + 126 * y * (x8)
            ),  # m = -1   n = 9
            52: c[51]
            * (
                -20 * y3
                + 60 * y * (x2)
                + 105 * y5
                - 210 * (y3) * (x2)
                - 315 * y * (x4)
                - 168 * y7
                + 168 * (y5) * (x2)
                + 840 * (y3) * (x4)
                + 504 * y * (x6)
                + 84 * y9
                - 504 * (y5) * (x4)
                - 672 * (y3) * (x6)
                - 252 * y * (x8)
            ),  # m = 3  n = 9
            53: c[52]
            * (
                21 * y5
                - 210 * (y3) * (x2)
                + 105 * y * (x4)
                - 56 * y7
                + 504 * (y5) * (x2)
                + 280 * (y3) * (x4)
                - 280 * y * (x6)
                + 36 * y9
                - 288 * (y7) * (x2)
                - 504 * (y5) * (x4)
                + 180 * y * (x8)
            ),  # m = 5     n = 9
            54: c[53]
            * (
                -8 * y7
                + 168 * (y5) * (x2)
                - 280 * (y3) * (x4)
                + 56 * y * (x6)
                + 9 * y9
                - 180 * (y7) * (x2)
                + 126 * (y5) * (x4)
                - 252 * (y3) * (x6)
                - 63 * y * (x8)
            ),  # m = 7     n = 9
            55: c[54]
            * (
                y9
                - 36 * (y7) * (x2)
                + 126 * (y5) * (x4)
                - 84 * (y3) * (x6)
                + 9 * y * (x8)
            ),  # m = 9       n = 9
            56: c[55]
            * (
                10 * (x9) * y
                - 120 * (x7) * (y3)
                + 252 * (x5) * (y5)
                - 120 * (x3) * (y7)
                + 10 * x * (y9)
            ),  # m = -10   n = 10
            57: c[56]
            * (
                72 * (x7) * y
                - 504 * (x5) * (y3)
                + 504 * (x3) * (y5)
                - 72 * x * (y7)
                - 80 * (x9) * y
                + 480 * (x7) * (y3)
                - 480 * (x3) * (y7)
                + 80 * x * (y9)
            ),  # m = -8    n = 10
            58: c[57]
            * (
                270 * (x9) * y
                - 360 * (x7) * (y3)
                - 1260 * (x5) * (y5)
                - 360 * (x3) * (y7)
                + 270 * x * (y9)
                - 432 * (x7) * y
                + 1008 * (x5) * (y3)
                + 1008 * (x3) * (y5)
                - 432 * x * (y7)
                + 168 * (x5) * y
                - 560 * (x3) * (y3)
                + 168 * x * (y5)
            ),  # m = -6   n = 10
            59: c[58]
            * (
                140 * (x3) * y
                - 140 * x * (y3)
                - 672 * (x5) * y
                + 672 * x * (y5)
                + 1008 * (x7) * y
                + 1008 * (x5) * (y3)
                - 1008 * (x3) * (y5)
                - 1008 * x * (y7)
                - 480 * (x9) * y
                - 960 * (x7) * (y3)
                + 960 * (x3) * (y7)
                + 480 * x * (y9)
            ),  # m = -4   n = 10
            60: c[59]
            * (
                30 * x * y
                - 280 * (x3) * y
                - 280 * x * (y3)
                + 840 * (x5) * y
                + 1680 * (x3) * (y3)
                + 840 * x * (y5)
                - 1008 * (x7) * y
                - 3024 * (x5) * (y3)
                - 3024 * (x3) * (y5)
                - 1008 * x * (y7)
                + 420 * (x9) * y
                + 1680 * (x7) * (y3)
                + 2520 * (x5) * (y5)
                + 1680 * (x3) * (y7)
                + 420 * x * (y9)
            ),  # m = -2   n = 10
            61: c[60]
            * (
                -1
                + 30 * x2
                + 30 * y2
                - 210 * x4
                - 420 * (x2) * (y2)
                - 210 * y4
                + 560 * x6
                + 1680 * (x4) * (y2)
                + 1680 * (x2) * (y4)
                + 560 * y6
                - 630 * x8
                - 2520 * (x6) * (y2)
                - 3780 * (x4) * (y4)
                - 2520 * (x2) * (y6)
                - 630 * y8
                + 252 * x10
                + 1260 * (x8) * (y2)
                + 2520 * (x6) * (y4)
                + 2520 * (x4) * (y6)
                + 1260 * (x2) * (y8)
                + 252 * y10
            ),  # m = 0    n = 10
            62: c[61]
            * (
                -15 * x2
                + 15 * y2
                + 140 * x4
                - 140 * y4
                - 420 * x6
                - 420 * (x4) * (y2)
                + 420 * (x2) * (y4)
                + 420 * y6
                + 504 * x8
                + 1008 * (x6) * (y2)
                - 1008 * (x2) * (y6)
                - 504 * y8
                - 210 * x10
                - 630 * (x8) * (y2)
                - 420 * (x6) * (y4)
                + 420 * (x4) * (y6)
                + 630 * (x2) * (y8)
                + 210 * y10
            ),  # m = 2  n = 10
            63: c[62]
            * (
                -35 * x4
                + 210 * (x2) * (y2)
                - 35 * y4
                + 168 * x6
                - 840 * (x4) * (y2)
                - 840 * (x2) * (y4)
                + 168 * y6
                - 252 * x8
                + 1008 * (x6) * (y2)
                + 2520 * (x4) * (y4)
                + 1008 * (x2) * (y6)
                - 252 * (y8)
                + 120 * x10
                - 360 * (x8) * (y2)
                - 1680 * (x6) * (y4)
                - 1680 * (x4) * (y6)
                - 360 * (x2) * (y8)
                + 120 * y10
            ),  # m = 4     n = 10
            64: c[63]
            * (
                -28 * x6
                + 420 * (x4) * (y2)
                - 420 * (x2) * (y4)
                + 28 * y6
                + 72 * x8
                - 1008 * (x6) * (y2)
                + 1008 * (x2) * (y6)
                - 72 * y8
                - 45 * x10
                + 585 * (x8) * (y2)
                + 630 * (x6) * (y4)
                - 630 * (x4) * (y6)
                - 585 * (x2) * (y8)
                + 45 * y10
            ),  # m = 6    n = 10
            65: c[64]
            * (
                -9 * x8
                + 252 * (x6) * (y2)
                - 630 * (x4) * (y4)
                + 252 * (x2) * (y6)
                - 9 * y8
                + 10 * x10
                - 270 * (x8) * (y2)
                + 420 * (x6) * (y4)
                + 420 * (x4) * (y6)
                - 270 * (x2) * (y8)
                + 10 * y10
            ),  # m = 8    n = 10
            66: c[65]
            * (
                -1 * x10
                + 45 * (x8) * (y2)
                - 210 * (x6) * (y4)
                + 210 * (x4) * (y6)
                - 45 * (x2) * (y8)
                + y10
            ),  # m = 10   n = 10
        }
        return sum(Z.values())
