"""Module defining analytic polynomial beams."""
import numpy as np
from pyuvsim import AnalyticBeam
from numpy.polynomial.chebyshev import chebval


class PolyBeam(AnalyticBeam):
    """
    Analytic, azimuthally-symmetric beam model based on Chebyshev polynomials.

    The frequency-dependence of the beam is implemented by scaling source zenith
    angles when the beam is interpolated, using a power law.

    See HERA memo
    http://reionization.org/wp-content/uploads/2013/03/HERA081_HERA_Primary_Beam_Chebyshev_Apr2020.pdf

    Parameters
    ----------
    beam_coeffs : array_like
        Co-efficients of the Chebyshev polynomial.
    spectral_index : float, optional
        Spectral index of the frequency-dependent power law scaling to
        apply to the width of the beam.
    ref_freq : float, optional
        Reference frequency for the beam width scaling power law, in Hz.
    """

    def __init__(self, beam_coeffs, spectral_index=0.0, ref_freq=1e8):
        self.ref_freq = ref_freq
        self.spectral_index = spectral_index
        self.data_normalization = "peak"
        self.freq_interp_kind = None
        self.beam_type = "efield"
        self.beam_coeffs = beam_coeffs

    def peak_normalize(self):
        """Normalize the beam to have peak of unity."""
        # Not required
        pass

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

        # Transformed zenith angle, also scaled with frequency
        x = 2.0 * np.sin(za_array[np.newaxis, ...] / fscale[:, np.newaxis]) - 1.0

        # Primary beam values from Chebyshev polynomial
        values = chebval(x, self.beam_coeffs)
        central_val = chebval(-1.0, self.beam_coeffs)
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


class PerturbedPolyBeam(PolyBeam):
    """
    A PolyBeam in which the shape of the beam has been modified.

    The perturbations can be applied to the mainlobe, sidelobes, or
    the entire beam.

    Mainlobe: A Gaussian of width FWHM is subtracted and then a new
    Gaussian with width `mainlobe_width` is added back in. This perturbs
    the width of the primary beam mainlobe, but leaves the sidelobes mostly
    unchanged.

    Sidelobes: The baseline primary beam model, PB, is moduled by a (sine)
    Fourier series at angles beyond some zenith angle.

    Entire beam: may be sheared, stretched, and rotated.

    Parameters
    ----------
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
    beam_coeffs : array_like
        Co-efficients of the baseline Chebyshev polynomial.
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
        perturb_coeffs=None,
        perturb_scale=0.1,
        mainlobe_width=0.3,
        mainlobe_scale=1.0,
        transition_width=0.05,
        xstretch=1.0,
        ystretch=1.0,
        rotation=0.0,
        **kwargs
    ):
        # Initialize base class
        super().__init__(**kwargs)

        # Set parameters
        if perturb_coeffs is not None:
            self.perturb_coeffs = np.array(perturb_coeffs)
            self.nmodes = self.perturb_coeffs.size
        else:
            self.perturb_coeffs = perturb_coeffs
            self.nmodes = 0
        self.perturb_scale = perturb_scale
        self.mainlobe_width = mainlobe_width
        self.mainlobe_scale = mainlobe_scale
        self.transition_width = transition_width
        self.xstretch, self.ystretch = xstretch, ystretch
        self.rotation = rotation

        # Sanity checks
        if perturb_scale >= 1.0:
            raise ValueError(
                "'perturb_scale' must be less than 1; otherwise "
                "the beam can go negative."
            )

    def interp(self, az_array, za_array, freq_array, reuse_spline=None):
        """Evaluate the primary beam, after shearing, stretching, or rotation."""
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
            theta_s = np.sqrt(Xs ** 2.0 + Ys ** 2.0)
            phi_s = np.arccos(Xs / theta_s)
            phi_s[Ys < 0.0] *= -1.0

            # Fix coordinates below the horizon of the unstretched beam
            theta_s[np.where(theta_s < 0.0)] = 0.5 * np.pi
            theta_s[np.where(theta_s >= np.pi / 2.0)] = 0.5 * np.pi

            # Update za_array and az_array
            az_array, za_array = phi_s, theta_s

        # Call interp() method on parent class
        interp_data, interp_basis_vector = super().interp(
            az_array, za_array, freq_array, reuse_spline
        )

        # Smooth step function
        step = 0.5 * (
            1.0 + np.tanh((za_array - self.mainlobe_width) / self.transition_width)
        )

        # Add sidelobe perturbations
        if self.nmodes > 0:
            # Build Fourier series
            p = np.zeros(za_array.size)
            f_fac = 2.0 * np.pi / (np.pi / 2.0)  # Fourier series with period pi/2
            for n in range(self.nmodes):
                p += self.perturb_coeffs[n] * np.sin(f_fac * n * za_array)
            p /= (np.max(p) - np.min(p)) / 2.0

            # Modulate primary beam by perturbation function
            interp_data *= 1.0 + step * p * self.perturb_scale

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
    """

    def __init__(self, beam_coeffs, spectral_index=0.0, ref_freq=1e8):
        self.ref_freq = ref_freq
        self.spectral_index = spectral_index
        self.data_normalization = "peak"
        self.freq_interp_kind = None
        self.beam_type = "efield"
        self.beam_coeffs = beam_coeffs

    def peak_normalize(self):
        """Normalize the beam to have peak of unity."""
        # Not required
        pass

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
            self.beam_coeffs,
            radial_coord * np.cos(axial_coord),
            radial_coord * np.sin(axial_coord),
        )
        central_val = self.zernike(self.beam_coeffs, 0.0, 0.0)
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

    def zernike(self, coeffs, x, y):
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

        # x2 = self.powl(x, 2)
        # y2 = self.powl(y, 2)
        x2, x3, x4, x5, x6, x7, x8, x9, x10 = (
            x ** 2.0,
            x ** 3.0,
            x ** 4.0,
            x ** 5.0,
            x ** 6.0,
            x ** 7.0,
            x ** 8.0,
            x ** 9.0,
            x ** 10.0,
        )
        y2, y3, y4, y5, y6, y7, y8, y9, y10 = (
            y ** 2.0,
            y ** 3.0,
            y ** 4.0,
            y ** 5.0,
            y ** 6.0,
            y ** 7.0,
            y ** 8.0,
            y ** 9.0,
            y ** 10.0,
        )

        # Setting the equations for the Zernike polynomials
        # r = np.sqrt(powl(x,2) + powl(y,2))
        Z1 = c[0] * 1  # m = 0    n = 0
        Z2 = c[1] * x  # m = -1   n = 1
        Z3 = c[2] * y  # m = 1    n = 1
        Z4 = c[3] * 2 * x * y  # m = -2   n = 2
        Z5 = c[4] * (2 * x2 + 2 * y2 - 1)  # m = 0  n = 2
        Z6 = c[5] * (-1 * x2 + y2)  # m = 2  n = 2
        Z7 = c[6] * (-1 * x3 + 3 * x * y2)  # m = -3     n = 3
        Z8 = c[7] * (-2 * x + 3 * (x3) + 3 * x * (y2))  # m = -1   n = 3
        Z9 = c[8] * (-2 * y + 3 * y3 + 3 * (x2) * y)  # m = 1    n = 3
        Z10 = c[9] * (y3 - 3 * (x2) * y)  # m = 3 n =3
        Z11 = c[10] * (-4 * (x3) * y + 4 * x * (y3))  # m = -4    n = 4
        Z12 = c[11] * (-6 * x * y + 8 * (x3) * y + 8 * x * (y3))  # m = -2   n = 4
        Z13 = c[12] * (
            1 - 6 * x2 - 6 * y2 + 6 * x4 + 12 * (x2) * (y2) + 6 * y4
        )  # m = 0  n = 4
        Z14 = c[13] * (3 * x2 - 3 * y2 - 4 * x4 + 4 * y4)  # m = 2    n = 4
        Z15 = c[14] * (x4 - 6 * (x2) * (y2) + y4)  # m = 4   n = 4
        Z16 = c[15] * (x5 - 10 * (x3) * y2 + 5 * x * (y4))  # m = -5   n = 5
        Z17 = c[16] * (
            4 * x3 - 12 * x * (y2) - 5 * x5 + 10 * (x3) * (y2) + 15 * x * y4
        )  # m =-3     n = 5
        Z18 = c[17] * (
            3 * x - 12 * x3 - 12 * x * (y2) + 10 * x5 + 20 * (x3) * (y2) + 10 * x * (y4)
        )  # m= -1  n = 5
        Z19 = c[18] * (
            3 * y - 12 * y3 - 12 * y * (x2) + 10 * y5 + 20 * (y3) * (x2) + 10 * y * (x4)
        )  # m = 1  n = 5
        Z20 = c[19] * (
            -4 * y3 + 12 * y * (x2) + 5 * y5 - 10 * (y3) * (x2) - 15 * y * x4
        )  # m = 3   n = 5
        Z21 = c[20] * (y5 - 10 * (y3) * x2 + 5 * y * (x4))  # m = 5 n = 5
        Z22 = c[21] * (6 * (x5) * y - 20 * (x3) * (y3) + 6 * x * (y5))  # m = -6 n = 6
        Z23 = c[22] * (
            20 * (x3) * y - 20 * x * (y3) - 24 * (x5) * y + 24 * x * (y5)
        )  # m = -4   n = 6
        Z24 = c[23] * (
            12 * x * y
            + 40 * (x3) * y
            - 40 * x * (y3)
            + 30 * (x5) * y
            + 60 * (x3) * (y3)
            - 30 * x * (y5)
        )  # m = -2   n = 6
        Z25 = c[24] * (
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
        )  # m = 0   n = 6
        Z26 = c[25] * (
            -6 * (x2)
            + 6 * (y2)
            + 20 * (x4)
            - 20 * (y4)
            - 15 * (x6)
            - 15 * (x4) * (y2)
            + 15 * (x2) * (y4)
            + 15 * (y6)
        )  # m = 2   n = 6
        Z27 = c[26] * (
            -5 * (x4)
            + 30 * (x2) * (y2)
            - 5 * (y4)
            + 6 * (x6)
            - 30 * (x4) * y2
            - 30 * (x2) * (y4)
            + 6 * (y6)
        )  # m = 4    n = 6
        Z28 = c[27] * (
            -1 * (x6) + 15 * (x4) * (y2) - 15 * (x2) * (y4) + y6
        )  # m = 6   n = 6
        Z29 = c[28] * (
            -1 * (x7) + 21 * (x5) * (y2) - 35 * (x3) * (y4) + 7 * x * (y6)
        )  # m = -7    n = 7
        Z30 = c[29] * (
            -6 * (x5)
            + 60 * (x3) * (y2)
            - 30 * x * (y4)
            + 7 * x7
            - 63 * (x5) * (y2)
            - 35 * (x3) * (y4)
            + 35 * x * (y6)
        )  # m = -5    n = 7
        Z31 = c[30] * (
            -10 * (x3)
            + 30 * x * (y2)
            + 30 * x5
            - 60 * (x3) * (y2)
            - 90 * x * (y4)
            - 21 * x7
            + 21 * (x5) * (y2)
            + 105 * (x3) * (y4)
            + 63 * x * (y6)
        )  # m =-3       n = 7
        Z32 = c[31] * (
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
        )  # m = -1  n = 7
        Z33 = c[32] * (
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
        )  # m = 1   n = 7
        Z34 = c[33] * (
            10 * (y3)
            - 30 * y * (x2)
            - 30 * y5
            + 60 * (y3) * (x2)
            + 90 * y * (x4)
            + 21 * y7
            - 21 * (y5) * (x2)
            - 105 * (y3) * (x4)
            - 63 * y * (x6)
        )  # m =3     n = 7
        Z35 = c[34] * (
            -6 * (y5)
            + 60 * (y3) * (x2)
            - 30 * y * (x4)
            + 7 * y7
            - 63 * (y5) * (x2)
            - 35 * (y3) * (x4)
            + 35 * y * (x6)
        )  # m = 5  n = 7
        Z36 = c[35] * (
            y7 - 21 * (y5) * (x2) + 35 * (y3) * (x4) - 7 * y * (x6)
        )  # m = 7  n = 7
        Z37 = c[36] * (
            -8 * (x7) * y + 56 * (x5) * (y3) - 56 * (x3) * (y5) + 8 * x * (y7)
        )  # m = -8  n = 8
        Z38 = c[37] * (
            -42 * (x5) * y
            + 140 * (x3) * (y3)
            - 42 * x * (y5)
            + 48 * (x7) * y
            - 112 * (x5) * (y3)
            - 112 * (x3) * (y5)
            + 48 * x * (y7)
        )  # m = -6  n = 8
        Z39 = c[38] * (
            -60 * (x3) * y
            + 60 * x * (y3)
            + 168 * (x5) * y
            - 168 * x * (y5)
            - 112 * (x7) * y
            - 112 * (x5) * (y3)
            + 112 * (x3) * (y5)
            + 112 * x * (y7)
        )  # m = -4   n = 8
        Z40 = c[39] * (
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
        )  # m = -2   n = 8
        Z41 = c[40] * (
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
        )  # m = 0    n = 8
        Z42 = c[41] * (
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
        )  # m = 2  n = 8
        Z43 = c[42] * (
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
        )  # m = 4     n = 8
        Z44 = c[43] * (
            7 * x6
            - 105 * (x4) * (y2)
            + 105 * (x2) * (y4)
            - 7 * y6
            - 8 * x8
            + 112 * (x6) * (y2)
            - 112 * (x2) * (y6)
            + 8 * y8
        )  # m = 6    n = 8
        Z45 = c[44] * (
            x8 - 28 * (x6) * (y2) + 70 * (x4) * (y4) - 28 * (x2) * (y6) + y8
        )  # m = 8     n = 9
        Z46 = c[45] * (
            x9 - 36 * (x7) * (y2) + 126 * (x5) * (y4) - 84 * (x3) * (y6) + 9 * x * (y8)
        )  # m = -9     n = 9
        Z47 = c[46] * (
            8 * x7
            - 168 * (x5) * (y2)
            + 280 * (x3) * (y4)
            - 56 * x * (y6)
            - 9 * x9
            + 180 * (x7) * (y2)
            - 126 * (x5) * (y4)
            - 252 * (x3) * (y6)
            + 63 * x * (y8)
        )  # m = -7    n = 9
        Z48 = c[47] * (
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
        )  # m = -5    n = 9
        Z49 = c[48] * (
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
        )  # m = -3  n = 9
        Z50 = c[49] * (
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
        )  # m = -1   n = 9
        Z51 = c[50] * (
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
        )  # m = -1   n = 9
        Z52 = c[51] * (
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
        )  # m = 3  n = 9
        Z53 = c[52] * (
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
        )  # m = 5     n = 9
        Z54 = c[53] * (
            -8 * y7
            + 168 * (y5) * (x2)
            - 280 * (y3) * (x4)
            + 56 * y * (x6)
            + 9 * y9
            - 180 * (y7) * (x2)
            + 126 * (y5) * (x4)
            - 252 * (y3) * (x6)
            - 63 * y * (x8)
        )  # m = 7     n = 9
        Z55 = c[54] * (
            y9 - 36 * (y7) * (x2) + 126 * (y5) * (x4) - 84 * (y3) * (x6) + 9 * y * (x8)
        )  # m = 9       n = 9
        Z56 = c[55] * (
            10 * (x9) * y
            - 120 * (x7) * (y3)
            + 252 * (x5) * (y5)
            - 120 * (x3) * (y7)
            + 10 * x * (y9)
        )  # m = -10   n = 10
        Z57 = c[56] * (
            72 * (x7) * y
            - 504 * (x5) * (y3)
            + 504 * (x3) * (y5)
            - 72 * x * (y7)
            - 80 * (x9) * y
            + 480 * (x7) * (y3)
            - 480 * (x3) * (y7)
            + 80 * x * (y9)
        )  # m = -8    n = 10
        Z58 = c[57] * (
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
        )  # m = -6   n = 10
        Z59 = c[58] * (
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
        )  # m = -4   n = 10
        Z60 = c[59] * (
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
        )  # m = -2   n = 10
        Z61 = c[60] * (
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
        )  # m = 0    n = 10
        Z62 = c[61] * (
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
        )  # m = 2  n = 10
        Z63 = c[62] * (
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
        )  # m = 4     n = 10
        Z64 = c[63] * (
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
        )  # m = 6    n = 10
        Z65 = c[64] * (
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
        )  # m = 8    n = 10
        Z66 = c[65] * (
            -1 * x10
            + 45 * (x8) * (y2)
            - 210 * (x6) * (y4)
            + 210 * (x4) * (y6)
            - 45 * (x2) * (y8)
            + y10
        )  # m = 10   n = 10

        ZW = (
            Z1
            + Z2
            + Z3
            + Z4
            + Z5
            + Z6
            + Z7
            + Z8
            + Z9
            + Z10
            + Z11
            + Z12
            + Z13
            + Z14
            + Z15
            + Z16
            + Z17
            + Z18
            + Z19
            + Z20
            + Z21
            + Z22
            + Z23
            + Z24
            + Z25
            + Z26
            + Z27
            + Z28
            + Z29
            + Z30
            + Z31
            + Z32
            + Z33
            + Z34
            + Z35
            + Z36
            + Z37
            + Z38
            + Z39
            + Z40
            + Z41
            + Z42
            + Z43
            + Z44
            + Z45
            + Z46
            + Z47
            + Z48
            + Z49
            + Z50
            + Z51
            + Z52
            + Z53
            + Z54
            + Z55
            + Z56
            + Z57
            + Z58
            + Z59
            + Z60
            + Z61
            + Z62
            + Z63
            + Z64
            + Z65
            + Z66
        )
        return ZW
