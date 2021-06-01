import numpy as np
from pyuvsim import AnalyticBeam
from numpy.polynomial.chebyshev import chebval, chebfit
from scipy.optimize import curve_fit


class PolyBeam(AnalyticBeam):
    def __init__(self, beam_coeffs=[], spectral_index=0.0, ref_freq=1e8, **kwargs):
        """
        Analytic, azimuthally-symmetric beam model based on Chebyshev polynomials.

        The frequency-dependence of the beam is implemented by scaling source zenith
        angles when the beam is interpolated, using a power law.

        See HERA memo
        http://reionization.org/wp-content/uploads/2013/03/Power_Spectrum_Normalizations_for_HERA.pdf.

        Parameters
        ----------
        beam_coeffs: array_like
            Co-efficients of the Chebyshev polynomial.

        spectral_index : float, optional
            Spectral index of the frequency-dependent power law scaling to
            apply to the width of the beam. Default: 0.0.

        ref_freq : float, optional
            Reference frequency for the beam width scaling power law, in Hz.
            Default: 1e8.
        """
        self.ref_freq = ref_freq
        self.spectral_index = spectral_index
        self.data_normalization = "peak"
        self.freq_interp_kind = None
        self.beam_type = "efield"
        self.beam_coeffs = beam_coeffs

    def peak_normalize(self):
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
        interp_data = np.zeros(
            (2, 1, 2, freq_array.size, az_array.size), dtype=np.float
        )

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
            power_data = np.zeros((1, 1, 4) + values.shape, dtype=np.float)
            for pol_i, pair in enumerate(pairs):
                power_data[:, :, pol_i] = (
                    interp_data[0, :, pair[0]] * np.conj(interp_data[0, :, pair[1]])
                ) + (interp_data[1, :, pair[0]] * np.conj(interp_data[1, :, pair[1]]))
            interp_data = power_data

        return interp_data, interp_basis_vector

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.beam_coeffs == other.beam_coeffs:
            return True
        else:
            return False


class PerturbedPolyBeam(PolyBeam):
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
            beam model. Default: None.

        perturb_scale : float, optional
            Overall scale of the primary beam modulation. Must be less than 1,
            otherwise the primary beam can go negative. Default: 0.1.

        mainlobe_width : float
            Width of the mainlobe, in radians. This determines the width of the
            Gaussian mainlobe model that is subtracted, as well as the location
            of the transition between the mainlobe and sidelobe regimes.
            Default: 0.3.

        mainlobe_scale : float, optional
            Factor to apply to the FHWM of the Gaussian that is used to rescale
            the mainlobe. Default: 1.

        transition_width : float, optional
            Width of the smooth transition between the range of angles
            considered to be in the mainlobe vs in the sidelobes, in radians.
            Default: 0.05.

        xstretch, ystretch : float, optional
            Stretching factors to apply to the beam in the x and y directions,
            which introduces beam ellipticity, as well as an overall
            stretching/shrinking. Default: 1.0 (no ellipticity or stretching).

        rotation : float, optional
            Rotation of the beam in the x-y plane, in degrees. Only has an
            effect if xstretch != ystretch. Default: 0.0.

        beam_coeffs: array_like
            Co-efficients of the baseline Chebyshev polynomial.

        spectral_index : float, optional
            Spectral index of the frequency-dependent power law scaling to
            apply to the width of the beam. Default: 0.0.

        ref_freq : float, optional
            Reference frequency for the beam width scaling power law, in Hz.
            Default: None.

        Other Parameters:
            Any other parameters are used to initialize superclass PolyBeam.
        """
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
        """
        Evaluate the primary beam, after applying, shearing, stretching, or rotation.
        """

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
