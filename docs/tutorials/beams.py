import numpy as np
from pyuvsim import AnalyticBeam
from numpy.polynomial.chebyshev import chebval

class PolyBeam(AnalyticBeam):
    """
    Defines an object with similar functionality to pyuvdata.UVBeam

    Anlytic HERA beam modelling using Chebyshev polynomial

    Args:
        beam_coeffs: (array_like, float)
                  co-effcients of the Chebyshev polynomial, 

        spectral_index : (float, optional)
            Scale gaussian beam width as a power law with frequency.

        ref_freq : (float, optional)
            If set, this sets the reference frequency for the beam width power law.
    """

    def __init__(self, beam_coeffs=[], spectral_index=0.0, ref_freq=None):
        if (spectral_index != 0.0) and (ref_freq is None):
            raise ValueError("ref_freq must be set for nonzero gaussian beam spectral index")
        elif ref_freq is None:
            ref_freq = 1.0
        self.ref_freq = ref_freq
        self.spectral_index = spectral_index
        self.data_normalization = 'peak'
        self.freq_interp_kind = None
        self.beam_type = 'efield'
        self.beam_coeffs = beam_coeffs

    def peak_normalize(self):
        pass
        
    def interp(self, az_array, za_array, freq_array, reuse_spline=None):
        """
        Evaluate the primary beam at given az, za locations (in radians).
        
        (similar to UVBeam.interp)

        Args:
            az_array: az values to evaluate at in radians (same length as za_array)
                The azimuth here has the UVBeam convention: North of East(East=0, North=pi/2)
            za_array: za values to evaluate at in radians (same length as az_array)
            freq_array: frequency values to evaluate at
            reuse_spline: Does nothing for analytic beams. Here for compatibility with UVBeam.

        Returns:
            an array of beam values, shape (Naxes_vec, Nspws, Nfeeds or Npols,
                Nfreqs or freq_array.size if freq_array is passed,
                Npixels/(Naxis1, Naxis2) or az_array.size if az/za_arrays are passed)
            an array of interpolated basis vectors (or self.basis_vector_array
                if az/za_arrays are not passed), shape: (Naxes_vec, Ncomponents_vec,
                Npixels/(Naxis1, Naxis2) or az_array.size if az/za_arrays are passed)
        """

        
        interp_data = np.zeros((2, 1, 2, freq_array.size, az_array.size), dtype=np.float)

        fscale = (freq_array / self.ref_freq) ** self.spectral_index

        x = 2.*np.sin( za_array[np.newaxis, ...] / fscale[:, np.newaxis]) - 1. # transformed zenith angle in radian, also scaled with frequency

        values = chebval(x, self.beam_coeffs)

        interp_data[1, 0, 0, :, :] = values
        interp_data[0, 0, 1, :, :] = values
        interp_basis_vector = None
    
        #FIXME:
        if self.beam_type == 'power':
            # Cross-multiplying feeds, adding vector components
            pairs = [(i, j) for i in range(2) for j in range(2)]
            power_data = np.zeros((1, 1, 4) + values.shape, dtype=np.float)
            for pol_i, pair in enumerate(pairs):
                power_data[:, :, pol_i] = ((interp_data[0, :, pair[0]]
                                           * np.conj(interp_data[0, :, pair[1]]))
                                           + (interp_data[1, :, pair[0]]
                                           * np.conj(interp_data[1, :, pair[1]])))
            interp_data = power_data

        return interp_data, interp_basis_vector

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.beam_coeffs == other.beam_coeffs:
            return True
        else:
            return False

