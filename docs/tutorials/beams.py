import numpy as np
from pyuvsim import AnalyticBeam
from numpy.polynomial.chebyshev import chebval, chebfit
from scipy.optimize import curve_fit

def perturb_PolyBeam(beam_coeffs=[], spectral_index=0., ref_freq=None, mainlobe_err=None, sidelobe_err=None, Nbeam=None):
    
    if (spectral_index != 0.0) and (ref_freq is None):
            raise ValueError("ref_freq must be set for nonzero beam spectral index")
    elif ref_freq is None:
            ref_freq = 1.0e8
    
    #Fagnoni beam(using Chebyshev polynomial)
    order = len(beam_coeffs) -1
    theta = np.linspace(0,np.pi/2.,100)
    tmptheta = 2.*np.sin(theta)-1.
    pbfitfag = chebval(tmptheta, beam_coeffs)
    
    #tanh function 
    theta_fwhm = 0.3 #approx. FWHM in radian
    y = 0.5 * (1. + np.tanh(50.*(theta - theta_fwhm))) #50 choose for sharpness
    
    #fagnoni * tanh
    pbfitfag_tanh = pbfitfag * y
    
    #fit (fagnoni * tanh) with Gaussian to perturb mainlobe only
    def gaussian(x, xalpha, A):
        return A*np.exp(-((x)/xalpha)**2)

    guess=(0.2,1.0)
    fit_params, cov_mat = curve_fit(gaussian, theta, pbfitfag_tanh, p0=guess)
    
    # Nbeam no. random sigma to perturb Gaussian mainlobe
    sigma = fit_params[0] * (1. + mainlobe_err * np.random.rand(Nbeam))
    
    ######################### Today's Discussion ########################
    #Fit Fagnoni Beam with Fourier Series
    def fn(x, *coeffs): 
        y = 0
        N=int(len(coeffs)/2)
        period=np.pi
        for n in range(N):
        
            an= coeffs[n]
            bn= coeffs[n+N]
            y += ( an * np.cos(2.*np.pi*n*x/period) + bn * np.sin(2.*np.pi*n*x/period))
        return y
    
    order_fourier = 20
    initial_coeffs = np.ones(order_fourier)
    coeff_fourier,cov_fourier=curve_fit(fn, theta, pbfitfag, initial_coeffs)
    pbfitfag_fourier=fn(theta,*coeff_fourier)
    
    
    pbfitfag_modfourier = (1. - y)*pbfitfag + y * pbfitfag_fourier
    pbfitfag_modfourier_res = pbfitfag - pbfitfag_modfourier
    
    ##########################################################################
    
    beams = []
    for sig in sigma:
        
        #set random sigma in fit_paramstmp[0]
        fit_paramstmp = np.copy(fit_params)
        fit_paramstmp[0] = sig #sig
    
        #subtract actual Gaussian and add perturb_Gaussian
        pbfitmod = pbfitfag - gaussian(theta, *fit_params)
        pbfitmod += gaussian(theta, *fit_paramstmp)
        
        #Fit the perturb_Gaussian with Chebyshev coeff, generate PolyBeam object
        coeffmod = chebfit(tmptheta, pbfitmod, order)
        beam = PolyBeam(beam_coeffs = coeffmod , spectral_index = spectral_index, ref_freq = ref_freq)
        beams.append(beam)
        
    return beams



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

