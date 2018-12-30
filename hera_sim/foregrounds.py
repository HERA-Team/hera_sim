'''A module for generating realistic foregrounds.'''

import numpy as np
from scipy import interpolate
import aipy
from . import noise
from . import utils


def diffuse_foreground(Tsky_mdl, lsts, fqs, bl_len_ns, bm_poly=noise.HERA_BEAM_POLY, scalar=30.,
                       fr_max_mult=2.0, standoff=0.0, delay_filter_type='gauss',
                       fringe_filter_type='gauss', **fringe_filter_kwargs):
    """
    Model diffuse foreground visibility.

    Args:
        Tsky_mdl : RectBivariateSpline interpolator, see noise.HERA_Tsky_mdl
            LST and freq dependent model of total autocorrelation power
        lsts : 1d array of LST [radians]
        fqs : 1d array of frequencies [GHz]
        bl_len_ns : float, projected East-West baseline separation in nanosec
        bm_poly : beam scalar polynomial object
        scalar : float, beam scalar
        fr_max_mult : float, maximum fringe-rate oversampling coefficient for lst_grid
        standoff : float, baseline horizon buffer for modeling suprahorizon emission
        delay_filter_type : str, type of delay filter to use, see utils.gen_delay_filter()
        fringe_filter_type : str, type of fringe-rate filter, see utils.gen_fringe_filter()
        fringe_filter_kwargs : kwargs given fringe_filter_type, see utils.gen_fringe_filter()
    Returns:
        vis : 2D array of diffuse foreground visibility
        fringe_filter : fringe-rate filter applied to data
        delay_filter : delay filter applied to data
    """
    # get maximum fringe-rate of baseline and produce lst_grid
    fr_max = np.max(utils.calc_max_fringe_rate(fqs, bl_len_ns))
    dt = 1.0/(fr_max_mult * fr_max)  # over-resolve by fr_mult factor
    ntimes = int(np.around(aipy.const.sidereal_day / dt))
    lst_grid = np.linspace(0, 2*np.pi, ntimes, endpoint=False)

    # generate a noise-like visibility of Tsky model
    data = Tsky_mdl(lst_grid, fqs) * noise.white_noise((ntimes, fqs.size))

    # fringe rate filter it
    data, fringe_filter = utils.rough_fringe_filter(data, lst_grid, fqs, bl_len_ns, filter_type=fringe_filter_type, **fringe_filter_kwargs)

    # convert from Temp to Jy and interpolate onto lsts
    data /= noise.jy2T(fqs, bm_poly=bm_poly)
    data_real = interpolate.interp1d(lst_grid, scalar * data.real, kind='quadratic', fill_value='extrapolate', axis=0)
    data_imag = interpolate.interp1d(lst_grid, scalar * data.imag, kind='quadratic', fill_value='extrapolate', axis=0)
    data = data_real(lsts) + 1j * data_imag(lsts)

    # delay filter it
    data, delay_filter = utils.rough_delay_filter(data, fqs, bl_len_ns, standoff=standoff, filter_type=delay_filter_type)

    return data, fringe_filter, delay_filter


def pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=1000):
    """
    Need a doc string...
    """
    ras = np.random.uniform(0,2*np.pi,nsrcs)
    indices = np.random.normal(-1, .5, size=nsrcs)
    mfreq = .15
    beam_width = (40*60.) * (mfreq/fqs) / aipy.const.sidereal_day * 2*np.pi # XXX hardcoded HERA
    x0,x1,n = .3, 300, -1.5
    # Draw flux densities from a power law between 1 and 1000 w/ index of -1.5
    flux_densities = ((x1**(n+1) - x0**(n+1))*np.random.uniform(size=nsrcs) + x0**(n+1))**(1./(n+1))
    vis = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    for ra,flux,index in zip(ras,flux_densities,indices):
        t = np.argmin(np.abs(utils.compute_ha(lsts,ra)))
        dtau = np.random.uniform(-.1*bl_len_ns,.1*bl_len_ns) # XXX adds a bit to total delay, increasing bl_len_ns
        vis[t,:] += flux * (fqs/mfreq)**index * np.exp(2j*np.pi*fqs*dtau)
    ha = utils.compute_ha(lsts, 0)
    for fi in xrange(fqs.size):
        bm = np.exp(-ha**2 / (2*beam_width[fi]**2))
        bm = np.where(np.abs(ha) > np.pi/2, 0, bm)
        w = .9*bl_len_ns * np.sin(ha) * fqs[fi] # XXX .9 to offset increase from dtau above
        phs = np.exp(2j*np.pi*w)
        kernel = bm * phs
        vis[:,fi] = np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(vis[:,fi]))
    return vis
