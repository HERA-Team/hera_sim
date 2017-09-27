'''A module for generating realistic foregrounds.'''

import numpy as np
from scipy.interpolate import RectBivariateSpline
import aipy
import noise

def rough_delay_filter(noise, fqs, bl_len_ns):
    dlys = np.fft.fftfreq(fqs.size, fqs[1]-fqs[0])
    _noise = np.fft.fft(noise)
    dly_filter = np.exp(-dlys**2 / (2*bl_len_ns**2))
    dly_filter.shape = (1,) * (_noise.ndim-1) + (-1,)
    return np.fft.ifft(_noise * dly_filter)

def calc_max_fringe_rate(fqs, bl_len_ns):
    bl_wavelen = fqs * bl_len_ns
    fr_max = 2*np.pi/aipy.const.sidereal_day * bl_wavelen
    return fr_max

def rough_fringe_filter(noise, lsts, fqs, bl_len_ns):
    fr_max = calc_max_fringe_rate(fqs, bl_len_ns)
    fr_max.shape = (1,) * (noise.ndim-1) + (-1,)
    times = lsts / (2*np.pi) * aipy.const.sidereal_day
    fringe_rates = np.fft.fftfreq(times.size, times[1]-times[0])
    fringe_rates.shape = (-1,) + (1,) * (noise.ndim-1)
    _noise = np.fft.fft(noise, axis=-2)
    fng_filter = np.where(np.abs(fringe_rates) < fr_max, 1., 0)
    return np.fft.ifft(_noise * fng_filter, axis=-2)
    
def diffuse_foreground(Tsky, lsts, fqs, bl_len_ns, bm_poly=noise.HERA_BEAM_POLY, scalar=30.):
    fr_max = np.max(calc_max_fringe_rate(fqs, bl_len_ns))
    dt = 0.5/fr_max # over-resolve by factor of 2
    ntimes = int(np.around(aipy.const.sidereal_day / dt))
    lst_grid = np.linspace(0, 2*np.pi, ntimes, endpoint=False)
    nos = Tsky(lst_grid,fqs) * noise.white_noise((ntimes,fqs.size))
    nos = rough_fringe_filter(nos, lst_grid, fqs, bl_len_ns)
    nos = rough_delay_filter(nos, fqs, bl_len_ns)
    nos /= noise.jy2T(fqs, bm_poly=bm_poly)
    mdl_real = RectBivariateSpline(lst_grid, fqs, scalar*nos.real)
    mdl_imag = RectBivariateSpline(lst_grid, fqs, scalar*nos.imag)
    return mdl_real(lsts,fqs) + 1j*mdl_imag(lsts,fqs)
    
def compute_ha(lsts, ra):
    ha = lsts - ra
    ha = np.where(ha > np.pi, ha-2*np.pi, ha)
    ha = np.where(ha < -np.pi, ha+2*np.pi, ha)
    return ha

def pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=1000):
    ras = np.random.uniform(0,2*np.pi,nsrcs)
    indices = np.random.normal(-1, .5, size=nsrcs)
    mfreq = .15
    beam_width = (40*60.) * (mfreq/fqs) / aipy.const.sidereal_day * 2*np.pi # XXX hardcoded HERA
    x0,x1,n = .3, 300, -1.5
    # Draw flux densities from a power law between 1 and 1000 w/ index of -1.5
    flux_densities = ((x1**(n+1) - x0**(n+1))*np.random.uniform(size=nsrcs) + x0**(n+1))**(1./(n+1))
    vis = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    for ra,flux,index in zip(ras,flux_densities,indices):
        t = np.argmin(np.abs(compute_ha(lsts,ra)))
        dtau = np.random.uniform(-.1*bl_len_ns,.1*bl_len_ns) # XXX adds a bit to total delay, increasing bl_len_ns
        vis[t,:] += flux * (fqs/mfreq)**index * np.exp(2j*np.pi*fqs*dtau)
    ha = compute_ha(lsts, 0)
    for fi in xrange(fqs.size):
        bm = np.exp(-ha**2 / (2*beam_width[fi]**2))
        bm = np.where(np.abs(ha) > np.pi/2, 0, bm)
        w = .9*bl_len_ns * np.sin(ha) * fqs[fi] # XXX .9 to offset increase from dtau above
        phs = np.exp(2j*np.pi*w)
        kernel = bm * phs
        vis[:,fi] = np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(vis[:,fi]))
    return vis
