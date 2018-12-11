'''A module for modeling HERA signal chains.'''

from . import noise
import numpy as np
import aipy

HERA_NRAO_BANDPASS = np.array([-2.04689451e+06, 1.90683718e+06, 
-7.41348361e+05, 1.53930807e+05, -1.79976473e+04, 1.12270390e+03, 
-2.91166102e+01]) # See "HERA's Passband to First Order"

def gen_bandpass(freqs, ants, gain_spread=.1):
    bp_base = np.polyval(HERA_NRAO_BANDPASS, freqs)
    window = aipy.dsp.gen_window(freqs.size, 'blackman-harris')
    _modes = np.abs(np.fft.fft(window*bp_base))
    g = {}
    for ai in ants:
        delta_bp = np.fft.ifft(noise.white_noise(freqs.size) * _modes * gain_spread)
        g[ai] = bp_base + delta_bp
    return g

def gen_delay_phs(freqs, ants, dly_rng=(-20,20)):
    phs = {}
    for ai in ants:
        dly = np.random.uniform(dly_rng[0], dly_rng[1])
        phs[ai] = np.exp(2j*np.pi*dly*freqs)
    return phs

def gen_gains(freqs, ants, gain_spread=.1, dly_rng=(-20,20)):
    bp = gen_bandpass(freqs, ants, gain_spread)
    phs = gen_delay_phs(freqs, ants, dly_rng)
    return {ai: bp[ai]*phs[ai] for ai in ants}

def apply_gains(vis, gains, bl):
    gij = gains[bl[0]] * gains[bl[1]].conj()
    gij.shape = (1,-1)
    return vis * gij

def gen_xtalk(freqs, amplitude=3.):
    xtalk = np.convolve(noise.white_noise(freqs.size), np.ones(50), 'same')
    return amplitude * xtalk

def apply_xtalk(vis, xtalk):
    xtalk = np.reshape(xtalk, (1,-1))
    return vis + xtalk
    
