'''A module for modeling HERA signal chains.'''

import numpy as np
import aipy

HERA_NRAO_BANDPASS = np.array([-2.04689451e+06, 1.90683718e+06, 
-7.41348361e+05, 1.53930807e+05, -1.79976473e+04, 1.12270390e+03, 
-2.91166102e+01]) # See "HERA's Passband to First Order"

def gen_bandpass(freqs, ants, gain_spread=.1, bandpass_spread=.1):
    bp_base = np.polyval(HERA_NRAO_BANDPASS, freqs)
    g = {}
    for ai in ants:
        bp_poly = np.random.normal(1, bandpass_spread, 
                size=HERA_NRAO_PASSBAND.size)
        g[ai] = np.random.normal(1., gain_spread) * np.polyval(bp_poly, freqs)
    return g

def gen_delay_phs(freqs, ants, dly_rng=(0,20)):
    phs = {}
    for ai in ants:
        dly = np.random.uniform(dly_rng[0], dly_rng[1])
        phs[ai] = np.exp(2j*np.pi*dly*freqs)
    return phs

def gen_gains(freqs, ants):
    bp = gen_bandpass(freqs, ants)
    phs = gen_delay_phs(freqs, ants)
    return {ai: bp[ai]*phs[ai] for ai in ants}

def apply_gains(vis, gains, bl):
    gij = gains[bl[0]] * gains[bl[1]].conj()
    gij.shape = (1,-1)
    return vis * gij
    
