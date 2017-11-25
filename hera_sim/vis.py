import numpy as np
import hera_cal.omni
import foregrounds, rfi, noise, sigchain

DEFAULT_LST = np.linspace(0, 2*np.pi, 10000, endpoint=False)
DEFAULT_FQS = np.linspace(0.1, 0.2, 1024, endpoint=False)

def make_hera_obs(aa, lsts=DEFAULT_LSTS, fqs=DEFAULT_FQS, pols=['xx','yy'], T_rx=150., inttime=10.7,
        rfi_impulse=.02, rfi_scatter=.001, nsrcs=200, gain_spread=.1, dly_rng=(-20,20), xtalk=3.):
    info = hera_cal.omni.aa_to_info(aa)
    reds = info.get_reds()
    ants = list(set([ant for bls in reds for bl in bls for ant in [(bl[0],bl[2][0]), (bl[1],bl[2][1])]]))
    data, true_vis = {}, {}
    if gains is None: gains = {}
    else: gains = deepcopy(gains)
    for ant in ants:
        gains[ant] = gains.get(ant, 1+gain_scatter*noise((1,))) * np.ones(shape,dtype=np.complex)
    for bls in reds:
        true_vis[bls[0]] = noise(shape)
        for (i,j,pol) in bls:
            data[(i,j,pol)] = true_vis[bls[0]] * gains[(i,pol[0])] * gains[(j,pol[1])].conj()
    return gains, true_vis, data
~                                   


# XXX from hera_cal.redcal
def sim_red_data(reds, gains=None, shape=(10,10), gain_scatter=.1):
    """ Simulate noise-free random but redundant (up to differing gains) visibilities.


        Args:
            reds: list of lists of baseline-pol tuples where each sublist has only
                redundant pairs
            gains: pre-specify base gains to then scatter on top of in the
                {(index,antpol): np.array} format. Default gives all ones.
            shape: tuple of (Ntimes, Nfreqs). Default is (10,10).
            gain_scatter: Relative amplitude of per-antenna complex gain scatter. Default is 0.1.

        Returns:
            gains: true gains used in the simulation in the {(index,antpol): np.array} format
            true_vis: true underlying visibilities in the {(ind1,ind2,pol): np.array} format
            data: simulated visibilities in the {(ind1,ind2,pol): np.array} format
    """

    data, true_vis = {}, {}
    ants = list(set([ant for bls in reds for bl in bls for ant in [(bl[0],bl[2][0]), (bl[1],bl[2][1])]]))
    if gains is None: gains = {}
    else: gains = deepcopy(gains)
    for ant in ants:
        gains[ant] = gains.get(ant, 1+gain_scatter*noise((1,))) * np.ones(shape,dtype=np.complex)
    for bls in reds:
        true_vis[bls[0]] = noise(shape)
        for (i,j,pol) in bls:
            data[(i,j,pol)] = true_vis[bls[0]] * gains[(i,pol[0])] * gains[(j,pol[1])].conj()
    return gains, true_vis, data
