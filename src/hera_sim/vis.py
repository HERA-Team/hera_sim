"""Functions for producing white-noise redundant visibilities."""

from copy import deepcopy

import numpy as np

from . import noise

DEFAULT_LSTS = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
DEFAULT_FQS = np.linspace(0.1, 0.2, 1024, endpoint=False)


def sim_red_data(
    reds,
    gains=None,
    shape=(10, 10),
    gain_scatter=0.1,
    rng: np.random.Generator | None = None,
):
    """
    Simulate thermal-noise-free random but redundant (up to gains) visibilities.

    Parameters
    ----------
    reds : list of list of tuples
        list of lists of baseline-pol tuples where each sublist has only
        redundant pairs
    gains : dict
        pre-specify base gains to then scatter on top of in the
        {(index,antpol): ndarray} format. Default gives all ones.
    shape : tuple
        (Ntimes, Nfreqs).
    gain_scatter : float
        relative amplitude of per-antenna complex gain scatter

    Returns
    -------
    dict
        true gains used in the simulation in the {(index,antpol): np.array} format
    dict
        true underlying visibilities in the {(ind1,ind2,pol): np.array} format
    dict
        simulated visibilities in the {(ind1,ind2,pol): np.array} format
    """
    from hera_cal.utils import split_bl

    if rng is None:
        rng = np.random.default_rng()

    data, true_vis = {}, {}
    ants = sorted({ant for bls in reds for bl in bls for ant in split_bl(bl)})
    gains = {} if gains is None else deepcopy(gains)
    for ant in ants:
        gains[ant] = gains.get(
            ant, 1 + gain_scatter * noise.white_noise((1,), rng=rng)
        ) * np.ones(shape, dtype=complex)
    for bls in reds:
        true_vis[bls[0]] = noise.white_noise(shape, rng=rng)
        for bl in bls:
            data[bl] = (
                true_vis[bls[0]]
                * gains[split_bl(bl)[0]]
                * gains[split_bl(bl)[1]].conj()
            )
    return gains, true_vis, data
