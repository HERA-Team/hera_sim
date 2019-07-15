"""
A module containing routines for interfacing data produced by `hera_sim` with other codes, especially UVData.
"""
import itertools

import numpy as np
import pyuvdata as uv

def make_data(season=None, *args, **kwargs):
    # simulate data for a given observing season
    # provide functionality for choosing which things to simulate
    # e.g. rfi, gains, foregrounds, etc.
    # require {ant:pos} as an argument?
    # require Nblts as an argument for baseline-dependent integration times?
    # or make baseline-dependent integration times a simulation option?
    # kwargs will be for various simulation-component parameters
    # such as integration time, number of sources (for pntsrc fgs), etc.
    # will return a UVData object filled with the simulated data
    # XXX this should include a check for sufficient memory, since
    # XXX simulating a full array's worth of visibilities will require
    # XXX substantial memory


# XXX move this to utils?
def _get_antpairs(ants, antpairs):
    # Generate antpairs
    if antpairs is None:
        # Use all pairs (including auto-correlations)
        antpairs = [(ant, ant) for ant in ants] + list(itertools.combinations(ants.keys(), 2))
    elif isinstance(antpairs, str):
        if antpairs == "cross":
            # Use all cross-pairs but no autos
            antpairs = list(itertools.combinations(ants.keys(), 2))
        elif antpairs == 'autos':
            antpairs = [(ant, ant) for ant in ants]
        elif antpairs == "EW":
            # Use only baselines that are close to EW-oriented (< 10% NS)
            antpairs = []
            antnums = list(ants)

            for i, ant1 in enumerate(antnums):
                pos1 = ants[ant1]

                for ant2 in antnums[i:]:
                    pos2 = ants[ant2]

                    if ant1 == ant2 or np.abs(pos1[1] - pos2[1]) / np.abs(pos1[0] - pos2[0]) < 0.1:
                        antpairs += [(ant1, ant2)]

        elif antpairs == 'redundant':
            # Use only a single baseline from all redundant "types", with redundancy within
            # 0.1m ?
            antpairs = []
            baseline_types = {}
            antnums = list(ants)
            for i, ant1 in enumerate(antnums):
                pos1 = ants[ant1]

                for ant2 in antnums[i:]:
                    pos2 = ants[ant2]

                    if ant1 == ant2:
                        antpairs += [(ant1, ant2)]
                    else:
                        id = str(list(np.round([p1 - p2 for p1, p2 in zip(pos1, pos2)], decimals=1)))
                        if id not in baseline_types:
                            antpairs += [(ant1, ant2)]
                            baseline_types[id] = (ant1, ant2)
        else:
            raise ValueError("if antpairs is a string, it must be one of 'cross', 'autos', 'EW' or 'redundant'.")

    # Check that baselines only involve antennas that have been defined
    try:
        ant1, ant2 = zip(*antpairs)
    except (TypeError, ValueError):
        raise TypeError("antpairs must be a list of 2-tuples")

    return antpairs, ant1, ant2
