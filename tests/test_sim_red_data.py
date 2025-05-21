import numpy as np
import pytest
from hera_cal import redcal as om

from hera_sim import vis
from hera_sim.antpos import linear_array


@pytest.fixture(scope="function")
def antpos():
    return linear_array(5)


@pytest.fixture(scope="function")
def gain_data_1pol(antpos):
    reds = om.get_reds(antpos, pols=["nn"], pol_mode="1pol")
    gains, true_vis, data = vis.sim_red_data(reds)
    return gains, data, reds


def gain_data_4pol(antpos):
    reds = om.get_reds(antpos, pols=["xx", "yy", "xy", "yx"], pol_mode="4pol")
    gains, true_vis, data = vis.sim_red_data(reds)
    return gains, data, reds


def gain_data_4pol_minv(antpos):
    reds = om.get_reds(antpos, pols=["xx", "yy", "xy", "yx"], pol_mode="minV")
    gains, true_vis, data = vis.sim_red_data(reds)
    return gains, data, reds


def test_sim_red_data_1pol(gain_data_1pol):
    # Test that redundant baselines are redundant up to the gains in single pol mode
    gains, data, reds = gain_data_1pol
    assert len(gains) == 5
    assert len(data) == 10
    for bls in reds:
        bl0 = bls[0]
        ai, aj, pol = bl0
        ans0 = data[bl0] / (gains[(ai, "Jnn")] * gains[(aj, "Jnn")].conj())
        for bl in bls[1:]:
            ai, aj, pol = bl
            ans = data[bl] / (gains[(ai, "Jnn")] * gains[(aj, "Jnn")].conj())
            # compare calibrated visibilities knowing the input gains
            np.testing.assert_almost_equal(ans0, ans, decimal=7)


@pytest.mark.parametrize("gain_data", [gain_data_4pol, gain_data_4pol_minv])
def test_sim_red_data_4pol(antpos, gain_data):
    # Test that redundant baselines are redundant up to the gains in 4-pol mode
    gains, data, reds = gain_data(antpos)
    assert len(gains) == 2 * (5)
    assert len(data) == 4 * (10)
    for bls in reds:
        bl0 = bls[0]
        for pol in ["xx", "xy", "yx", "yy"]:
            ai, aj, pol = bl0
            ans0 = data[ai, aj, pol] / (
                gains[(ai, f"J{pol[0] * 2}")] * gains[(aj, f"J{pol[1] * 2}")].conj()
            )

            for bl in bls[1:]:
                ai, aj, pol = bl
                ans = data[ai, aj, pol] / (
                    gains[(ai, f"J{pol[0] * 2}")] * gains[(aj, f"J{pol[1] * 2}")].conj()
                )

                # compare calibrated visibilities knowing the input gains
                np.testing.assert_almost_equal(ans0, ans, decimal=7)
