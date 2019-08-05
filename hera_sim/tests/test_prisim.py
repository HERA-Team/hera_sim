import unittest

import numpy as np
import pytest

from hera_sim import visibilities as vis, io


#####################################
#SIMULATORS = (vis.HealVis, vis.VisCPU)
SIMULATORS = (vis.PRISim, vis.HealVis, vis.VisCPU)
#####################################

np.random.seed(0)
NTIMES = 10
BM_PIX = 31
NPIX = 12 * 16 ** 2
NFREQ = 5


@pytest.fixture
def uvdata():
    return io.empty_uvdata(
        nfreq=NFREQ,
        time_per_integ=io.SEC_PER_SDAY / NTIMES,
        ntimes=NTIMES,
        ants={
            0: (0, 0, 0),
        },
        antpairs=[(0, 0)]
    )

@pytest.fixture
def uvdata2():
    return io.empty_uvdata(
        nfreq=NFREQ,
        time_per_integ=io.SEC_PER_SDAY / NTIMES,
        ntimes=NTIMES,
        ants={
            0: (0, 0, 0),
            1: (1, 0, 0)
        },
        antpairs=[(0, 0), (1, 1), (1, 0), (0, 1)]
    )


def create_uniform_sky(nbase=4, scale=1, nfreq=NFREQ):
    """Create a uniform sky with total (integrated) flux density of `scale`"""
    nside = 2 ** nbase
    npix = 12 * nside ** 2
    return np.ones((nfreq, npix)) * scale / (4 * np.pi)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_shapes(uvdata, simulator):
    I_sky = create_uniform_sky()

    v = simulator(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        sky_intensity=I_sky,
    )

    assert v.simulate().shape == (NTIMES*len(uvdata.get_antpairs()), 1, NFREQ, 1) ############################ADDED NUMBASELINES



@pytest.mark.parametrize("simulator", SIMULATORS)
def test_zero_sky(uvdata, simulator):
    I_sky = create_uniform_sky(scale=0)

    sim = simulator(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        sky_intensity=I_sky
    )
    v = sim.simulate()
    np.testing.assert_equal(v, 0)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_autocorr_flat_beam(uvdata, simulator):
    I_sky = create_uniform_sky(nbase=6)

    v = simulator(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        sky_intensity=I_sky,
    ).simulate()

    np.testing.assert_allclose(np.abs(v), np.mean(v), rtol=1e-3)
    np.testing.assert_almost_equal(np.abs(v), 0.5, 2)


def test_viscpu_res_autocorr(uvdata):
    I_sky = create_uniform_sky(nbase=5) #was 5
    v = vis.PRISim(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        sky_intensity=I_sky,
    ).simulate()

    I_sky = create_uniform_sky(nbase=6) #was 6
    v2 = vis.PRISim(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        sky_intensity=I_sky,
    ).simulate()

    ''' 
    for i in range(10):
        I_sky = create_uniform_sky(nbase=i)
        v3 = vis.PRISim(
            uvdata=uvdata,
            sky_freqs=np.unique(uvdata.freq_array),
            sky_intensity=I_sky,
        ).simulate()
        print("ISKY STD", np.std(I_sky))
        print("V3 SUM", np.sum(v3))
        print("NBASE=", i, "=> np.std(np.abs(v3)) =", np.std(np.abs(v3)))
    assert 1 == 0
    '''

    # Ensure that increasing sky resolution smooths out
    # any 'wiggles' in the auto-correlations of a flat sky.
    assert np.std(np.abs(v)) >= np.std(np.abs(v2))

@pytest.mark.parametrize("simulator", SIMULATORS)
def test_single_source_autocorr(uvdata, simulator):
    freqs = np.unique(uvdata.freq_array)

    # put a point source in that will go through zenith.
    point_source_pos = np.array([[0, uvdata.telescope_lat_lon_alt[0]]])
    point_source_flux = np.array([[1.0]] * len(freqs))

    v = simulator(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2**4,
    ).simulate()

    # Make sure the source is over the horizon half the time
    # (+/- 1 because of the discreteness of the times)
    # 1e-3 on either side to account for float inaccuracies.
    assert -1e-3 + (NTIMES/2.0 - 1.0)/NTIMES <= np.round(np.abs(np.mean(v)), 3) <= (NTIMES/2.0 + 1.0)/NTIMES + 1e-3


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_single_source_autocorr_past_horizon(uvdata, simulator):
    freqs = np.unique(uvdata.freq_array)

    # put a point source in that will never be up
    point_source_pos = np.array([[0, uvdata.telescope_lat_lon_alt[0] + 1.1 * np.pi / 2]])
    point_source_flux = np.array([[1.0]] * len(freqs))

    v = simulator(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2**4
    ).simulate()

    assert np.abs(np.mean(v)) == 0

    # def test_exact_value_two_sources(self):
    #
    #     # For ant[0] at (0,0,1), ant[1] at (1,1,1), src[0] at (0,0,1) and src[1] at (0,.707,.707)
    #     antpos[0, 0] = 0
    #     antpos[0, 1] = 0
    #     v = simulators.vis_cpu(antpos, 1.0, eq2tops, crd_eq, I_sky, bm_cube)
    #     np.testing.assert_almost_equal(
    #         v[:, 0, 1], 1 + np.exp(-2j * np.pi * np.sqrt(0.5)), 7
    #     )


def test_prisim_healvis1(uvdata):
    freqs = np.unique(uvdata.freq_array)

    # put a point source in
    point_source_pos = np.array([[0, uvdata.telescope_lat_lon_alt[0]]])
    point_source_flux = np.array([[1.0]] * len(freqs))

    prisim = vis.PRISim(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2**4
    ).simulate()

    healvis = vis.HealVis(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2 ** 4
    ).simulate()

    np.save("prisim.npy", prisim)
    np.save("healvis.npy", healvis)

    print "SUM PRISIM", np.sum(prisim)
    print "SUM HEALVIS", np.sum(healvis)

    assert prisim.shape == healvis.shape
    assert np.testing.assert_allclose(prisim, healvis) 

def test_prisim_healvis2(uvdata):
    freqs = np.unique(uvdata.freq_array)

    # put a point source in
    point_source_pos = np.array([[0, uvdata.telescope_lat_lon_alt[0] + np.pi/4]])
    point_source_flux = np.array([[1.0]] * len(freqs))

    prisim = vis.PRISim(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2**4
    ).simulate()

    healvis = vis.HealVis(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2 ** 4
    ).simulate()
    
    
    np.save("prisim2.npy", prisim)
    np.save("healvis2.npy", healvis)

    print "SUM PRISIM", np.sum(prisim)
    print "SUM HEALVIS", np.sum(healvis)

    assert prisim.shape == healvis.shape
    assert np.testing.assert_allclose(prisim, healvis)

def test_prisim_healvis3(uvdata2):
    freqs = np.unique(uvdata2.freq_array)

    # put a point source in
    point_source_pos = np.array([[0, uvdata2.telescope_lat_lon_alt[0] + np.pi/4]])
    point_source_flux = np.array([[1.0]] * len(freqs))

    prisim = vis.PRISim(
        uvdata=uvdata2,
        sky_freqs=np.unique(uvdata2.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2**4
    ).simulate()

    healvis = vis.HealVis(
        uvdata=uvdata2,
        sky_freqs=np.unique(uvdata2.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2 ** 4
    ).simulate()

    np.save("prisim3.npy", prisim)
    np.save("healvis3.npy", healvis)
    print "SUM PRISIM", np.sum(prisim)
    print "SUM HEALVIS", np.sum(healvis)
    assert prisim.shape == healvis.shape
    assert np.testing.assert_allclose(prisim, healvis) 

def test_prisim_viscpu1(uvdata):
    freqs = np.unique(uvdata.freq_array)

    # put a point source in
    point_source_pos = np.array([[0, uvdata.telescope_lat_lon_alt[0]]])
    point_source_flux = np.array([[1.0]] * len(freqs))

    prisim = vis.PRISim(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2**4
    ).simulate()

    viscpu = vis.VisCPU(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2 ** 4
    ).simulate()

    np.save("viscpu.npy", viscpu)

    print "SUM PRISIM", np.sum(prisim)
    print "SUM VISCPU", np.sum(viscpu)

    assert prisim.shape == viscpu.shape
    assert np.testing.assert_allclose(prisim, viscpu) 

def test_prisim_viscpu2(uvdata):
    freqs = np.unique(uvdata.freq_array)

    # put a point source in
    point_source_pos = np.array([[0, uvdata.telescope_lat_lon_alt[0] + np.pi/4]])
    point_source_flux = np.array([[1.0]] * len(freqs))

    prisim = vis.PRISim(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2**4
    ).simulate()

    viscpu = vis.VisCPU(
        uvdata=uvdata,
        sky_freqs=np.unique(uvdata.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2 ** 4
    ).simulate()

    np.save("viscpu2.npy", viscpu)

    print "SUM PRISIM", np.sum(prisim)
    print "SUM VISCPU", np.sum(viscpu)

    assert prisim.shape == viscpu.shape
    assert np.testing.assert_allclose(prisim, viscpu)

def test_prisim_viscpu3(uvdata2):
    freqs = np.unique(uvdata2.freq_array)

    # put a point source in
    point_source_pos = np.array([[0, uvdata2.telescope_lat_lon_alt[0] + np.pi/4]])
    point_source_flux = np.array([[1.0]] * len(freqs))

    prisim = vis.PRISim(
        uvdata=uvdata2,
        sky_freqs=np.unique(uvdata2.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2**4
    ).simulate()

    viscpu = vis.VisCPU(
        uvdata=uvdata2,
        sky_freqs=np.unique(uvdata2.freq_array),
        point_source_flux=point_source_flux,
        point_source_pos=point_source_pos,
        nside=2 ** 4
    ).simulate()

    np.save("viscpu3.npy", viscpu)

    print "SUM PRISIM", np.sum(prisim)
    print "SUM VISCPU", np.sum(viscpu)

    assert prisim.shape == viscpu.shape
    assert np.testing.assert_allclose(prisim, viscpu)


if __name__ == "__main__":
    unittest.main()
