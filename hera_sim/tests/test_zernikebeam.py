import numpy as np
from hera_sim.visibilities import VisCPU
from hera_sim import io
from hera_sim.beams import ZernikeBeam
from hera_sim.defaults import defaults

np.seterr(invalid="ignore")


def antennas():
    locs = [[308, 253, 0.49], [8, 299, 0.22]]

    ants = {}
    for i in range(len(locs)):
        ants[i] = (locs[i][0], locs[i][1], locs[i][2])

    return ants


def sources():
    sources = np.array(
        [
            [
                128,
                -29,
                4,
                0,
            ]
        ]
    )
    ra_dec = sources[:, :2]
    flux = sources[:, 2]
    spectral_index = sources[:, 3]

    ra_dec = np.deg2rad(ra_dec)

    return ra_dec, flux, spectral_index


def beams(nants):
    """
    Zernike polynomial beams with some randomly-chosen coefficients.
    """
    cfg_beam = dict(
        ref_freq=1.0e8,
        spectral_index=-0.6975,
        beam_coeffs=[
            1.29778665,
            0.2,
            0.3,
            -0.10030698,
            -0.01195859,  # nofmt
            0.06063853,
            -0.04593295,
            0.0107879,
            0.01390283,
            -0.01881641,
            -0.00177106,
            0.01265177,
            -0.00568299,
            -0.00333975,
            0.00452368,
            0.00151808,
            -0.00593812,
            0.00351559,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.01,
        ],
    )
    beams = [ZernikeBeam(**cfg_beam) for i in range(nants)]

    return beams


class DummyMPIComm:
    """
    Exists so the MPI interface can be tested, but not run.
    """

    def Get_size(self):
        return 2  # Pretend there are 2 processes


def run_sim(use_pixel_beams=True, use_gpu=False, use_mpi=False):
    """
    Run a simple sim using a rotated elliptic polybeam.
    """
    defaults.set("h1c")

    ants = antennas()

    # Observing parameters in a UVData object.
    uvdata = io.empty_uvdata(
        Nfreqs=1,
        start_freq=100000000.0,
        channel_width=97000.0,
        start_time=2458902.4,
        integration_time=40,
        Ntimes=1,
        array_layout=ants,
    )

    freqs = np.unique(uvdata.freq_array)
    ra_dec, flux, spectral_index = sources()

    # calculate source fluxes for hera_sim
    flux = (freqs[:, np.newaxis] / freqs[0]) ** spectral_index * flux

    simulator = VisCPU(
        uvdata=uvdata,
        beams=beams(len(ants.keys())),
        beam_ids=list(ants.keys()),
        sky_freqs=freqs,
        point_source_pos=ra_dec,
        point_source_flux=flux,
        use_pixel_beams=use_pixel_beams,
        use_gpu=use_gpu,
        mpi_comm=DummyMPIComm() if use_mpi else None,
        bm_pix=300,
        precision=2,
    )
    simulator.simulate()

    auto = np.abs(simulator.uvdata.get_data(0, 0, "XX")[0][0])

    return auto


def test_zernike_beam():

    # Calculate visibilities using pixel and interpolated beams
    pix_result = run_sim(use_pixel_beams=True)
    calc_result = run_sim(use_pixel_beams=False)

    # Check that the maximum difference between pixel beams/direct calculation
    # cases is no more than 5%. This shows the direct calculation of the beam
    # tracks the pixel beam interpolation. They won't be exactly the same.
    np.testing.assert_allclose(pix_result, calc_result, rtol=0.05)

    # Check basic methods
    beam1, beam2 = beams(2)  # get two beams
    assert beam1 == beam2  # test __eq__ method

    # Coords to evaluate at
    za = np.linspace(0.0, 0.5 * np.pi, 40)
    az = np.zeros(za.size)
    freqs = np.array(
        [
            100.0e6,
        ]
    )

    # Check peak normalize works (beams are peak normalized by default)
    beam1.peak_normalized = False
    y1 = beam1.interp(az_array=az, za_array=za, freq_array=freqs)[0]
    y2 = beam2.interp(az_array=az, za_array=za, freq_array=freqs)[0]
    beam1.peak_normalize()
    y1a = beam1.interp(az_array=az, za_array=za, freq_array=freqs)[0]

    assert ~np.allclose(y1, y2)  # Unnormalized vs peak normalized should be different
    assert np.allclose(y1a, y2)  # Peak normalized beams should give same results
