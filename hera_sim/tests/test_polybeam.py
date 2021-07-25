import pytest
import numpy as np
from hera_sim.visibilities import VisCPU
from hera_sim import io
from hera_sim.beams import PerturbedPolyBeam
from vis_cpu import HAVE_GPU
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


def beams(rotation, nants, polarized=False):
    """
    Elliptical PerturbedPolyBeam.

    This will also test PolyBeam, from which PerturbedPolybeam is derived.
    """
    cfg_beam = dict(
        ref_freq=1.0e8,
        spectral_index=-0.6975,
        perturb=True,
        mainlobe_width=0.3,
        nmodes=8,
        beam_coeffs=[
            0.29778665,
            -0.44821433,
            0.27338272,
            -0.10030698,
            -0.01195859,
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
        ],
    )
    beams = [
        PerturbedPolyBeam(
            perturb_coeff=np.array(
                [
                    -0.20437532,
                    -0.4864951,
                    -0.18577532,
                    -0.38053642,
                    0.08897764,
                    0.06367166,
                    0.29634711,
                    1.40277112,
                ]
            ),
            mainlobe_scale=1.0,
            xstretch=1.1,
            ystretch=0.8,
            rotation=rotation,
            polarized=polarized,
            **cfg_beam
        )
        for i in range(nants)
    ]

    return beams


class DummyMPIComm:
    """
    Exists so the MPI interface can be tested, but not run.
    """

    def Get_size(self):
        return 2  # Pretend there are 2 processes


def run_sim(
    beam_rotation,
    use_pixel_beams=True,
    use_gpu=False,
    use_pol=False,
    use_mpi=False,
    pol="xx",
):
    """
    Run a simple sim using a rotated elliptic polybeam.
    """
    defaults.set("h1c")
    pol_array = ["xx"]
    if use_pol:
        pol_array = np.array(
            ["yx", "xy", "yy", "xx"]
        )  # yx, xy, yy, xx = ne, en, nn, ee

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
        polarization_array=pol_array,
    )
    freqs = np.unique(uvdata.freq_array)
    ra_dec, flux, spectral_index = sources()

    # calculate source fluxes for hera_sim
    flux = (freqs[:, np.newaxis] / freqs[0]) ** spectral_index * flux

    simulator = VisCPU(
        uvdata=uvdata,
        beams=beams(beam_rotation, len(ants.keys()), polarized=use_pol),
        beam_ids=list(ants.keys()),
        sky_freqs=freqs,
        point_source_pos=ra_dec,
        point_source_flux=flux,
        use_pixel_beams=use_pixel_beams,
        use_gpu=use_gpu,
        polarized=use_pol,
        mpi_comm=DummyMPIComm() if use_mpi else None,
        bm_pix=200,
        precision=2,
    )
    simulator.simulate()
    auto = np.abs(simulator.uvdata.get_data(0, 0, pol)[0][0])

    return auto


class TestPerturbedPolyBeam:
    def test_perturbed_polybeam(self):

        # Rotate the beam from 0 to 180 degrees, and check that autocorrelation
        # of antenna 0 has approximately the same value when pixel beams are
        # used, and when pixel beams not used (direct beam calculation).
        rotations = np.zeros(180 + 1)
        pix_results = np.zeros(180 + 1)
        calc_results = np.zeros(180 + 1)
        for r in range(180 + 1):
            pix_result = run_sim(r, use_pixel_beams=True)

            # Direct beam calculation - no pixel beams
            calc_result = run_sim(r, use_pixel_beams=False)

            rotations[r] = r
            pix_results[r] = pix_result
            calc_results[r] = calc_result

        # Check that the maximum difference between pixel beams/direct calculation
        # cases is no more than 5%. This shows the direct calculation of the beam
        # tracks the pixel beam interpolation. They won't be exactly the same.
        np.testing.assert_allclose(pix_results, calc_results, rtol=0.05)

        # Check that rotations 0 and 180 produce the same values.
        assert pix_results[0] == pytest.approx(pix_results[180], abs=1e-8)
        assert calc_results[0] == pytest.approx(calc_results[180], abs=1e-8)

        # Check that the values are not all the same. Shouldn't be, due to
        # elliptic beam.
        assert np.min(pix_results) != pytest.approx(np.max(pix_results), abs=0.1)
        assert np.min(calc_results) != pytest.approx(np.max(calc_results), abs=0.1)

        # Check that attempting to use GPU with Polybeam raises an error.
        with pytest.raises(RuntimeError if HAVE_GPU else ImportError):
            run_sim(r, use_pixel_beams=False, use_gpu=True)

        # Check that attempting to use GPU with MPI raises an error.
        with pytest.raises(RuntimeError):
            run_sim(r, use_gpu=True, use_mpi=True)

    def test_perturbed_polybeam_polarized(self):

        # Calculate all polarizations for a beam rotated by 12 degrees
        r = 12.0  # degrees
        calc_result_ee = run_sim(r, use_pixel_beams=False, use_pol=True, pol="ee")
        calc_result_nn = run_sim(r, use_pixel_beams=False, use_pol=True, pol="nn")
        calc_result_en = run_sim(r, use_pixel_beams=False, use_pol=True, pol="en")
        calc_result_ne = run_sim(r, use_pixel_beams=False, use_pol=True, pol="ne")

        # Calculate with unrotated, unpolarized beam
        calc_result_unpol = run_sim(r, use_pixel_beams=False, use_pol=False, pol="ee")

        # Check that ee + nn == unpol, since V_pI = 0.5 * (V_nn + V_ee)
        unpol = 0.5 * (calc_result_ee + calc_result_nn)
        np.testing.assert_almost_equal(unpol, calc_result_unpol, decimal=7)

        # Check that ne and en have valid values
        assert np.all(~np.isnan(calc_result_en))
        assert np.all(~np.isinf(calc_result_en))
        assert np.all(~np.isnan(calc_result_ne))
        assert np.all(~np.isinf(calc_result_ne))
