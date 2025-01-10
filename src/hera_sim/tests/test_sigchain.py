import numpy as np
import pytest
import uvtools
from astropy import constants, units
from pyuvdata import UniformBeam, UVBeam

import hera_sim
from hera_sim import DATA_PATH, foregrounds, noise, sigchain
from hera_sim.interpolators import Bandpass, Beam
from hera_sim.io import empty_uvdata


@pytest.fixture(scope="function")
def fqs():
    return np.linspace(0.1, 0.2, 100, endpoint=False)


@pytest.fixture(scope="function")
def lsts():
    return np.linspace(0, 2 * np.pi, 200)


@pytest.fixture(scope="function")
def Tsky_mdl():
    return noise.HERA_Tsky_mdl["xx"]


@pytest.fixture(scope="function")
def Tsky(Tsky_mdl, lsts, fqs):
    return Tsky_mdl(lsts, fqs)


@pytest.fixture(scope="function")
def bl_vec():
    return np.array([50, 0, 0], dtype=float)


@pytest.fixture(scope="function")
def vis(lsts, fqs, bl_vec, Tsky_mdl):
    beamfile = DATA_PATH / "HERA_H1C_BEAM_POLY.npy"
    omega_p = Beam(beamfile)
    return (
        foregrounds.diffuse_foreground(
            lsts,
            fqs,
            bl_vec,
            Tsky_mdl=Tsky_mdl,
            omega_p=omega_p,
            delay_filter_kwargs={"delay_filter_type": "gauss"},
        )
        + 20
    )


@pytest.fixture(scope="function")
def dlys(fqs):
    return uvtools.utils.fourier_freqs(fqs)


@pytest.fixture(scope="function")
def vfft(vis):
    return uvtools.utils.FFT(vis, axis=1)


def test_gen_bandpass():
    fqs = np.linspace(0.1, 0.2, 1024, endpoint=False)
    g = sigchain.gen_bandpass(fqs, [1, 2], gain_spread=0)
    assert 1 in g
    assert 2 in g
    assert g[1].size == fqs.size
    assert np.all(g[1] == g[2])
    g = sigchain.gen_bandpass(fqs, list(range(10)), 0.2, rng=np.random.default_rng(0))
    assert not np.all(g[1] == g[2])


def test_gen_delay_phs():
    fqs = np.linspace(0.12, 0.18, 1024, endpoint=False)
    phs = sigchain.gen_delay_phs(fqs, [1, 2], dly_rng=(0, 20))
    assert len(phs) == 2
    assert 1 in phs
    assert 2 in phs
    assert np.allclose(np.abs(phs[1]), 1)
    p = np.polyfit(fqs, np.unwrap(np.angle(phs[1])), deg=1)
    assert np.any(np.isclose(p[-1] % (2 * np.pi), (0, 2 * np.pi), atol=1e-2))
    assert p[0] <= 20 * 2 * np.pi
    assert p[0] >= 0


def test_gen_gains():
    fqs = np.linspace(0.12, 0.18, 1024, endpoint=False)
    g = sigchain.gen_gains(fqs, [1, 2], gain_spread=0, dly_rng=(10, 20))
    assert np.allclose(np.abs(g[1]), np.abs(g[2]), rtol=1e-5)
    for i in g:
        p = np.polyfit(fqs, np.unwrap(np.angle(g[i])), deg=1)
        assert np.any(np.isclose(p[-1] % (2 * np.pi), (0, 2 * np.pi), atol=1e-2))
        assert p[0] <= 20 * 2 * np.pi
        assert p[0] >= 10 * 2 * np.pi


def test_apply_gains():
    fqs = np.linspace(0.12, 0.18, 1024, endpoint=False)
    vis = np.ones((100, fqs.size), dtype=complex)
    g = sigchain.gen_gains(fqs, [1, 2], gain_spread=0, dly_rng=(10, 10))
    gvis = sigchain.apply_gains(vis, g, (1, 2))
    assert np.allclose(np.angle(gvis), 0, rtol=1e-5)


@pytest.mark.parametrize("taper", ["tanh", "bh", "custom", "callable"])
def test_bandpass_with_taper(fqs, taper):
    taper_kwds = None
    if taper == "custom":
        taper = np.linspace(0, 1, fqs.size)
    elif taper == "callable":

        def taper(freqs):
            return np.sin(np.pi * (freqs - freqs.mean() / freqs.max()))

    elif taper == "tanh":
        taper_kwds = dict(x_min=fqs[10], x_max=fqs[-10])

    base_bandpass = sigchain.gen_gains(fqs, [0], gain_spread=0, dly_rng=(0, 0))[0]
    bandpass = sigchain.gen_gains(
        fqs, [0], gain_spread=0, dly_rng=(0, 0), taper=taper, taper_kwds=taper_kwds
    )[0]
    assert not np.allclose(base_bandpass, bandpass)


def test_bandpass_bad_taper(fqs):
    with pytest.raises(ValueError, match="Unsupported choice of taper."):
        sigchain.gen_gains(fqs, [0], taper=13)


def test_reflection_gains_correct_delays(fqs, vis, dlys):
    # introduce a cable reflection into the autocorrelation
    rng = np.random.default_rng(0)
    gains = sigchain.gen_reflection_gains(
        fqs, [0], amp=[1e-1], dly=[300], phs=[1], rng=rng
    )
    outvis = sigchain.apply_gains(vis, gains, [0, 0])
    ovfft = uvtools.utils.FFT(outvis, axis=1, taper="blackman-harris")

    # assert reflection is at +300 ns and check its amplitude
    select = dlys > 200
    assert np.allclose(
        dlys[select][np.argmax(np.mean(np.abs(ovfft), axis=0)[select])], 300, atol=1e-7
    )
    select = np.argmin(np.abs(dlys - 300))
    m = np.mean(np.abs(ovfft), axis=0)
    assert np.isclose(m[select] / m[np.argmin(np.abs(dlys))], 1e-1, atol=1e-2)

    # assert also reflection at -300 ns
    select = dlys < -200
    assert np.allclose(
        dlys[select][np.argmax(np.mean(np.abs(ovfft), axis=0)[select])], -300, atol=1e-7
    )
    select = np.argmin(np.abs(dlys - -300))
    m = np.mean(np.abs(ovfft), axis=0)
    assert np.isclose(m[select] / m[np.argmin(np.abs(dlys))], 1e-1, atol=1e-2)


def test_reflection_gains_reshape(fqs):
    # test reshaping into Ntimes
    amp = np.linspace(1e-2, 1e-3, 3)
    gains = sigchain.gen_reflection_gains(fqs, [0], amp=[amp], dly=[300], phs=[1])
    assert gains[0].shape == (3, 100)


def test_reflection_gains_evolution_single_time(fqs):
    # test frequency evolution with one time
    amp = np.linspace(1e-2, 1e-3, 100).reshape(1, -1)
    gains = sigchain.gen_reflection_gains(fqs, [0], amp=[amp], dly=[300], phs=[1])
    assert gains[0].shape == (1, 100)


def test_reflection_gains_evolution_many_times(fqs):
    # now test with multiple times
    amp = np.repeat(np.linspace(1e-2, 1e-3, 100).reshape(1, -1), 10, axis=0)
    gains = sigchain.gen_reflection_gains(fqs, [0], amp=[amp], dly=[300], phs=[1])
    assert gains[0].shape == (10, 100)


def test_reflection_gains_exception(fqs):
    # exception
    amp = np.linspace(1e-2, 1e-3, 2).reshape(1, -1)
    with pytest.raises(AssertionError):
        sigchain.gen_reflection_gains(fqs, [0], amp=[amp], dly=[300], phs=[1])


def test_reflection_spectrum():
    # Turns out testing that things check out with the jitter isn't so
    # straightforward... so we'll do this with no jitter.
    n_copies = 5
    amp_range = (-3, -7)
    dly_range = (100, 800)
    amp_jitter = 0
    dly_jitter = 0
    amp_logbase = 5
    amplitudes = np.logspace(*amp_range, n_copies, base=amp_logbase)
    delays = np.linspace(*dly_range, n_copies)  # All dlys are multiples of 5
    reflections = sigchain.ReflectionSpectrum(
        n_copies=n_copies,
        amp_range=amp_range,
        dly_range=dly_range,
        amp_jitter=amp_jitter,
        dly_jitter=dly_jitter,
        amp_logbase=amp_logbase,
    )

    # This is kind of backwards, but I want to specify the delays
    dlys = np.arange(-1000, 1001, 5)
    fqs = uvtools.utils.fourier_freqs(dlys)
    fqs += 0.1 - fqs.min()  # Range from 100 MHz to whatever the upper bound is
    reflections = reflections(fqs, range(100), rng=np.random.default_rng(0))
    reflections = np.vstack(list(reflections.values()))
    spectra = np.abs(uvtools.utils.FFT(reflections, axis=1))
    spectra = spectra / spectra.max(axis=1).reshape(-1, 1)
    dly_inds = np.argwhere(dlys[:, None] - delays[None, :] == 0)[:, 0].astype(int)
    for amp, ind in zip(amplitudes, dly_inds.flat):
        assert np.allclose(spectra[:, ind], amp, rtol=0.01)


def test_cross_coupling_xtalk_correct_delay(fqs, dlys, Tsky):
    # introduce a cross reflection at a single delay
    outvis = sigchain.gen_cross_coupling_xtalk(fqs, Tsky, amp=1e-2, dly=300, phs=1)
    ovfft = uvtools.utils.FFT(outvis, axis=1, taper="blackman-harris")

    # take covariance across time and assert delay 300 is highly covariant
    # compared to neighbors
    cov = np.cov(ovfft.T)
    mcov = np.mean(np.abs(cov), axis=0)
    select = np.argsort(np.abs(dlys - 300))[:10]
    assert np.isclose(dlys[select][np.argmax(mcov[select])], 300, atol=1e-7)
    # inspect for yourself: plt.matshow(np.log10(np.abs(cov)))


def test_cross_coupling_xtalk_conj_correct_delay(fqs, dlys, Tsky):
    # conjugate it and assert it shows up at -300
    outvis = sigchain.gen_cross_coupling_xtalk(
        fqs, Tsky, amp=1e-2, dly=300, phs=1, conj=True
    )
    ovfft = uvtools.utils.FFT(outvis, axis=1, taper="blackman-harris")
    cov = np.cov(ovfft.T)
    mcov = np.mean(np.abs(cov), axis=0)
    select = np.argsort(np.abs(dlys - -300))[:10]
    assert np.isclose(dlys[select][np.argmax(mcov[select])], -300, atol=1e-7)


def test_cross_coupling_xtalk_phase_stability(fqs, dlys, Tsky):
    # assert its phase stable across time
    outvis = sigchain.gen_cross_coupling_xtalk(
        fqs, Tsky, amp=1e-2, dly=300, phs=1, conj=True
    )
    ovfft = uvtools.utils.FFT(outvis, axis=1, taper="blackman-harris")
    select = np.argmin(np.abs(dlys - -300))
    assert np.allclose(np.angle(ovfft[:, select]), -1, atol=1e-4, rtol=1e-4)


def test_amp_jitter():
    ants = range(10000)
    amp = 5
    amp_jitter = 0.1
    rng = np.random.default_rng(0)
    jittered_amps = sigchain.Reflections._complete_params(
        ants, amp=amp, amp_jitter=amp_jitter, rng=rng
    )[0]
    assert np.isclose(jittered_amps.mean(), amp, rtol=0.05)
    assert np.isclose(jittered_amps.std(), amp * amp_jitter, rtol=0.05)


def test_dly_jitter():
    ants = range(10000)
    dly = 500
    dly_jitter = 20
    rng = np.random.default_rng(0)
    jittered_dlys = sigchain.Reflections._complete_params(
        ants, dly=dly, dly_jitter=dly_jitter, rng=rng
    )[1]
    assert np.isclose(jittered_dlys.mean(), dly, rtol=0.05)
    assert np.isclose(jittered_dlys.std(), dly_jitter, rtol=0.05)


def test_cross_coupling_spectrum(fqs, dlys, Tsky):
    n_copies = 5
    amp_range = (-2, -5)
    dly_range = (50, 450)
    xtalk_spectrum = sigchain.CrossCouplingSpectrum(
        n_copies=n_copies,
        amp_range=amp_range,
        dly_range=dly_range,
        symmetrize=True,
        rng=np.random.default_rng(0),
    )
    amplitudes = np.logspace(*amp_range, n_copies)
    delays = np.linspace(*dly_range, n_copies)
    xtalk = xtalk_spectrum(freqs=fqs, autovis=Tsky)
    Tsky_avg = np.abs(
        uvtools.utils.FFT(Tsky, axis=1, taper="bh7")[:, np.argmin(np.abs(dlys))]
    )  # Take the average this way to avoid numerical oddities with FFTs
    xt_fft = uvtools.utils.FFT(xtalk, axis=1, taper="bh7")
    # Check that there are spikes at expected delays w/ expected amplitudes.
    for dly, amp in zip(delays, amplitudes):
        # The crosstalk should be a scaled down, phased version of zero delay mode.
        dly_ind = np.argmin(np.abs(dlys - dly))
        neg_dly_ind = np.argmin(np.abs(dlys - -dly))
        for ind in (dly_ind, neg_dly_ind):
            ratio = np.abs(xt_fft[:, ind]) / Tsky_avg
            assert np.allclose(ratio, amp, rtol=0.01)


def test_over_air_cross_coupling(Tsky_mdl, lsts):
    # Setup various parameters. To make it easy, only have cable lengths vary.
    n_copies = 5
    emitter_pos = np.array([0, 0, 0])
    # Both antennas have same distance to emitter.
    antpos = {0: np.array([30, 40, 0]), 1: np.array([50, 0, 0])}
    cable_delays = {0: 400, 1: 600}
    max_delay = 1500
    amp_decay_fac = 1e-2
    base_amp = 2e-5
    amp_norm = 100
    amp_slope = -2.3
    fqs = np.linspace(0.1, 0.2, 1024, endpoint=False)
    dlys = uvtools.utils.fourier_freqs(fqs)
    Tsky = Tsky_mdl(lsts, fqs)

    # Calculate the expected delays/amplitudes
    base_delay = 50 / constants.c.to("m/ns").value
    pos_dlys = np.linspace(base_delay + cable_delays[1], max_delay, n_copies)
    neg_dlys = -np.linspace(base_delay + cable_delays[0], max_delay, n_copies)
    start_amp = np.log10(base_amp * (50 / amp_norm) ** amp_slope)
    end_amp = start_amp + np.log10(amp_decay_fac)
    amplitudes = np.logspace(start_amp, end_amp, n_copies)
    all_dlys = np.concatenate([neg_dlys, pos_dlys])
    all_amps = np.concatenate([amplitudes, amplitudes])
    Tsky_avg = np.abs(
        uvtools.utils.FFT(Tsky, axis=1, taper="bh7")[:, np.argmin(np.abs(dlys))]
    )
    gen_xtalk = sigchain.OverAirCrossCoupling(
        base_amp=base_amp,
        amp_norm=amp_norm,
        amp_slope=amp_slope,
        n_copies=n_copies,
        emitter_pos=emitter_pos,
        cable_delays=cable_delays,
        max_delay=max_delay,
        amp_decay_fac=amp_decay_fac,
        rng=np.random.default_rng(0),
    )
    xtalk = gen_xtalk(fqs, (0, 1), antpos, Tsky, Tsky)
    xt_fft = uvtools.utils.FFT(xtalk, axis=1, taper="bh7")
    for dly, amp in zip(all_dlys, all_amps):
        dly_ind = np.argmin(np.abs(dlys - dly))
        ratio = np.abs(xt_fft[:, dly_ind]) / Tsky_avg
        assert np.allclose(ratio, amp, rtol=0.05)


def test_over_air_xtalk_skips_autos(fqs, Tsky):
    gen_xtalk = sigchain.OverAirCrossCoupling()
    xtalk = gen_xtalk(fqs, (0, 0), range(3), Tsky, Tsky)
    assert xtalk.shape == Tsky.shape and np.all(xtalk == 0)


@pytest.mark.parametrize("use_numba", [False, True])
def test_mutual_coupling(use_numba):
    hera_sim.defaults.deactivate()
    array_layout = {
        0: np.array([0, 0, 0]),
        1: np.array([50, 0, 0]),
        2: np.array([0, 100, 0]),
    }
    uvdata = empty_uvdata(
        Nfreqs=300,
        start_freq=140e6,
        channel_width=100e3,
        Ntimes=300,
        start_time=2458091.23423,
        integration_time=10.7,
        array_layout=array_layout,
        telescope_location=hera_sim.io.HERA_LAT_LON_ALT,
    )

    # Mock up visibilities that are localized in delay/frate in the same
    # way that foregrounds are localized. First, some prep work.
    freqs = uvdata.freq_array.squeeze()
    times = np.unique(uvdata.time_array) * units.day.to("s")
    freq_mesh, time_mesh = np.meshgrid(freqs, times - times.mean())
    omega0 = units.cycle.to("rad") / units.sday.to("s")
    omega = np.array([0, 0, omega0])
    delay_width = 50e-9
    ref_freq = freqs.mean()
    enu_antpos = dict(zip(*uvdata.get_ENU_antpos()[::-1]))
    ecef_antpos = dict(zip(uvdata.antenna_numbers, uvdata.antenna_positions))

    # We'll want to keep track of where we expect the features to show up.
    fringe_rates = {}
    delays = {}

    # Now let's actually mock up the visibilities.
    for ai, aj in uvdata.get_antpairs():
        if uvdata.antpair2ind(ai, aj) is None:
            continue
        ecef_bl = ecef_antpos[aj] - ecef_antpos[ai]
        enu_bl = enu_antpos[aj] - enu_antpos[ai]
        dbdt = np.linalg.norm(np.cross(ecef_bl, omega))
        sign = np.sign(np.round(enu_bl[0], 3))
        frates = sign * freq_mesh * dbdt / constants.c.si.value
        blt_inds = uvdata.antpair2ind(ai, aj)

        # Make visibilities compact in delay/freq, and localized in fringe-rate.
        uvdata.data_array[blt_inds, :, 0] = np.exp(
            -((delay_width * (freq_mesh - ref_freq)) ** 2)
        ) * np.exp(2j * np.pi * frates[None, :] * time_mesh)

        # Let's also track the delays/fringe-rates
        delays[(ai, aj)] = np.linalg.norm(enu_bl) / constants.c.to("m/ns").value
        delays[(aj, ai)] = delays[(ai, aj)]
        mean_frate = sign * dbdt * np.mean(freqs) / constants.c.si.value
        fringe_rates[(ai, aj)] = mean_frate * units.Hz.to("mHz")
        fringe_rates[(aj, ai)] = -fringe_rates[(ai, aj)]

    # Take note of the visibility amplitudes for later comparison.
    vis_amps = {}
    for (ai, aj, _pol), vis in uvdata.antpairpol_iter():
        vis_fft = uvtools.utils.FFT(
            uvtools.utils.FFT(vis, axis=0, taper="bh"), axis=1, taper="bh"
        )
        vis_amps[(ai, aj)] = np.max(np.abs(vis_fft))
        vis_amps[(aj, ai)] = vis_amps[(ai, aj)]

    # Set the coupling parameters so that it's simple to check the amplitudes.
    refl_amp = 1

    # Actually simulate the coupling.
    mutual_coupling = sigchain.MutualCoupling(
        uvbeam=UniformBeam(),
        ant_1_array=uvdata.ant_1_array,
        ant_2_array=uvdata.ant_2_array,
        pol_array=uvdata.polarization_array,
        array_layout=dict(zip(*uvdata.get_ENU_antpos()[::-1])),
        reflection=np.ones(uvdata.Nfreqs) * refl_amp,
        omega_p=constants.c.si.value / uvdata.freq_array,
    )
    uvdata.data_array += mutual_coupling(
        freqs=freqs / 1e9, visibilities=uvdata.data_array, use_numba=use_numba
    )

    # Now run the checks.
    delay_ns = uvtools.utils.fourier_freqs(freqs) * units.s.to("ns")
    frate_mHz = uvtools.utils.fourier_freqs(times) * units.Hz.to("mHz")
    for (ai, aj, _pol), vis in uvdata.antpairpol_iter():
        vis_fft = uvtools.utils.FFT(
            uvtools.utils.FFT(vis, axis=0, taper="bh"), axis=1, taper="bh"
        )
        for ak in uvdata.antenna_numbers:
            dly_ik = delays[(ai, ak)]
            dly_kj = -delays[(ak, aj)]
            frate_ik = fringe_rates[(ai, ak)]
            frate_kj = fringe_rates[(ak, aj)]

            # Check that the coupling was performed correctly.
            if aj != ak:
                ik_ind = (
                    np.argmin(np.abs(frate_ik - frate_mHz)),
                    np.argmin(np.abs(dly_kj - delay_ns)),
                )
                bl_len = np.linalg.norm(enu_antpos[aj] - enu_antpos[ak])
                exp_amp = vis_amps[(ai, ak)] * refl_amp / bl_len
                actual_amp = np.abs(vis_fft[ik_ind])
                assert np.isclose(exp_amp, actual_amp, atol=1e-7, rtol=0.05)

            if ai != ak:
                kj_ind = (
                    np.argmin(np.abs(frate_kj - frate_mHz)),
                    np.argmin(np.abs(dly_ik - delay_ns)),
                )
                bl_len = np.linalg.norm(enu_antpos[ak] - enu_antpos[ai])
                exp_amp = vis_amps[(ak, aj)] * refl_amp / bl_len
                actual_amp = np.abs(vis_fft[kj_ind])
                assert np.isclose(exp_amp, actual_amp, atol=1e-7, rtol=0.05)


@pytest.fixture
def sample_uvdata():
    hera_sim.defaults.set("debug")
    uvdata = empty_uvdata()
    hera_sim.defaults.deactivate()
    return uvdata


@pytest.fixture
def sample_coupling(sample_uvdata):
    coupling = sigchain.MutualCoupling(
        uvbeam=UniformBeam(),
        ant_1_array=sample_uvdata.ant_1_array,
        ant_2_array=sample_uvdata.ant_2_array,
        pol_array=sample_uvdata.polarization_array,
        array_layout=dict(zip(*sample_uvdata.get_ENU_antpos()[::-1])),
    )
    return coupling


@pytest.fixture
def uvbeam(tmp_path):
    beam_fn = str(tmp_path / "test_beam.fits")
    beam = UniformBeam()

    # Setup some things needed to mock up the UVBeam object
    az = np.linspace(0, 2 * np.pi, 100)
    za = np.linspace(np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6, 15)
    freqs = np.linspace(99e6, 101e6, 20)
    uvb = beam.to_uvbeam(
        freq_array=freqs, beam_type='efield', axis1_array=az, axis2_array=za
    )

    # Finally, write it to disk.
    uvb.write_beamfits(beam_fn)
    return beam_fn


def test_mutual_coupling_with_uvbeam(uvbeam, sample_uvdata, sample_coupling):
    rng = np.random.default_rng(0)
    data = rng.normal(size=sample_uvdata.data_array.shape) + 0j
    sample_uvdata.data_array[...] = data[...]
    sample_uvdata.data_array += sample_coupling(
        freqs=sample_uvdata.freq_array.squeeze() / 1e9,
        visibilities=sample_uvdata.data_array,
        reflection=np.ones(sample_uvdata.Nfreqs) * 10,
        uvbeam=uvbeam,
    )
    assert not np.any(np.isclose(data, sample_uvdata.data_array))


@pytest.mark.parametrize("omega_p", [None, "callable"])
@pytest.mark.parametrize("reflection", [None, "callable"])
def test_mutual_coupling_input_types(
    omega_p, reflection, sample_uvdata, sample_coupling
):
    def func(freqs):
        return np.ones_like(freqs)

    rng = np.random.default_rng(0)
    omega_p = func if omega_p == "callable" else None
    reflection = func if reflection == "callable" else None
    data = rng.normal(size=sample_uvdata.data_array.shape) + 0j
    sample_uvdata.data_array += data
    sample_uvdata.data_array += sample_coupling(
        freqs=sample_uvdata.freq_array.squeeze() / 1e9,
        visibilities=sample_uvdata.data_array,
        reflection=reflection,
        omega_p=omega_p,
    )
    assert not np.any(np.isclose(data, sample_uvdata.data_array))


def test_mutual_coupling_bad_ants(sample_uvdata, sample_coupling):
    full_array = dict(zip(*sample_uvdata.get_ENU_antpos()[::-1]))
    bad_array = {ant: full_array[ant] for ant in range(len(full_array) - 1)}
    with pytest.raises(ValueError, match="Full array layout not provided."):
        _ = sample_coupling(
            freqs=sample_uvdata.freq_array.squeeze() / 1e9,
            visibilities=sample_uvdata.data_array,
            array_layout=bad_array,
        )


@pytest.mark.parametrize("isbad", ["reflection", "omega_p"])
def test_mutual_coupling_bad_feed_params(isbad, sample_uvdata, sample_coupling):
    kwargs = {"omega_p": None, "reflection": None}
    kwargs[isbad] = np.ones(sample_uvdata.Nfreqs + 1)
    with pytest.raises(ValueError, match="the wrong shape"):
        _ = sample_coupling(
            freqs=sample_uvdata.freq_array.squeeze() / 1e9,
            visibilities=sample_uvdata.data_array,
            **kwargs,
        )


@pytest.fixture(scope="function")
def freqs():
    return np.linspace(0.1, 0.2, 1024)


@pytest.fixture(scope="function")
def times():
    return np.linspace(0, 1, 500)


@pytest.fixture(scope="function")
def delays():
    dlys = {0: 20}  # ns
    return dlys


@pytest.fixture(scope="function")
def bp_poly():
    return Bandpass(datafile="HERA_H1C_BANDPASS.npy")


@pytest.fixture(scope="function")
def gains(freqs, delays, bp_poly):
    dly = delays[0]
    gain_dict = sigchain.gen_gains(
        freqs, ants=delays, gain_spread=0, dly_rng=(dly, dly), bp_poly=bp_poly
    )
    # Add a phase offset to the gains
    gain_dict[0] *= np.exp(1j * np.pi / 4)
    return gain_dict


@pytest.fixture(scope="function")
def delay_phases(freqs, delays):
    dly = delays[0]
    return (2 * np.pi * freqs * dly) % (2 * np.pi) - np.pi


@pytest.fixture(scope="function")
def phase_offsets(gains, delay_phases):
    phases = np.angle(gains[0])
    return (phases - delay_phases) % (2 * np.pi) - np.pi


@pytest.fixture(scope="function")
def fringe_rates(times):
    return uvtools.utils.fourier_freqs(times * units.h.to("s"))


@pytest.fixture(scope="function")
def fringe_keys(fringe_rates):
    pos_fringe_key = np.argwhere(
        fringe_rates > 5 * np.mean(np.diff(fringe_rates))
    ).flatten()
    neg_fringe_key = np.argwhere(
        fringe_rates < -5 * np.mean(np.diff(fringe_rates))
    ).flatten()
    return (pos_fringe_key, neg_fringe_key)


def varies_as_expected(quantity, vary_freq, fringe_key, fringe_rates):
    quantity_fft = uvtools.utils.FFT(quantity, axis=0, taper="bh7")
    peak_index = np.argmax(np.abs(quantity_fft[fringe_key]))
    return np.isclose(
        vary_freq, np.abs(fringe_rates[fringe_key][peak_index]), rtol=0.01
    )


def test_vary_gain_amp_linear(gains, times):
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        parameter="amp",
        variation_ref_time=times.mean(),
        variation_timescale=2 * (times[-1] - times[0]),
        variation_amp=0.1,
        variation_mode="linear",
    )[0]

    # Check that the original value is at the center time.
    assert np.allclose(
        varied_gain[np.argmin(np.abs(times - times.mean())), :], gains[0], rtol=0.001
    )

    # Check that the variation amount is as expected.
    assert np.allclose(varied_gain[-1, :] / gains[0], 1.1)
    assert np.allclose(varied_gain[0, :] / gains[0], 0.9)


def test_vary_gain_amp_sinusoidal(gains, times, fringe_rates, fringe_keys):
    vary_timescale = 30 * units.s.to("hour")
    vary_freq = 1 / (vary_timescale * units.h.to("s"))
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        parameter="amp",
        variation_timescale=vary_timescale,
        variation_amp=0.5,
        variation_mode="sinusoidal",
    )[0]

    # Check that there's variation at the expected timescale.
    for fringe_key in fringe_keys:
        assert varies_as_expected(
            varied_gain[:, 150], vary_freq, fringe_key, fringe_rates
        )


def test_vary_gain_amp_noiselike(gains, times):
    vary_amp = 0.1
    rng = np.random.default_rng(0)
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        parameter="amp",
        variation_mode="noiselike",
        variation_amp=vary_amp,
        rng=rng,
    )[0]

    # Check that the mean gain amplitude is the original gain amplitude.
    gain_avg = np.mean(np.abs(varied_gain), axis=0)
    assert np.allclose(gain_avg, np.abs(gains[0]), rtol=0.1)

    # Check that the spread in gain amplitudes is as expected.
    standard_deviations = np.std(np.abs(varied_gain), axis=0)
    assert np.allclose(standard_deviations, vary_amp * np.abs(gains[0]), rtol=0.1)


def test_vary_gain_phase_linear(gains, times, phase_offsets, delay_phases):
    vary_amp = 0.1
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        parameter="phs",
        variation_mode="linear",
        variation_amp=vary_amp,
        variation_ref_time=times.mean(),
        variation_timescale=2 * (times[-1] - times[0]),
    )[0]

    varied_phases = np.angle(varied_gain)
    varied_phase_offsets = (varied_phases - delay_phases) % (2 * np.pi) - np.pi

    # Check that the original phase offset occurs at the central time.
    assert np.allclose(
        phase_offsets,
        varied_phase_offsets[np.argmin(np.abs(times - times.mean()))],
        rtol=0.01,
    )

    # Check that the variation amount is as expected.
    assert np.allclose(varied_phase_offsets[-1] - phase_offsets, vary_amp, rtol=0.01)
    assert np.allclose(varied_phase_offsets[0] - phase_offsets, -vary_amp, rtol=0.01)


def test_vary_gain_phase_sinusoidal(
    gains, times, delay_phases, fringe_rates, fringe_keys
):
    timescale = 1 * units.min.to("h")
    vary_freq = 1 / (timescale * units.h.to("s"))
    vary_amp = 0.1
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        parameter="phs",
        variation_mode="sinusoidal",
        variation_amp=vary_amp,
        variation_timescale=timescale,
    )[0]

    varied_phases = np.angle(varied_gain)
    varied_phase_offsets = (varied_phases - delay_phases) % (2 * np.pi) - np.pi

    # Check that there's variation at the expected timescale.
    for fringe_key in fringe_keys:
        assert varies_as_expected(
            varied_phase_offsets[:, 150], vary_freq, fringe_key, fringe_rates
        )


def test_vary_gain_phase_noiselike(gains, times, delay_phases, phase_offsets):
    vary_amp = 0.1
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        parameter="phs",
        variation_mode="noiselike",
        variation_amp=vary_amp,
        rng=np.random.default_rng(0),
    )[0]

    varied_phases = np.angle(varied_gain)
    varied_phase_offsets = (varied_phases - delay_phases) % (2 * np.pi) - np.pi

    # Check that the mean phase offset is close to the original phase offset.
    mean_offset = np.mean(varied_phase_offsets, axis=0)
    assert np.allclose(mean_offset, phase_offsets, rtol=0.1)

    # Check that the spread in phase offsets is as expected.
    offset_std = np.std(varied_phase_offsets, axis=0)
    assert np.allclose(offset_std, vary_amp, rtol=0.1)


def test_vary_gain_delay_linear(gains, times, freqs, delays):
    vary_amp = 0.5
    dly = delays[0]
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        freqs=freqs,
        delays=delays,
        parameter="dly",
        variation_amp=vary_amp,
        variation_mode="linear",
    )[0]

    varied_gain_fft = uvtools.utils.FFT(varied_gain, axis=1, taper="bh7")
    dlys = uvtools.utils.fourier_freqs(freqs)
    min_dly = dlys[np.argmax(np.abs(varied_gain_fft[0, :]))]
    max_dly = dlys[np.argmax(np.abs(varied_gain_fft[-1, :]))]
    center_index = np.argmin(np.abs(times - times.mean()))
    mid_dly = dlys[np.argmax(np.abs(varied_gain_fft[center_index, :]))]

    # Check that the delays vary as expected.
    assert np.isclose(min_dly, (1 - vary_amp) * dly, rtol=0.01)
    assert np.isclose(mid_dly, dly, rtol=0.01)
    assert np.isclose(max_dly, (1 + vary_amp) * dly, rtol=0.01)


def test_vary_gain_delay_sinusoidal(
    gains, times, freqs, delays, fringe_rates, fringe_keys
):
    vary_amp = 0.5
    timescale = 30 * units.s.to("h")
    vary_freq = 1 / (timescale * units.h.to("s"))
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        freqs=freqs,
        delays=delays,
        parameter="dly",
        variation_amp=vary_amp,
        variation_mode="sinusoidal",
        variation_timescale=timescale,
    )[0]

    # Determine the bandpass delay at each time.
    varied_gain_fft = uvtools.utils.FFT(varied_gain, axis=1, taper="bh7")
    dlys = uvtools.utils.fourier_freqs(freqs)
    gain_delays = np.array([dlys[np.argmax(np.abs(gain))] for gain in varied_gain_fft])

    # Check that delays vary at the expected timescale.
    for fringe_key in fringe_keys:
        assert varies_as_expected(gain_delays, vary_freq, fringe_key, fringe_rates)


def test_vary_gain_delay_noiselike(gains, times, freqs, delays):
    vary_amp = 0.5
    varied_gain = sigchain.vary_gains_in_time(
        gains=gains,
        times=times,
        freqs=freqs,
        delays=delays,
        parameter="dly",
        variation_amp=vary_amp,
        variation_mode="noiselike",
        rng=np.random.default_rng(0),
    )[0]

    # Determine the bandpass delay at each time.
    dlys = uvtools.utils.fourier_freqs(freqs)
    varied_gain_fft = uvtools.utils.FFT(varied_gain, axis=1, taper="bh7")
    gain_delays = np.array([dlys[np.argmax(np.abs(gain))] for gain in varied_gain_fft])

    # Check that the delays vary as expected.
    assert np.isclose(gain_delays.mean(), delays[0], rtol=0.1)
    assert np.isclose(gain_delays.std(), vary_amp * delays[0], rtol=0.2)


def test_vary_gains_exception_bad_times():
    with pytest.raises(TypeError) as err:
        sigchain.vary_gains_in_time(gains={}, times=42)
    assert err.value.args[0] == "times must be an array of real numbers."


def test_vary_gains_exception_complex_times():
    with pytest.raises(TypeError) as err:
        sigchain.vary_gains_in_time(gains={}, times=np.ones(10, dtype=complex))
    assert err.value.args[0] == "times must be an array of real numbers."


def test_vary_gains_exception_bad_gains():
    with pytest.raises(TypeError) as err:
        sigchain.vary_gains_in_time(gains=[], times=[1, 2, 3])
    assert err.value.args[0] == "gains must be provided as a dictionary."


def test_vary_gains_exception_bad_param():
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(gains={}, times=[1, 2], parameter="bad choice")
    assert "parameter must be one of" in err.value.args[0]


def test_vary_gains_exception_bad_gain_shapes():
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains={0: np.ones(10), 1: np.ones(15)}, times=[1, 2, 3]
        )
    assert err.value.args[0] == "Gains must all have the same shape."


def test_vary_gains_exception_insufficient_delay_info(gains, times):
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains=gains, times=times, parameter="dly", freqs=freqs
        )
    assert "you must provide both" in err.value.args[0]


def test_vary_gains_exception_insufficient_freq_info(gains, times):
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains=gains, times=times, parameter="dly", delays=delays
        )
    assert "you must provide both" in err.value.args[0]


def test_vary_gains_exception_mismatched_keys(times, freqs):
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains={0: 0, 1: 1},
            times=times,
            parameter="dly",
            freqs=freqs,
            delays={1: 1, 2: 2},
        )
    assert err.value.args[0] == "Delays and gains must have the same keys."


def test_vary_gains_exception_bad_gain_waterfall_shape(times, freqs):
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains={0: np.ones((10, 15))},
            times=times,
            freqs=freqs,
            delays={0: 20},
            parameter="dly",
        )
    assert err.value.args[0] == "Gain waterfalls must have shape (Ntimes, Nfreqs)."


def test_vary_gains_exception_bad_gain_spectrum_shape(times, freqs):
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains={0: np.ones(37)},
            times=times,
            freqs=freqs,
            delays={0: 20},
            parameter="dly",
        )
    assert "Gain spectra must be" in err.value.args[0]


def test_vary_gains_exception_too_many_dimensions(times, freqs, delays):
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains={0: np.ones((10, 10, 10))},
            times=times,
            freqs=freqs,
            delays=delays,
            parameter="dly",
        )
    assert "must be at most 2-dimensional." in err.value.args[0]


def test_vary_gains_exception_not_enough_parameters(gains, times):
    with pytest.raises(ValueError) as err:
        sigchain.vary_gains_in_time(
            gains=gains,
            times=times,
            parameter="amp",
            variation_mode=("linear", "sinusoidal"),
            variation_amp=1,
        )
    assert "does not have the same number of entries" in err.value.args[0]


def test_vary_gains_exception_bad_variation_mode(gains, times):
    with pytest.raises(NotImplementedError) as err:
        sigchain.vary_gains_in_time(
            gains=gains, times=times, parameter="amp", variation_mode="foobar"
        )
    assert err.value.args[0] == "Variation mode 'foobar' not supported."

def test_mutual_coupling_handle_beam(tmp_path):
    beam = UniformBeam()

    assert sigchain.MutualCoupling._handle_beam(beam) is beam

    uvbeam = beam.to_uvbeam(freq_array=np.array([150e6]), nside=64)
    assert sigchain.MutualCoupling._handle_beam(uvbeam) is uvbeam

    uvbeam.write_beamfits(tmp_path / 'a_beam.fits')
    _read = sigchain.MutualCoupling._handle_beam(tmp_path / 'a_beam.fits')
    assert isinstance(_read, UVBeam)

    with pytest.raises(ValueError, match="uvbeam has incorrect format"):
        sigchain.MutualCoupling._handle_beam('non.existent')
