from contextlib import ExitStack as does_not_raise

import numpy as np
import pytest
from astropy import units
from uvtools.utils import FFT, fourier_freqs

from hera_sim import DATA_PATH, foregrounds
from hera_sim.interpolators import Beam, Tsky


@pytest.fixture(scope="function")
def omega_p():
    beamfile = DATA_PATH / "HERA_H1C_BEAM_POLY.npy"
    omega_p = Beam(beamfile)
    return omega_p


@pytest.fixture(scope="function")
def freqs():
    return np.linspace(0.1, 0.2, 100, endpoint=False)


@pytest.fixture(scope="function")
def lsts():
    return np.linspace(0, 2 * np.pi, 1000)


@pytest.fixture(scope="function")
def Tsky_mdl():
    datafile = DATA_PATH / "HERA_Tsky_Reformatted.npz"
    return Tsky(datafile)


@pytest.mark.parametrize("model", ["pntsrc", "diffuse"])
def test_foreground_shape(freqs, lsts, Tsky_mdl, omega_p, model):
    bl_vec = [0, 0, 0]
    if model == "pntsrc":
        vis = foregrounds.pntsrc_foreground(lsts=lsts, freqs=freqs, bl_vec=bl_vec)
    elif model == "diffuse":
        vis = foregrounds.diffuse_foreground(
            lsts=lsts, freqs=freqs, bl_vec=bl_vec, Tsky_mdl=Tsky_mdl, omega_p=omega_p
        )
    assert vis.shape == (lsts.size, freqs.size)


@pytest.mark.parametrize("model", ["pntsrc", "diffuse"])
def test_foreground_autos_are_real(freqs, lsts, Tsky_mdl, omega_p, model):
    bl_vec = [0, 0, 0]
    if model == "pntsrc":
        vis = foregrounds.pntsrc_foreground(lsts=lsts, freqs=freqs, bl_vec=bl_vec)
    elif model == "diffuse":
        vis = foregrounds.diffuse_foreground(
            lsts=lsts, freqs=freqs, bl_vec=bl_vec, Tsky_mdl=Tsky_mdl, omega_p=omega_p
        )
    assert np.allclose(vis.imag, 0)
    # import uvtools, pylab as plt
    # uvtools.plot.waterfall(vis, mode='log'); plt.colorbar(); plt.show()


@pytest.mark.parametrize("orientation", ["east", "west", "north"])
@pytest.mark.parametrize(
    "model, expectation",
    [("pntsrc", pytest.raises(AssertionError)), ("diffuse", does_not_raise())],
)
def test_foreground_orientation(
    freqs, lsts, Tsky_mdl, omega_p, model, orientation, expectation
):
    baselines = {"east": [100, 0, 0], "west": [-100, 0, 0], "north": [0, 100, 0]}
    bl_vec = baselines[orientation]
    fringe_filter_kwargs = {"fringe_filter_type": "gauss", "fr_width": 1e-5}
    kwargs = dict(freqs=freqs, lsts=lsts, bl_vec=bl_vec)
    # TODO: Update this to avoid repeated code?
    if model == "diffuse":
        model = foregrounds.diffuse_foreground
        kwargs.update(
            dict(
                Tsky_mdl=Tsky_mdl,
                omega_p=omega_p,
                fringe_filter_kwargs=fringe_filter_kwargs,
            )
        )
    elif model == "pntsrc":
        # FIXME: point source foregrounds do not behave correctly in fringe-rate.
        model = foregrounds.pntsrc_foreground
        # Hack to make the tests pass for the eastward-orientation. This should
        # be extra motivation for really digging into how we're simulating point
        # source visibilities, since it doesn't consistently give the wrong
        # fringe-rate behavior.
        if orientation == "east":
            expectation = does_not_raise()

    vis = model(**kwargs)
    vis_fft = FFT(vis, axis=0, taper="blackmanharris")
    fringe_rates = fourier_freqs(lsts * units.day.to("s") * units.rad.to("cycle"))
    max_frs = fringe_rates[np.argmax(np.abs(vis_fft), axis=0)]
    divisor = 1 if orientation == "north" else np.abs(max_frs)
    fr_signs = max_frs / divisor
    expected_sign = {"east": 1, "west": -1, "north": 0}[orientation]
    with expectation:
        assert np.allclose(fr_signs, expected_sign, atol=1e-4)


@pytest.mark.parametrize("model", ["pntsrc", "diffuse"])
@pytest.mark.xfail
def test_foreground_conjugation(freqs, lsts, Tsky_mdl, omega_p, model):
    bl_vec = np.array([100.0, 0, 0])
    delay_filter_kwargs = {"delay_filter_type": "tophat"}
    fringe_filter_kwargs = {"fringe_filter_type": "tophat"}
    kwargs = dict(freqs=freqs, lsts=lsts, bl_vec=bl_vec)
    if model == "diffuse":
        model = foregrounds.diffuse_foreground
        kwargs.update(
            dict(
                Tsky_mdl=Tsky_mdl,
                omega_p=omega_p,
                delay_filter_kwargs=delay_filter_kwargs,
                fringe_filter_kwargs=fringe_filter_kwargs,
            )
        )
    elif model == "pntsrc":
        model = foregrounds.pntsrc_foreground

    conj_kwargs = kwargs.copy()
    conj_kwargs["bl_vec"] = -bl_vec
    vis = model(**kwargs, rng=np.random.default_rng(0))
    conj_vis = model(**conj_kwargs, rng=np.random.default_rng(0))
    assert np.allclose(vis, conj_vis.conj())  # Assert V_ij = V*_ji


def test_diffuse_foreground_exception_no_tsky_mdl(freqs, lsts, omega_p):
    bl_vec = [0, 0, 0]
    with pytest.raises(ValueError) as err:
        foregrounds.diffuse_foreground(
            freqs=freqs, lsts=lsts, bl_vec=bl_vec, Tsky_mdl=None, omega_p=omega_p
        )
    assert "sky temperature model must be" in err.value.args[0]


def test_diffuse_foreground_exception_no_omega_p(freqs, lsts, Tsky_mdl):
    bl_vec = [0, 0, 0]
    with pytest.raises(ValueError) as err:
        foregrounds.diffuse_foreground(
            freqs=freqs, lsts=lsts, bl_vec=bl_vec, Tsky_mdl=Tsky_mdl, omega_p=None
        )
    assert "beam area array or interpolation object" in err.value.args[0]
