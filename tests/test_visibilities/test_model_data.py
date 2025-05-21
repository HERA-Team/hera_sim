import copy

import pytest
from astropy import units
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import GaussianBeam
from pyuvdata.beam_interface import BeamInterface

from hera_sim.visibilities import ModelData


def test_power_polsky(uvdata, sky_model):
    new_sky = copy.deepcopy(sky_model)
    new_sky.stokes[1:] = 1.0 * units.Jy

    beams = [BeamInterface(GaussianBeam(diameter=14.0), beam_type='power')]
    with pytest.raises(
        TypeError,
        match='Cannot use power beams when the sky model contains polarized sources'
    ):
        ModelData(uvdata=uvdata, sky_model=new_sky, beams=beams)


def test_bad_uvdata(sky_model):
    with pytest.raises(TypeError, match="uvdata must be a UVData object"):
        ModelData(uvdata=3, sky_model=sky_model)


def test_str_uvdata(uvdata, sky_model, tmp_path):
    pth = tmp_path / "tmp_uvdata.uvh5"
    uvdata.write_uvh5(str(pth))

    model_data = ModelData(uvdata=pth, sky_model=sky_model)
    assert model_data.uvdata.Nants_data == uvdata.Nants_data

def test_peak_normalize(uvdata, sky_model, uvbeam: UVBeam):
    beam = copy.deepcopy(uvbeam)
    beam.data_normalization='not peak'
    model_data = ModelData(
        uvdata=uvdata, sky_model=sky_model, beams=[uvbeam], normalize_beams=True
    )
    assert model_data.beams[0].beam.data_normalization == 'peak'

def test_setting_rectangularity(uvdata, sky_model, uvbeam):
    uvdata.blts_are_rectangular = None  # mock for testing
    model_data = ModelData(uvdata=uvdata, sky_model=sky_model, beams=[uvbeam])
    assert model_data.uvdata.blts_are_rectangular  # set to True now!

def test_beam_ids(uvdata, sky_model, uvbeam):
    beam = copy.deepcopy(uvbeam)
    beam.data_normalization='not peak'

    model_data = ModelData(
        uvdata=uvdata, sky_model=sky_model, beams=[uvbeam]
    )
    assert all(v==0 for v in model_data.beam_ids.values())

    model_data = ModelData(
        uvdata=uvdata, sky_model=sky_model, beams=[uvbeam]*uvdata.Nants_data
    )
    assert all(v==i for i, v in enumerate(model_data.beam_ids.values()))

    # Try bad length of beams
    with pytest.raises(ValueError, match="Need to give beam_ids if beams is given"):
        model_data = ModelData(
            uvdata=uvdata, sky_model=sky_model, beams=[uvbeam, uvbeam],
        )

    # Try bad length of beam_ids
    with pytest.raises(ValueError, match="Number of beam_ids given must match n_ant"):
        model_data = ModelData(
            uvdata=uvdata, sky_model=sky_model, beams=[uvbeam],
            beam_ids=[0, 1]
        )

    # Correct beam_ids
    model_data = ModelData(
        uvdata=uvdata, sky_model=sky_model, beams=[uvbeam],
        beam_ids=[0]*uvdata.Nants_data
    )

    assert all(v==0 for v in model_data.beam_ids.values())

    # Finally, just terrible beam_ids
    with pytest.raises(
        TypeError,
        match="beam_ids should be a dict or sequence of integers"
    ):
        model_data = ModelData(
            uvdata=uvdata, sky_model=sky_model, beams=[uvbeam],
            beam_ids='hey there'
        )
