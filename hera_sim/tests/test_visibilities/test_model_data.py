import pytest

import copy
from astropy import units
from pyuvsim.analyticbeam import AnalyticBeam
from pyuvsim.telescope import BeamConsistencyError

from hera_sim.visibilities import ModelData


def test_beam_type_consistency(uvdata, sky_model):
    beams = [AnalyticBeam("gaussian"), AnalyticBeam("airy")]
    beams[0].efield_to_power()

    with pytest.raises(BeamConsistencyError):
        ModelData(uvdata=uvdata, sky_model=sky_model, beams=beams)


def test_power_polsky(uvdata, sky_model):
    new_sky = copy.deepcopy(sky_model)
    new_sky.stokes[1:] = 1.0 * units.Jy

    beams = [AnalyticBeam("gaussian")]
    beams[0].efield_to_power()

    with pytest.raises(TypeError):
        ModelData(uvdata=uvdata, sky_model=new_sky, beams=beams)


def test_bad_uvdata(sky_model):
    with pytest.raises(TypeError, match="uvdata must be a UVData object"):
        ModelData(uvdata=3, sky_model=sky_model)


def test_str_uvdata(uvdata, sky_model, tmp_path):
    pth = tmp_path / "tmp_uvdata.uvh5"
    print(type(pth))
    uvdata.write_uvh5(str(pth))

    model_data = ModelData(uvdata=pth, sky_model=sky_model)
    assert model_data.uvdata.Nants_data == uvdata.Nants_data
