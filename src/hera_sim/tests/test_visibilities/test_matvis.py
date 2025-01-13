from pathlib import Path

import pytest

from hera_sim.visibilities import (
    ModelData,
    VisibilitySimulation,
    load_simulator_from_yaml,
)

matvis = pytest.importorskip("hera_sim.visibilities.matvis")
MatVis = matvis.MatVis



def test_stokespol(uvdata_linear, sky_model):
    uvdata_linear.polarization_array = [0, 1, 2, 3]
    with pytest.raises(ValueError):
        VisibilitySimulation(
            data_model=ModelData(uvdata=uvdata_linear, sky_model=sky_model),
            simulator=MatVis(),
        )


def test_load_from_yaml(tmpdir):
    example_dir = Path(__file__).parent.parent.parent.parent / "config_examples"

    simulator = load_simulator_from_yaml(example_dir / "simulator.yaml")
    assert isinstance(simulator, MatVis)
    assert simulator._precision==2

    sim2 = MatVis.from_yaml(example_dir / "simulator.yaml")

    assert sim2.diffuse_ability == simulator.diffuse_ability
