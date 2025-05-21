"""Module for testing the various new YAML tags in this package."""

import pytest
import yaml
from astropy.units.quantity import Quantity


def test_astropy_units_constructor(tmp_path):
    tfile = tmp_path / "test_astro.yaml"
    with open(tfile, "w") as f:
        f.write(
            """
            time: !dimensionful
                value: 1
                units: s
            angle1: !dimensionful
                value: 3
                units: deg
            angle2: !dimensionful
                value: 1
                units: rad
                """
        )
    with open(tfile) as f:
        mydict = yaml.load(f.read(), Loader=yaml.FullLoader)
    for value in mydict.values():
        assert isinstance(value, Quantity)


def test_astropy_constructor_nonetypes(tmp_path):
    tfile = tmp_path / "test_astro_nones.yaml"
    with open(tfile, "w") as f:
        f.write(
            """
            none_value: !dimensionful
                units: s
            none_units: !dimensionful
                value: 1
                """
        )
    with open(tfile) as f:
        mydict = yaml.load(f.read(), Loader=yaml.FullLoader)
    for value in mydict.values():
        assert value is None


def bad_astropy_units(tmp_path):
    tfile = tmp_path / "bad_units.yaml"
    with open(tfile, "w") as f:
        f.write(
            """
            quantity: !dimensionful
                value: 1
                units: bad_units"""
        )
    with pytest.raises(ValueError) as err:
        with open(tfile) as f:
            yaml.load(f.read(), Loader=yaml.FullLoader)
    assert "Please check your configuration" in err.value.args[0]
