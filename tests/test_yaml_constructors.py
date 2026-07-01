"""Module for testing the various new YAML tags in this package."""

import numpy as np
import pytest
import yaml
from astropy.units.quantity import Quantity

# We need to import the module to register the constructors, but we don't need to use
# it directly
from hera_sim import __yaml_constructors as yaml_constructors


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


def test_npz_constructor(tmp_path):
    arr = np.arange(10)
    np.savez(tmp_path / "test.npz", arr=arr)
    tfile = tmp_path / "test_npz.yaml"
    with open(tfile, "w") as f:
        f.write(
            f"""
            array: !npz {tmp_path}/test.npz
            """
        )
    with open(tfile) as f:
        mydict = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert isinstance(mydict["array"]['arr'], np.ndarray)


def test_npy_constructor(tmp_path):
    arr = np.arange(10)
    np.save(tmp_path / "test.npy", arr)
    tfile = tmp_path / "test_npy.yaml"
    with open(tfile, "w") as f:
        f.write(
            f"""
            array: !npy {tmp_path}/test.npy
            """
        )
    with open(tfile) as f:
        mydict = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert isinstance(mydict["array"], np.ndarray)
