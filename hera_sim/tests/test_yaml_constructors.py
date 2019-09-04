"""Module for testing the various new YAML tags in this package."""

import yaml
import tempfile
import os

from nose.tools import raises
from astropy.units.quantity import Quantity

import hera_sim.__yaml_constructors

def make_empty_yaml():
    tempdir = tempfile.mkdtemp()
    tfile = os.path.join(tempdir, "temp.yaml")
    return tfile

def test_astropy_units_constructor():
    tfile = make_empty_yaml()
    with open(tfile, 'w') as f:
        f.write("""
            time: !dimensionful
                value: 1
                units: s
            angle1: !dimensionful
                value: 3
                units: deg
            angle2: !dimensionful
                value: 1
                units: rad
                """)
    with open(tfile, 'r') as f:
        mydict = yaml.load(f.read(), Loader=yaml.FullLoader)
    for value in mydict.values():
        assert isinstance(value, Quantity)

def test_astropy_constructor_nonetypes():
    tfile = make_empty_yaml()
    with open(tfile, 'w') as f:
        f.write("""
            none_value: !dimensionful
                units: s
            none_units: !dimensionful
                value: 1
                """)
    with open(tfile, 'r') as f:
        mydict = yaml.load(f.read(), Loader=yaml.FullLoader)
    for value in mydict.values():
        assert value is None


@raises(ValueError)
def bad_astropy_units():
    tfile = make_empty_yaml()
    with open(tfile, 'w') as f:
        f.write("""
            quantity: !dimensionful
                value: 1
                units: seconds""")
    with open(tfile, 'r') as f:
        mydict = yaml.load(f.read(), Loader=yaml.FullLoader)

