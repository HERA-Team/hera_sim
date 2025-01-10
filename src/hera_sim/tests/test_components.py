import numpy as np
import pytest

from hera_sim.components import component, get_model, get_models
from hera_sim.noise import Noise, ThermalNoise


def test_get_component():
    tn = ThermalNoise()

    with pytest.raises(ValueError) as e:
        tn(lsts=np.array([0]), freqs=np.array([0]), bad_param=0)
    assert "bad_param" in str(e)


def test_bad_docstring():
    with pytest.raises(SyntaxError) as e:

        @component
        class BadComponent:
            def __init__(self):
                """
                Notes
                -----
                Parameters section has to be first...

                Parameters
                ----------
                shoulda been first.
                """
                super().__init__()

            def __call__(self, *args, **kwargs):
                """
                Parameters
                ----------
                args
                """
                print(args)

    assert "Parameters" in str(e)


def test_cls_get_model():
    assert Noise.get_model("thermalnoise") == ThermalNoise
    assert Noise.get_model("ThermalNoise") == ThermalNoise


def test_get_models():
    assert "thermalnoise" in get_models("noise")


def test_get_model():
    assert get_model("thermalnoise", "noise") == ThermalNoise
