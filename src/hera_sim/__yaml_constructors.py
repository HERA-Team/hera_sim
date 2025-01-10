"""
A module for generating new YAML tags for the various ``hera_sim`` interpolator objects.

This may need to be updated if the :mod:`.interpolators` module is updated.
"""

import inspect
import warnings

import astropy.units as u
import yaml

from . import antpos, interpolators


def make_interp_constructor(tag, interpolator):
    """Wrap :func:`yaml.add_constructor` to easily make new YAML tags."""

    def constructor(loader, node):
        params = loader.construct_mapping(node, deep=True)
        datafile = params["datafile"]
        interp_kwargs = params.pop("interp_kwargs", {})
        return interpolator(datafile, **interp_kwargs)

    yaml.add_constructor(tag, constructor, yaml.FullLoader)


def predicate(obj):
    """Check if the passed object `obj` is an interpolator."""
    return hasattr(obj, "_interpolator")


interps = dict(inspect.getmembers(interpolators, predicate))
for tag, interp in interps.items():
    make_interp_constructor(f"!{tag}", interp)


def astropy_unit_constructor(loader, node):
    """Construct an astropy unit."""
    params = loader.construct_mapping(node, deep=True)
    value = params.get("value", None)
    units = params.get("units", None)
    if value is None:
        return None
    elif units is None:
        warnings.warn(
            "You have not specified the units for this item. Returning None.",
            stacklevel=2,
        )
        return None
    else:
        try:
            return value * getattr(u, units)
        except AttributeError:
            raise ValueError(
                "You have selected units that are not an astropy "
                "Quantity. Please check your configuration file."
            )


yaml.add_constructor("!dimensionful", astropy_unit_constructor, yaml.FullLoader)


def antpos_constructor(loader, node):
    """Construct an antenna position."""
    params = loader.construct_mapping(node, deep=True)
    array_type = params.pop("array_type") + "_array"
    antpos_func = getattr(antpos, array_type)
    return antpos_func(**params)


yaml.add_constructor("!antpos", antpos_constructor, yaml.FullLoader)
