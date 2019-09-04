"""
A module for generating new YAML tags for the various `hera_sim` interpolator
objects. This may need to be updated if the `interpolators` module is updated.
"""
import yaml
import inspect
from . import interpolators

def make_interp_constructor(tag, interpolator):
    """Wrapper for yaml.add_constructor to easily make new YAML tags."""
    def constructor(loader, node):
        params = loader.construct_mapping(node, deep=True)
        datafile = params['datafile']
        interp_kwargs = params.pop('interp_kwargs', {})
        return interpolator(datafile, **interp_kwargs)
    yaml.add_constructor(tag, constructor, yaml.FullLoader)

def predicate(obj):
    """Checks if the passed object `obj` is an interpolator."""
    return hasattr(obj, '_interpolator')

interps = dict(inspect.getmembers(interpolators, predicate))
for tag, interp in interps.items():
    make_interp_constructor("!%s"%tag, interp)
