import yaml
import inspect
from . import interpolators

def make_constructor(tag, interpolator):
    def constructor(loader, node):
        params = loader.construct_mapping(node, deep=True)
        print(type(params))
        datafile = params['datafile']
        print(type(datafile))
        interp_kwargs = params.pop('interp_kwargs', {})
        print(type(interp_kwargs))
        return interpolator(datafile, **interp_kwargs)
    yaml.add_constructor(tag, constructor, yaml.FullLoader)

def predicate(obj):
    return hasattr(obj, '_interpolator')

interps = dict(inspect.getmembers(interpolators, predicate))
for tag, interp in interps.items():
    make_constructor("!%s"%tag, interp)
