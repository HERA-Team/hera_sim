import yaml
from . import interpolators

# TODO: make a wrapper that will automate this for each interpolator object
def tsky_constructor(loader, node):
    params = loader.construct_mapping(node, deep=True)
    datafile = params['datafile']
    interp_kwargs = params.pop('interp_kwargs', {})
    return interpolators.Tsky(datafile, **interp_kwargs)

def beam_constructor(loader, node):
    params = loader.construct_mapping(node, deep=True)
    # TODO: figure out a better way to do this
    datafile = params['datafile']
    interp_kwargs = params.pop('interp_kwargs', {})
    return interpolators.freq_interp1d(datafile, **interp_kwargs)

def bandpass_constructor(loader, node):
    params = loader.construct_mapping(node, deep=True)
    datafile = params['datafile']
    interp_kwargs = params.pop('interp_kwargs', {})
    return interpolators.freq_interp1d(datafile, **interp_kwargs)

yaml.add_constructor('!Tsky', tsky_constructor, yaml.FullLoader)
yaml.add_constructor('!Beam', beam_constructor, yaml.FullLoader)
yaml.add_constructor('!Bandpass', bandpass_constructor, yaml.FullLoader)
