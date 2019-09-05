"""
The purpose of this script is to provide an interface for the user to use 
hera_sim to create, populate, and save to disk a UVData object.
"""
import hera_sim
import sys
import os
import yaml

# syntax for running from command line:
# python simulation.py path_to_config path_to_save verbose 
# anything else we need to specify?
# notes: 
# path_to_config should be an absolute path to a configuration YAML
# the config YAML will completely specify the instrument and sim component
# parameters (to be applied with defaults and run_sim)
# the decision to perform BDA or not (and which parameters to use) will also
# be made in the config file
#
# path_to_save should be an absolute path specifying the save location with an extension
#
# how to handle verbose?

# get configuration path, simulation parameter path, save path
config, save_path = sys.argv[1:3]

# confirm that config and simulation parameter paths exist
assert os.path.exists(config), \
        "The config file could not be found. Please ensure a path to the file " \
        "exists. The provided path is: {}".format(config)

# verbose option?
verbose = True if "verbose" in sys.argv else False

# make sure save path has an appropriate extension
save_type = os.path.splitext(save_path)[1]
assert save_type != '', \
        "Please specify the file type you would like to use for for saving " \
        "the simulated data by ensuring that the save path has the same " \
        "extension as the desired file type."
# assert save_type in allowed_save_types

# load in config, split into different components
with open(config, 'r') as f:
    yaml_contents = yaml.load(f.read(), Loader=yaml.FullLoader)

bda_params = yaml_contents.get("bda", {})
apply_bda = bda_params.pop("apply_bda", False)
# figure out a good way to load in configuration parameters
# in particular, the current version of hera_sim has a fair amount of redundancy
# in how defaults and run_sim are handled
defaults = yaml_contents.get("defaults", {})
override_defaults = defaults.pop("override_defaults", False)
if override_defaults and defaults:
    hera_sim.defaults.set(**defaults)

sim_params = yaml_contents.get("sim_params", {})
antennas = sim_params.pop("antennas")
# pull number of freq channels, number of times from configuration file
# have all the other stuff (integration time, channel width, etc.) set by defaults
sim = hera_sim.Simulator(n_freq=n_freq, n_times=n_times, antennas=antennas)

# now run the simulation
sim.run_sim(**sim_params)

if apply_bda:
    sim.data = bda_tools.apply_bda(sim.data, **bda_params)

# save the simulation
# note that we may want to allow the functionality for the user to choose some
# kwargs to pass to the write method
sim.write_data(save_path, file_type=save_type)

# should be done now

