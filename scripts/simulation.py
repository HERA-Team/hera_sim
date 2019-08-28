"""
The purpose of this script is to provide an interface for the user to use 
hera_sim to create, populate, and save to disk a UVData object.
"""
import hera_sim
import sys
import os

# syntax for running from command line:
# python simulation.py path_to_config path_to_sim path_to_save perform_bda
# anything else we need to specify?
# notes: 
# path_to_config should be an absolute path to a configuration YAML
# the configuration YAML will be loaded in and its settings applied using hera_sim.defaults
#
# path_to_sim should be an absolute path to a configuration YAML
# the configuration YAML will be loaded in and applied using hera_sim.Simulator.run_sim
#
# path_to_save should be an absolute path specifying the save location with an extension
#
# perform_bda is an optional argument; figure out how we want to deal with this
#
# add a verbose option?

# get configuration path, simulation parameter path, save path
config, sim_params, save_path = sys.argv[1:4]

# confirm that config and simulation parameter paths exist
for path in (config, sim_params):
    assert os.path.exists(path), \
            "The path {} could not be found. Please ensure the path exists.".format(path)
# verbose option?
verbose = True if "verbose" in sys.argv else False

# assume that config and sim_params are YAMLs, but make sure that save_path
# has an extension
save_type = os.path.splitext(save_path)[1]
assert save_type != '', \
        "Please specify the file type you would like to use for for saving " \
        "the simulated data by ensuring that the save path has the same " \
        "extension as the desired file type."

# apply the configuration
hera_sim.defaults.set(config)

# figure out how to allow this command to initialize a Simulator object
# with parameters specified in the configuration file
sim = hera_sim.Simulator()

# now run the simulation
sim.run_sim(sim_params)

# do BDA here? figure out how to do it, ideally by just rewriting the Simulator
# object's UVData attribute

# save the simulation
# note that we may want to allow the functionality for the user to choose some
# kwargs to pass to the write method
sim.write_data(save_path, file_type=save_type)

# should be done now

