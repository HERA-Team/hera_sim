"""
The purpose of this script is to provide an interface for the user to use 
hera_sim to create, populate, and save to disk a UVData object.
"""
import hera_sim
import sys
import os

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
assert os.path.exists(path), \
        "The config file could not be found. Please ensure a path to the file " \
        "exists. The provided path is: {}".format(path)

# verbose option?
verbose = True if "verbose" in sys.argv else False

# make sure save path has an appropriate extension
save_type = os.path.splitext(save_path)[1]
assert save_type != '', \
        "Please specify the file type you would like to use for for saving " \
        "the simulated data by ensuring that the save path has the same " \
        "extension as the desired file type."
# assert save_type in allowed_save_types

# this is a little complicated, but maybe parse config file and make new files

# apply the configuration
hera_sim.defaults.set(config)

# figure out how to allow this command to initialize a Simulator object
# with parameters specified in the configuration file
sim = hera_sim.Simulator()

# now run the simulation
sim.run_sim(sim_params)



# save the simulation
# note that we may want to allow the functionality for the user to choose some
# kwargs to pass to the write method
sim.write_data(save_path, file_type=save_type)

# should be done now

