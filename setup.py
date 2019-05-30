from setuptools import setup
import glob
import os
import sys

# Import version this way so as not to import the entire hera_sim package.
# Importing all of hera_sim creates problems when requirements aren't installed.
sys.path.append("hera_sim")
import version

import json

data = [
    str(version.git_origin),
    str(version.git_hash),
    str(version.git_description),
    str(version.git_branch),
]

if sys.version[0] == '2':
    with open(os.path.join("hera_sim", "GIT_INFO"), "w") as outfile:
        json.dump(data, outfile)
else:
    with open(os.path.join("hera_sim", "GIT_INFO"), "w", encoding='utf8') as outfile:
        json.dump(data, outfile)


def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + "/", "")
            paths.append(os.path.join(path, filename))
    return paths


data_files = package_files("hera_sim", "data")

setup_args = {
    "name": "hera_sim",
    "author": "HERA Team",
    "url": "https://github.com/HERA-Team/hera_sim",
    "license": "BSD",
    "description": "collection of simulation routines describing the HERA instrument.",
    "package_dir": {"hera_sim": "hera_sim"},
    "packages": ["hera_sim"],
    "include_package_data": True,
    "install_requires": [
        'numpy>=1.14',
        'scipy',
        'cached_property',
        'mpi4py',   # this is a dependency of pyuvsim which currently is not automatically installed. Remove when possible.
        'pyuvsim',
        'pyuvdata',
        'aipy'
    ],
    "version": version.version,
    "package_data": {"hera_sim": data_files},
    "zip_safe": False,
}


# If on Python 2, we also need the "future" module
if sys.version.startswith("2"):
    setup_args['install_requires'].append("future")

if __name__ == "__main__":
    setup(*(), **setup_args)
