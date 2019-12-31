from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

try:
    from future import standard_library
except ImportError:
    from pip._internal import main as pip
    pip(['install', '--user', 'future'])
    from future import standard_library

standard_library.install_aliases()
from builtins import *
from builtins import str
import os
import subprocess
import json


def construct_version_info():
    hera_sim_dir = os.path.dirname(os.path.realpath(__file__))
    version_file = os.path.join(hera_sim_dir, "VERSION")
    version = open(version_file).read().strip()

    try:
        git_origin = subprocess.check_output(
            ["git", "-C", hera_sim_dir, "config", "--get", "remote.origin.url"],
            stderr=subprocess.STDOUT,
        ).strip()
        git_hash = subprocess.check_output(
            ["git", "-C", hera_sim_dir, "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).strip()
        git_description = subprocess.check_output(
            ["git", "-C", hera_sim_dir, "describe", "--dirty", "--tag", "--always"]
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "-C", hera_sim_dir, "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.STDOUT,
        ).strip()
        git_version = subprocess.check_output(
            ["git", "-C", hera_sim_dir, "describe", "--always", "--tags", "--abbrev=0"]
        ).strip()
    except:  # pragma: no cover - can't figure out how to test exception.
        try:
            # Check if a GIT_INFO file was created when installing package
            git_file = os.path.join(hera_sim_dir, "GIT_INFO")
            with open(git_file) as data_file:
                data = [x.encode("UTF8") for x in json.loads(data_file.read().strip())]
                git_origin = data[0]
                git_hash = data[1]
                git_description = data[2]
                git_branch = data[3]
        except:
            git_origin = ""
            git_hash = ""
            git_description = ""
            git_branch = ""

    version_info = {
        "version": version,
        "git_origin": git_origin,
        "git_hash": git_hash,
        "git_description": git_description,
        "git_branch": git_branch,
    }
    return version_info


version_info = construct_version_info()
version = version_info["version"]
git_origin = version_info["git_origin"]
git_hash = version_info["git_hash"]
git_description = version_info["git_description"]
git_branch = version_info["git_branch"]

# String to add to history of any files written with this version of hera_sim
hera_sim_version_str = "hera_sim version: " + version + "."
if git_hash is not "":
    hera_sim_version_str += (
        "  Git origin: "
        + str(git_origin)
        + ".  Git hash: "
        + str(git_hash)
        + ".  Git branch: "
        + str(git_branch)
        + ".  Git description: "
        + str(git_description)
        + "."
    )


def main():
    print(("Version = {0}".format(version)))
    print(("git origin = {0}".format(git_origin)))
    print(("git branch = {0}".format(git_branch)))
    print(("git description = {0}".format(git_description)))


if __name__ == "__main__":
    main()
