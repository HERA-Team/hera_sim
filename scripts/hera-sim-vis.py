#!/bin/env python

"""
Command-line interface for simulating visibilities with ``hera_sim``.

This script may be used to run a visibility simulation from a configuration file and
write the result to disk.
"""

from hera_cli_utils import parse_args, run_with_profiling

from hera_sim.visibilities.cli import run_vis_sim, vis_cli_argparser

if __name__ == "__main__":
    parser = vis_cli_argparser()
    args = parse_args(parser)

    run_with_profiling(run_vis_sim, args, args)
