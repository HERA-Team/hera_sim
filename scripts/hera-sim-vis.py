#!/bin/env python

"""
Command-line interface for simulating visibilities with ``hera_sim``.

This script may be used to run a visibility simulation from a configuration file and
write the result to disk.
"""
import argparse
import importlib
import numpy as np
import psutil
import pyradiosky
import pyuvdata
import pyuvsim
import sys
import yaml
from pathlib import Path

import hera_sim

try:
    from mpi4py import MPI

    HAVE_MPI = True
except ImportError:
    HAVE_MPI = False

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from hera_sim.visibilities import (
    ModelData,
    VisibilitySimulation,
    load_simulator_from_yaml,
)

cns = Console()


def cprint(*args, **kwargs):
    """Print only if root worker."""
    if myid == 0:
        cns.print(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run vis_cpu via hera_sim given an obsparam."
    )
    parser.add_argument("obsparam", type=str, help="pyuvsim-formatted obsparam file.")
    parser.add_argument(
        "simulator_config", type=str, help=" YAML configuration file for the simulator."
    )
    parser.add_argument(
        "--object_name", type=str, default=None, help="Set object_name in the UVData"
    )
    parser.add_argument(
        "--compress", action="store_true", help="Compress by redundancy."
    )
    parser.add_argument(
        "--normalize_beams", action="store_true", help="Peak normalize the beams."
    )
    parser.add_argument(
        "--fix_autos", action="store_true", help="Check and fix non-real xx/yy autos"
    )
    parser.add_argument(
        "--max-auto-imag",
        type=float,
        default=5e-14,
        help="Maximum fraction of imaginary/absolute for autos before raising an error",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="",
        help="If given, do line-profiling on the simulation, and output to given file.",
    )
    parser.add_argument(
        "-p",
        "--extra-profile-func",
        type=str,
        action="append",
        dest="profile_funcs",
        help=(
            "Extra functions to profile. Can be given multiple times. Each must be a "
            "fully-qualified path to a function or method, eg. package.module:function "
            "or package.module:Class.method"
        ),
    )
    args = parser.parse_args()

    if HAVE_MPI and not MPI.Is_initialized():
        MPI.Init()
        comm = MPI.COMM_WORLD
        myid = comm.Get_rank()
    else:
        myid = 0

    cprint(Panel("hera-sim-vis: Simulating Visibilities"))

    # Make data_model, simulator, and simulation objects
    cprint("Initializing ModelData objects... ", end="")
    data_model = ModelData.from_config(
        args.obsparam, normalize_beams=args.normalize_beams
    )
    cprint("[green]:heavy_check_mark:[/green]")

    cprint("Initializing VisibilitySimulator object... ", end="")
    simulator = load_simulator_from_yaml(args.simulator_config)
    cprint("[green]:heavy_check_mark:[/green]")

    # Print versions
    cprint(
        f"""
        [bold]Using the following packages:[/bold]

        \tpyuvdata: {pyuvdata.__version__}
        \tpyuvsim: {pyuvsim.__version__}
        \tpyradiosky: {pyradiosky.__version__}
        \thera_sim: {hera_sim.__version__}
        \t{simulator.__class__.__name__}: {simulator.__version__}
        """
    )

    ram = simulator.estimate_memory(data_model)
    ram_avail = psutil.virtual_memory().available / 1024**3

    cprint(
        f"[bold {'red' if ram < 1.5*ram_avail else 'green'}] This simulation will use "
        f"at least {ram:.2f}GB of RAM (Available: {ram_avail:.2f}GB).[/]"
    )
    if args.object_name is None:
        data_model.uvdata.object_name = simulator.__class__.__name__
    else:
        data_model.uvdata.object_name = args.object_name

    if args.profile:
        cprint(f"Profiling simulation. Output to {args.profile}")
        from line_profiler import LineProfiler

        profiler = LineProfiler()
        profiler.add_function(simulator.simulate)
        for fnc in simulator._functions_to_profile:
            profiler.add_function(fnc)

        # Now add any user-defined functions that they want to be profiled.
        # Functions must be sent in as "path.to.module:function_name" or
        # "path.to.module:Class.method".
        for fnc in args.profile_funcs:
            module = importlib.import_module(fnc.split(":")[0])
            _fnc = module
            for att in fnc.split(":")[-1].split("."):
                _fnc = getattr(_fnc, att)
            profiler.add_function(_fnc)

    simulation = VisibilitySimulation(data_model=data_model, simulator=simulator)

    # Run simulation
    cprint()
    cprint(Rule("Running Simulation"))
    if args.profile:
        profiler.runcall(simulation.simulate)
    else:
        simulation.simulate()
    cprint("[green]:heavy_check_mark:[/] Completed Simulation!")
    cprint(Rule())

    if myid != 0:
        # Wait for root worker to finish IO before ending all other worker procs
        comm.Barrier()
        sys.exit(0)

    if myid == 0:
        # if data_model.uvdata.x_orientation is None:
        #     data_model.uvdata.x_orientation = 'east'

        # Check imaginary of xx/yy autos and fix non-real values if the option is
        # selected in the arguments
        uvd_autos = data_model.uvdata.select(ant_str="auto", inplace=False)
        max_xx_autos_to_abs = (
            np.abs(uvd_autos.get_data("xx").imag) / np.abs(uvd_autos.get_data("xx"))
        ).max()
        if 0 < max_xx_autos_to_abs < args.max_auto_imag:
            cns.print(
                f"[orange]Some autos have very small imaginary components (max ratio "
                f"[blue]{max_xx_autos_to_abs:1.2e}[/])"
            )

            if args.fix_autos:
                cns.print("Setting the autos to be purely real... ", end="")
                data_model.uvdata._fix_autos()
                cns.print("[green]:heavy_check_mark:[/]")
        elif max_xx_autos_to_abs >= args.max_auto_imag:
            raise ValueError(
                f"Some autos have large fractional imaginary components "
                f"(>{args.max_auto_imag:1.2e}). Largest value = "
                f"{np.abs(uvd_autos.get_data('xx').imag).max():1.2e}, largest fraction="
                f"{max_xx_autos_to_abs:1.2e}."
            )

        if args.compress:
            cns.print("Compressing data by redundancy... ", end="")
            data_model.uvdata.compress_by_redundancy(keep_all_metadata=True)
            cns.print("[green]:heavy_check_mark:[/]")

        # Read obsparams to get filing config
        with open(args.obsparam) as file:
            obsparam_dict = yaml.safe_load(file)
        cfg_filing = obsparam_dict["filing"]
        base_path = Path(cfg_filing["outdir"])
        base_path.mkdir(parents=True, exist_ok=True)
        outfile = (
            base_path / f"{cfg_filing['outfile_name']}.{cfg_filing['output_format']}"
        )
        clobber = cfg_filing.get("clobber", False)

        # Write output
        cns.print("Writing output... ", end="")
        data_model.uvdata.write_uvh5(outfile.as_posix(), clobber=clobber)
        cns.print("[green]:heavy_check_mark:[/]")

        if args.profile:
            cns.print(Rule("Profiling Information"))

            profiler.print_stats()

            with open(f"{args.profile}", "w") as fl:
                profiler.print_stats(stream=fl)

            cns.print(Rule())
            cns.print()
    # Sync with other workers and finalise
    if HAVE_MPI:
        comm.Barrier()
    cprint("[green][bold]Complete![/][/]")
