#!/bin/env python

"""
Command-line interface for simulating visibilities with ``hera_sim``.

This script may be used to run a visibility simulation from a configuration file and
write the result to disk.
"""

import argparse
import atexit
import logging
import sys
from pathlib import Path

import numpy as np
import psutil
import pyradiosky
import pyuvdata
import pyuvsim
import yaml

import hera_sim

try:
    from mpi4py import MPI

    HAVE_MPI = True
except ImportError:  # pragma: no cover
    HAVE_MPI = False

from hera_cli_utils.logging import RicherHandler
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from hera_sim.visibilities import (
    ModelData,
    VisibilitySimulation,
    load_simulator_from_yaml,
)

# Use the root logger here so that we can update the log-level of the underlying
# simulator code (where applicable). Unfortunately, this has the side effect that
# other third party code also gets the same log level.
logger = logging.getLogger()

cns = Console(width=160)

logging.basicConfig(
    format="%(message)s",
    handlers=[
        RicherHandler(
            console=cns,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_path=False,
            show_time_as_diff=True,
        )
    ],
)

if HAVE_MPI:
    if not MPI.Is_initialized():
        MPI.Init()
        atexit.register(MPI.Finalize)
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
else:  # pragma: no cover
    myid = 0


def cprint(*args, **kwargs):
    """Print only if root worker."""
    if myid == 0:
        cns.print(*args, **kwargs)


def print_sim_config(obsparam):
    cprint()
    cprint(Rule("[bold]Simulation Configuration:"))
    with open(obsparam) as fl:
        d = yaml.load(fl, Loader=yaml.FullLoader)

    cprint(yaml.dump(d, default_flow_style=False))
    cprint(Rule())
    cprint()


def run_vis_sim(args):
    cprint(Panel("hera-sim-vis: Simulating Visibilities"))

    logger.info("Initializing VisibilitySimulator object... ")
    simulator = load_simulator_from_yaml(args.simulator_config)
    logger.info("Finished VisibilitySimulator Init")

    # Make data_model, simulator, and simulation objects
    logger.info("Initializing ModelData object... ")
    data_model = ModelData.from_config(
        args.obsparam, normalize_beams=args.normalize_beams
    )
    if args.phase_center_name is not None:
        if len(data_model.uvdata.phase_center_catalog) > 1:
            cprint(
                "[bold red]:warning: Phase center catalog has length > 1. "
                f"Cannot set phase center name to {args.phase_center_name}[/]"
            )
        else:
            next(iter(data_model.uvdata.phase_center_catalog.values()))[
                "name"
            ] = args.phase_center_name
            cprint(f"Phase center name set to {args.phase_center_name}")
    logger.info("Finished Setting up ModelData object")
    print_sim_config(args.obsparam)

    cprint(f"Using {simulator.__class__.__name__} Simulator")

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

    cns.print(Rule("Important Simulation Parameters"))
    cns.print(f"Nfreqs  : {data_model.uvdata.Nfreqs}")
    cns.print(f"Ntimes  : {len(data_model.lsts)}")
    cns.print(f"Npols   : {data_model.uvdata.Npols}")
    cns.print(f"Nants   : {data_model.uvdata.Nants_data}")
    cns.print(f"Nsources: {data_model.sky_model.Ncomponents}")
    cns.print(f"Nbeams  : {data_model.n_beams}")
    cns.print()

    cns.print(Rule("Large Memory Components"))
    cns.print(
        f"Visibility Array  : {data_model.uvdata.data_array.nbytes / 1024**2:.2f} MB"
    )
    beam_array_sizes = [
        b.data_array.nbytes for b in data_model.beams if hasattr(b, "data_array")
    ]
    if beam_array_sizes:
        cns.print(f"Largest Beam Array: {max(beam_array_sizes) / 1024**2:.2f} MB")
        cns.print(f"Total Beam Arrays : {sum(beam_array_sizes) / 1024**2:.2f} MB")

    ram = simulator.estimate_memory(data_model)
    ram_avail = psutil.virtual_memory().available / 1024**3

    cprint(
        f"[bold {'red' if ram < 1.5 * ram_avail else 'green'}] This simulation will use"
        f" at least {ram:.2f}GB of RAM (Available: {ram_avail:.2f}GB).[/]"
    )

    if args.object_name is None:
        data_model.uvdata.object_name = simulator.__class__.__name__
    else:
        data_model.uvdata.object_name = args.object_name

    if args.dry_run:
        cprint("Dry run finished.")
        return

    simulation = VisibilitySimulation(data_model=data_model, simulator=simulator)

    # Run simulation
    cprint()
    cprint(Rule("Running Simulation"))
    logger.info("About to Run Simulation")
    simulation.simulate()
    logger.info("Simulation Complete")
    cprint(Rule())

    if myid != 0:  # pragma: no cover
        # Wait for root worker to finish IO before ending all other worker procs
        comm.Barrier()
        sys.exit(0)

    if args.run_auto_check:
        # Check imaginary of xx/yy autos and fix non-real values if the option is
        # selected in the arguments
        # xxpol = data_model.uvdata.get_data("xx")
        # auto_idx = data_model.uvdata.ant_1_array == data_model.uvdata.ant_2_array
        # xxpol = xxpol[auto_idx]

        # max_xx_autos_to_abs = (np.abs(xxpol.imag) / np.abs(xxpol)).max()

        uvd_autos = data_model.uvdata.select(
            ant_str="auto",
            inplace=False,
            run_check=False,
            run_check_acceptability=False,
            check_extra=False,
        )
        xx = uvd_autos.get_data("xx")
        max_xx_autos_to_abs = (np.abs(xx.imag) / np.abs(xx)).max()
        if 0 < max_xx_autos_to_abs < args.max_auto_imag:
            logger.warning(
                f"[orange]Some autos have very small imaginary components (max ratio "
                f"[blue]{max_xx_autos_to_abs:1.2e}[/])"
            )

            if args.fix_autos:
                logger.info("Setting the autos to be purely real... ")
                data_model.uvdata._fix_autos()
                logger.info("Done fixing autos.")

        elif max_xx_autos_to_abs >= args.max_auto_imag:
            raise ValueError(
                f"Some autos have large fractional imaginary components "
                f"(>{args.max_auto_imag:1.2e}). Largest value = "
                f"{np.abs(xx.imag).max():1.2e}, largest fraction="
                f"{max_xx_autos_to_abs:1.2e}."
            )

    if args.compress:
        logger.info("Compressing data by redundancy... ")
        # Here, we don't call the convenience function directly, because we want to
        # be able to short-circuit the process by reading in a file.
        if not Path(args.compress).exists():
            # Note that we use tol=4, which corresponds to tol=1 for the hera_cal
            # method (which is used when use_grid_alg=True). This ensures consistency
            # when constructing redundancies for hera analyses.
            red_gps = data_model.uvdata.get_redundancies(
                tol=4.0, include_conjugates=True, use_grid_alg=True
            )[0]
            bl_ants = [data_model.uvdata.baseline_to_antnums(gp[0]) for gp in red_gps]
            blt_inds = data_model.uvdata._select_preprocess(
                antenna_nums=None,
                antenna_names=None,
                ant_str=None,
                bls=bl_ants,
                frequencies=None,
                freq_chans=None,
                times=None,
                time_range=None,
                lsts=None,
                lst_range=None,
                polarizations=None,
                blt_inds=None,
                phase_center_ids=None,
                catalog_names=None,
            )[0]

            np.save(args.compress, blt_inds)
        else:
            blt_inds = np.load(args.compress)

        data_model.uvdata._select_by_index(
            blt_inds=blt_inds,
            pol_inds=None,
            freq_inds=None,
            history_update_string="Compressed by redundancy",
            keep_all_metadata=True,
        )

        logger.info("Done with compression.")

    # Read obsparams to get filing config
    with open(args.obsparam) as file:
        obsparam_dict = yaml.safe_load(file)
    cfg_filing = obsparam_dict["filing"]
    base_path = Path(cfg_filing["outdir"])
    base_path.mkdir(parents=True, exist_ok=True)
    outfile = base_path / f"{cfg_filing['outfile_name']}.{cfg_filing['output_format']}"
    clobber = cfg_filing.get("clobber", False)

    # Write output
    logger.info("Writing output... ")
    data_model.uvdata.write_uvh5(
        outfile.as_posix(),
        clobber=clobber,
        run_check=False,
        run_check_acceptability=False,
    )
    logger.info("Done Writing.")

    # Sync with other workers and finalise
    if HAVE_MPI:
        comm.Barrier()

    cprint("[green][bold]Complete![/][/]")


def vis_cli_argparser():
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
        "--compress",
        type=str,
        help="Compress by redundancy. A file name to store the cache.",
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
        "-d",
        "--dry-run",
        action="store_true",
        help="If set, create the simulator and data model but don't run simulation.",
    )
    parser.add_argument(
        "--run-auto-check", action="store_true", help="whether to check autos are real"
    )
    parser.add_argument(
        '--phase-center-name', default=None, help="name of phase center"
    )

    return parser
