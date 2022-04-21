#!/bin/env python

"""
Command-line interface for simulating visibilities with ``hera_sim``.

This script may be used to run a visibility simulation from a configuration file and
write the result to disk.
"""
import sys
import yaml
import argparse
from pathlib import Path

import numpy as np
import pyuvdata
import pyuvsim
import pyradiosky
import hera_sim
import vis_cpu
from mpi4py import MPI
from astropy.time import Time
from hera_sim.visibilities import VisCPU, ModelData, VisibilitySimulation
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel

cns = Console()


parser = argparse.ArgumentParser(
    description="Run vis_cpu via hera_sim given an obsparam."
)
parser.add_argument("obsparam", type=str, help="pyuvsim-formatted obsparam file.")
parser.add_argument(
    "simulator-config", type=str, help=" YAML configuration file for the simulator."
)
parser.add_argument("--use_pixel_beams", action="store_true", help="Use pixel beam.")
parser.add_argument("--bm_pix", type=int, default=200)
parser.add_argument(
    "--use_mpi", action="store_true", help="Use MPI for parallelization."
)
parser.add_argument(
    "--object_name", type=str, default=None, help="Set object_name in the UVData"
)
parser.add_argument("--compress", action="store_true", help="Compress by redundancy.")
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
    help="Maximum fraction of imaginary/absolute for autos before raising an error.",
)

args = parser.parse_args()

if args.use_mpi:
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
else:
    comm = None
    myid = 0

cns.print(Panel("hera-sim-vis: Simulating Visibilities"))


# Print versions
cns.print(
    f"""
    [bold]Using the following packages:[/bold]

    \tpyuvdata: {pyuvdata.__version__}
    \tpyuvsim: {pyuvsim.__version__}
    \tpyradiosky: {pyradiosky.__version__}
    \tvis_cpu: {vis_cpu.__version__}
    \thera_sim: {hera_sim.__version__}
    """
)

# Make data_model, simulator, and simulation objects
cns.print("Initializing ModelData objects... ", end="")
data_model = ModelData.from_config(args.obsparam, normalize_beams=args.normalize_beams)
cns.print("[green]:heavy-checkmark:[/green]")

if args.object_name is None:
    data_model.uvdata.object_name = "viscpu"
else:
    data_model.uvdata.object_name = args.object_name

cns.print("Initializing VisibilitySimulator object... ", end="")
simulator = VisCPU(
    bm_pix=args.bm_pix,
    use_pixel_beams=args.use_pixel_beams,
    precision=2,
    ref_time=Time(data_model.uvdata.time_array.mean(), format="jd"),
    correct_source_positions=True,
    mpi_comm=comm,
)
cns.print("[green]:heavy-checkmark:[/green]")


simulation = VisibilitySimulation(data_model=data_model, simulator=simulator)

# Run simulation
cns.print()
cns.print(Rule("Running Simulation"))
simulation.simulate()
cns.print("[green]:heavy-checkmark:[/] Completed Simulation!")
cns.print(Rule())


if myid != 0:
    # Wait for root worker to finish IO before ending all other worker procs
    comm.Barrier()
    sys.exit(0)

if myid == 0:
    # if data_model.uvdata.x_orientation is None:
    #     data_model.uvdata.x_orientation = 'east'

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
            cns.print("[green]:heavy-checkmark:[/]")
    elif max_xx_autos_to_abs >= args.max_auto_imag:
        raise ValueError(
            f"Some autos have large fractional imaginary components "
            f"(>{args.max_auto_imag:1.2e}). Largest value = "
            f"{np.abs(uvd_autos.get_data('xx').imag).max():1.2e}, largest fraction = "
            f"{max_xx_autos_to_abs:1.2e}."
        )

    # Check imaginary of xx/yy autos and fix non-real values if the option is
    # selected in the arguments
    if args.compress:
        cns.print("Compressing data by redundancy... ", end="")
        data_model.uvdata.compress_by_redundancy(keep_all_metadata=True)
        cns.print("[green]:heavy-checkmark:[/]")

    # Read obsparams to get filing config
    with open(args.obsparam) as file:
        obsparam_dict = yaml.safe_load(file)
    cfg_filing = obsparam_dict["filing"]
    base_path = Path(cfg_filing["outdir"])
    base_path.mkdir(parents=True, exist_ok=True)
    outfile = base_path / f"{cfg_filing['outfile_name']}.{cfg_filing['output_format']}"

    # Write output
    cns.print("Writing output... ", end="")
    data_model.uvdata.write_uvh5(outfile.as_posix(), clobber=cfg_filing["clobber"])
    cns.print("[green]:heavy-checkmark:[/]")


# Sync with other workers and finalise
if args.use_mpi:
    comm.Barrier()
cns.print("[green][bold]Complete![/][/]")
sys.exit(0)
