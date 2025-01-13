import os
from pathlib import Path

import pytest
from hera_cli_utils import parse_args
from pyuvdata import UVData

from hera_sim.visibilities.cli import run_vis_sim, vis_cli_argparser

pytest.importorskip("hera_sim.visibilities.matvis")

DATA_PATH = Path(__file__).parent.parent / "testdata" / "hera-sim-vis-config"


def get_config_files(tmp_path, ntimes, nfreqs=2):
    txt = f"""
'filing':
  'outdir': {tmp_path}
  'outfile_name': 'out'
  'output_format': 'uvh5'
  'clobber': true
'freq':
  'Nfreqs': {nfreqs}
  'channel_width': 1000000.0
  'start_freq': 100000000.0
'sources':
  'catalog': '{DATA_PATH}/gleam_50srcs.vot'
'telescope':
  'array_layout': '{DATA_PATH}/H4C-antennas-slim.csv'
  'telescope_config_name': '{tmp_path}/telescope.yaml'
  'select':
    'freq_buffer': 3000000.0
'time':
  'Ntimes': {ntimes}
  'integration_time': 4.986347833333333
  'start_time': 2458208.916228965
'polarization_array':
  - -5
  - -7
  - -8
  - -6
    """

    with open(tmp_path / "obsparams.yaml", "w") as fl:
        fl.write(txt)

    telfile = f"""
beam_paths:
  0: !UVBeam
    filename: {DATA_PATH}/NF_HERA_Dipole_small.fits
telescope_location: (-30.72152612068957, 21.428303826863015, 1051.6900000218302)
telescope_name: HERA
    """
    with open(tmp_path / "telescope.yaml", "w") as fl:
        fl.write(telfile)

    return tmp_path / "obsparams.yaml"


@pytest.mark.parametrize("phase_center_name", [None, 'zenith'])
def test_vis_cli(tmp_path_factory, phase_center_name):
    outdir = tmp_path_factory.mktemp("vis-sim")
    cfg = get_config_files(outdir, 5, 2)

    parser = vis_cli_argparser()
    args = parse_args(
        parser,
        [
            str(cfg),
            str(DATA_PATH / "matvis_cpu.yaml"),
            "--compress",
            str(outdir / "compression-cache.npy"),
            "--normalize_beams",
            "--fix_autos",
        ],
    )

    run_vis_sim(args)
    contents = os.listdir(outdir)
    assert "out.uvh5" in contents

    # Run again to use the compression cache.
    # But this time, check the autos...
    args.run_auto_check = True
    run_vis_sim(args)


def test_vis_cli_dry(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("vis-sim")
    cfg = get_config_files(outdir, 5, 2)

    parser = vis_cli_argparser()
    args = parse_args(
        parser,
        [
            str(cfg),
            str(DATA_PATH / "matvis_cpu.yaml"),
            "--compress",
            str(outdir / "compression-cache.npy"),
            "--dry",
            "--object_name",
            "matvis",
        ],
    )

    run_vis_sim(args)
    contents = os.listdir(outdir)
    assert "out.uvh5" not in contents

def test_vis_cli_phase_center_name(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("vis-sim")
    cfg = get_config_files(outdir, 5, 2)

    parser = vis_cli_argparser()
    args = parse_args(
        parser,
        [
            str(cfg),
            str(DATA_PATH / "matvis_cpu.yaml"),
            "--compress",
            str(outdir / "compression-cache.npy"),
            "--phase-center-name", 'zenith',
            "--object_name",
            "matvis",
        ],
    )

    run_vis_sim(args)
    uvd = UVData.from_file(outdir / 'out.uvh5')
    print(uvd.phase_center_catalog[0].keys())
    assert uvd.phase_center_catalog[0]['name'] == 'zenith'
