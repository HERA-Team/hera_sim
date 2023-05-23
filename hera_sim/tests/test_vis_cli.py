import os
from pathlib import Path

DATA_PATH = Path(__file__).parent / "testdata" / "hera-sim-vis-config"


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
  0: {DATA_PATH}/NF_HERA_Dipole_small.fits
telescope_location: (-30.72152612068957, 21.428303826863015, 1051.6900000218302)
telescope_name: HERA
    """
    with open(tmp_path / "telescope.yaml", "w") as fl:
        fl.write(telfile)

    return tmp_path / "obsparams.yaml"


def test_vis_cli(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("vis-sim")
    cfg = get_config_files(outdir, 5, 2)

    args = f"--compress {outdir/'compression-cache.npy'} --normalize_beams --fix_autos"
    args += f" --profile {outdir}/profile.txt --log-level INFO"
    os.system(f"hera-sim-vis.py {str(cfg)} {DATA_PATH/'viscpu.yaml'} {args}")
    contents = os.listdir(outdir)
    assert "out.uvh5" in contents
