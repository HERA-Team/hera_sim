def get_config_file():
    return """
'filing':
  'outdir': {outdir}
  'outfile_name': 'out'
  'output_format': 'uvh5'
  'clobber': true
'freq':
  'Nfreqs': 1
  'channel_width': 97656.25
  'start_freq': 46920776.3671875
'sources':
  'catalog': '{DATA_PATH}/fch0000.skyh5'
'telescope':
  'array_layout': '{DATA_PATH}/H4C-antennas.csv'
  'telescope_config_name': '{DATA_PATH}/h4c_idr2.1_teleconfig.yaml'
  'select':
    'freq_buffer': 3000000.0
'time':
  'Ntimes': 1000
  'integration_time': 4.986347833333333
  'start_time': 2458208.916228965
'polarization_array':
  - -5
  - -7
  - -8
  - -6
    """


def test_vis_cli(config_file):
    pass
    # os.system(f"hera-sim-simulate.py {str(config_file)} --save_all --verbose")
    # outdir = config_file.parent
    # contents = os.listdir(outdir)
    # assert "test.uvh5" in contents
    # assert "test_gains.calfits" in contents
    # assert "test_thermal_noise.uvh5" in contents
    # assert "test_diffuse_foreground.uvh5" in contents

    # # Quick sanity check
    # uvd = UVData()
    # uvd.read(outdir / "test.uvh5")
    # assert np.all(uvd.data_array != 0)
    # assert uvd.Nfreqs == 100
    # assert uvd.Ntimes == 20
    # assert uvd.Nants_data == 2
    # cal = UVCal()
    # cal.read_calfits(outdir / "test_gains.calfits")
    # assert np.all(cal.gain_array != 1)
