"""Basic tests of the hera-sim-simulate script."""

import os

import numpy as np
import pytest
from pyuvdata import UVCal, UVData


@pytest.fixture(scope="session")
def config_file(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("test_data")
    cfg_file = tmpdir / "config.yaml"
    cfg_file.write_text(
        f"""
        filing:
            outdir: {str(tmpdir)}
            outfile_name: test.uvh5
            output_format: uvh5
            clobber: True
        freq:
            Nfreqs: 100
            channel_width: !!float 1e5
            start_freq: !!float 100e6
        time:
            Ntimes: 20
            integration_time: 10.7
            start_time: 2458799.0
        telescope:
            array_layout:
                0: [0, 0, 0]
                1: [14.6, 0, 0]
            omega_p: !Beam
                datafile: HERA_H1C_BEAM_POLY.npy
                interp_kwargs:
                    interpolator: poly1d
        systematics:
            sigchain:
                gains:
                    seed: once
                    bp_poly: HERA_H1C_BANDPASS.npy
            noise:
                thermal_noise:
                    seed: initial
                    Trx: 100
        sky:
            Tsky_mdl: !Tsky
                datafile: HERA_Tsky_Reformatted.npz
                interp_kwargs:
                    pol: xx
            foregrounds:
                diffuse_foreground:
                    seed: redundant
        simulation:
            components:
                - foregrounds
                - noise
                - sigchain
        """
    )
    return cfg_file


def test_cli(config_file):
    os.system(f"hera-sim-simulate.py {str(config_file)} --save_all --verbose")
    outdir = config_file.parent
    contents = os.listdir(outdir)
    assert "test.uvh5" in contents
    assert "test_gains.calfits" in contents
    assert "test_thermal_noise.uvh5" in contents
    assert "test_diffuse_foreground.uvh5" in contents

    # Quick sanity check
    uvd = UVData()
    uvd.read(outdir / "test.uvh5")
    assert np.all(uvd.data_array != 0)
    assert uvd.Nfreqs == 100
    assert uvd.Ntimes == 20
    assert uvd.Nants_data == 2
    cal = UVCal()
    cal.read_calfits(outdir / "test_gains.calfits")
    assert np.all(cal.gain_array != 1)
