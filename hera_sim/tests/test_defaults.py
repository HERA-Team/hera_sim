from os.path import join
import numpy as np
from hera_sim.defaults import defaults
from hera_sim.config import CONFIG_PATH
from hera_sim.sigchain import gen_bandpass
from hera_sim.noise import bm_poly_to_omega_p

if defaults._version_is_compatible:
    def test_config_swap():
        defaults.set('h1c')
        config1 = defaults._config
        defaults.set('h2c')
        assert config1 != defaults._config

    def test_direct_config_path():
        config = join(CONFIG_PATH, 'HERA_H2C_CONFIG.yaml')
        defaults.set(config)

    def test_beam_poly_changes():
        defaults.set('h1c')
        fqs = np.linspace(0.1,0.2,100)
        omega_p = bm_poly_to_omega_p(fqs)
        defaults.set('h2c')
        assert not np.all(omega_p==bm_poly_to_omega_p(fqs))

    def test_bandpass_changes():
        defaults.set('h1c')
        fqs = np.linspace(0.1,0.2,100)
        np.random.seed(0)
        bp = gen_bandpass(fqs, [0])[0]
        defaults.set('h2c')
        np.random.seed(0)
        assert not np.all(bp==gen_bandpass(fqs,[0])[0])
        defaults.deactivate()

    def test_activate_and_deactivate():
        defaults.activate()
        assert defaults._use_season_defaults
        defaults.deactivate()
        assert not defaults._use_season_defaults
