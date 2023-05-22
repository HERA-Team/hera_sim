API Reference
=============

.. testsetup::

    from hera_sim import *

Modules
-------
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    hera_sim.vis
    hera_sim.antpos
    hera_sim.defaults
    hera_sim.eor
    hera_sim.foregrounds
    hera_sim.interpolators
    hera_sim.io
    hera_sim.noise
    hera_sim.rfi
    hera_sim.sigchain
    hera_sim.simulate
    hera_sim.utils
    hera_sim.cli_utils
    hera_sim.components

Visibility Simulators
---------------------

Simulation Framework
++++++++++++++++++++

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    hera_sim.visibilities.simulators

Built-In Simulators
+++++++++++++++++++

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    hera_sim.visibilities.vis_cpu.VisCPU
    hera_sim.visibilities.pyuvsim_wrapper.UVSim
