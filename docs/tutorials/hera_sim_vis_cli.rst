==============================================
Running ``hera-sim-vis`` from the command line
==============================================

As of v2.4.0 of ``hera_sim``, we have included a command-line interface for performing
visibility simulations using *any* compatible visiblity simulator. The interface for the
script is (this can be run from anywhere if ``hera_sim`` is installed):

.. runblock:: console

   $ hera-sim-vis.py --help

Two main configuration files are required. The first is an "obsparam" file, which should
follow the formatting defined by `pyuvsim <https://pyuvsim.readthedocs.io/en/latest/parameter_files.html>`_.
As described in the `visibility simulator <visibility_simulator.html>`_ tutorial, the ``hera_sim.visibilities``
module provides a universal interface between this particular configuration setup and a number of
particular simulators.

To specify the simulator to be used, the second configuration file must be provided.
An example of this configuration file, defined for the ``VisCPU`` simulator, can be
found in the repo's `config_examples <https://github.com/HERA-Team/hera_sim/tree/main/config_examples>`_
directory. Here are its contents:

.. runblock:: console

   $ cat -n ../config_examples/simulator.yaml

Notice that the file contains a ``simulator:`` entry. This must be the name of a simulator
derived from the base :class:`VisibilitySimulator` class provided in ``hera_sim``.
Usually,  this will be one of the built-in simulators listed in the
`API reference </reference/index.html#built-in-simulators>`_ under "Built-In Simulators".


However, it may also be a custom simulator, defined outside of ``hera_sim``, so long as
it inherits from :class:`VisibilitySimulator`. To use one of these via command line, you
need to provide the dot-path to the class, so for example: ``my_library.module.MySimulator``.

The remaining parameters are passed to the constructor for the given simulator. So, for
example, for the ``VisCPU`` simulator, all available parameters are listed under the
`Parameters section </reference/_autosummary/hera_sim.visibilities.vis_cpu.VisCPU.html#hera_sim.visibilities.vis_cpu.VisCPU>`_
of its class API. In general, the class may do some transformation of the YAML inputs
before constructing its instance, using the ``from_yaml_dict()`` method. Check out the
documentation for that method for your particular simulator to check if it requires any
transformations (eg. if it required data loaded from a file, it might take the filename
from the YAML, and read the file in its ``from_yaml_dict()`` method before constructing
the object with the full data).
