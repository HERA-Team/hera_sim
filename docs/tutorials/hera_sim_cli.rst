==========================================
Running ``hera_sim`` from the command line
==========================================

As of v0.2.0 of ``hera_sim``, quick-and-dirty simulations can be run from
the command line by creating a configuration file and using ``hera_sim``'s
``run`` command to create simulated data in line with the configuration's
specifications. The basic syntax of using ``hera_sim``'s command-line
interface is (this can be run from anywhere if ``hera_sim`` is installed):

.. runblock:: console

   $ hera-sim-simulate.py run --help

An example configuration file can be found in the ``config_examples``
directory of the repo's top-level directory. Here are its contents:

.. runblock:: console

   $ cat -n ../config_examples/template_config.yaml

The remainder of this tutorial will be spent on exploring each of the
items in the above configuration file.

BDA
---

The following block of text shows all of the options that must be specified
if you would like to apply BDA to the simulated data. Note that BDA is
applied at the very end of the script, and requires the BDA package to be
installed from http://github.com/HERA-Team/baseline_dependent_averaging.

.. runblock:: console

   $ sed -n 4,17p ../config_examples/template_config.yaml

Please refer to the ``bda.apply_bda`` documentation for details on what each
parameter represents. Note that practically each entry has the tag
``!dimensionful``; this YAML tag converts the entries in ``value`` and
``units`` to an ``astropy.units.quantity.Quantity`` object with the
specified value and units.

Filing
------

The following block of text shows all of the options that may be specified
in the ``filing`` section; however, not all of these *must* be specified.
In fact, the only parameter that is required to be specified in the config
YAML is ``output_format``, and it must be either ``miriad``, ``uvfits``,
or ``uvh5``. These are currently the only supported write methods for
``UVData`` objects.


.. runblock:: console

   $ sed -n 18,24p ../config_examples/template_config.yaml

Recall that ``run`` can be called with the option ``--outfile``; this
specifies the full path to where the simulated data should be saved and
overrides the ``outdir`` and ``outfile_name`` settings from the config
YAML. Additionally, one can choose to use the flag ``-c`` or ``--clobber``
in place of specifying ``clobber`` in the config YAML. Finally, the
dictionary defined by the ``kwargs`` entry has its contents passed to
whichever write method is chosen, and the ``save_seeds`` option should
only be used if the ``seed_redundantly`` option is specified for any of
the simulation components.

Setup
-----

The following block of text contains three sections: ``freq``, ``time``,
and ``telescope``. These sections are used to initialize the ``Simulator``
object that is used to perform the simulation. Note that the config YAML
shows all of the options that may be specified, but not all options are
necessarily required.

.. runblock:: console

   $ sed -n 26,53p ../config_examples/template_config.yaml

If you are familiar with using configuration files with ``pyuvsim``, then
you'll notice that the sections shown above look very similar to the way
config files are constructed for use with ``pyuvsim``. The config files
for ``run`` were designed as an extension of the ``pyuvsim`` config files,
with the caveat that some of the naming conventions used in ``pyuvsim``
are somewhat different than those used in ``hera_sim``. For information
on the parameters listed in the ``freq`` and ``time`` sections, please
refer to the documentation for ``hera_sim.io.empty_uvdata``. As for the
``telescope`` section, this is where the antenna array and primary beam
are defined. The ``array_layout`` entry specifies the array, either by
specifying an antenna layout file or by using the ``!antpos`` YAML tag
and specifying the type of array (currently only ``linear`` and ``hex``
are supported) and the parameters to be passed to the corresponding
function in ``hera_sim.antpos``. The ``omega_p`` entry is where the
primary beam is specified, and it is currently assumed that the beam
is the same for each simulation component (indeed, this simulator is not
intended to produce super-realistic simulations, but rather perform
simulations quickly and give somewhat realistic results). This entry
defines an interpolation object to be used for various ``hera_sim``
functions which require such an object; please refer to the documentation
for ``hera_sim.interpolators.Beam`` for more information. Future versions
of ``hera_sim`` will provide support for specifying the beam in an
antenna layout file, similar to how it is done by ``pyuvsim``.

Defaults
--------

This section of the configuration file is optional to include. This
section gives the user the option to use a default configuration to
specify different parameters throughout the codebase. Users may define
their own default configuration files, or they may use one of the
provided season default configurations, located in the ``config`` folder.
The currently supported season configurations are ``h1c`` and ``h2c``.
Please see the ``defaults`` module/documentation for more information.

.. runblock:: console

   $ sed -n 54,57p ../config_examples/template_config.yaml

Systematics
-----------

This is the section where any desired systematic effects can be specified.
The block of text shown below details all of the possible options for
systematic effects. Note that currently the ``sigchain_reflections`` and
``gen_cross_coupling_xtalk`` sections cannot easily be worked with; in
fact, ``gen_cross_coupling_xtalk`` does not work as intended (each
baseline has crosstalk show up at the same phase and delay, with the same
amplitude, but uses a different autocorrelation visibility). Also note
that the ``rfi`` section is subject to change, pending a rework of the
``rfi`` module.

.. runblock:: console

   $ sed -n 58,96p ../config_examples/template_config.yaml

Note that although these simulation components are listed under
``systematics``, they do not necessarily need to be listed here; the
configuration file is formatted as such just for semantic clarity. For
information on any particular simulation component listed here, please
refer to the corresponding function's documentation. For those who may
not know what it means, ``!!null`` is how ``NoneType`` objects are
specified using ``pyyaml``.

Sky
---

This section specifies both the sky temperature model to be used
throughout the simulation as well as any simulation components which are
best interpreted as being associated with the sky (rather than as a
systematic effect). Just like the ``systematics`` section, these do not
necessarily need to exist in the ``sky`` section (however, the
``Tsky_mdl`` entry *must* be placed in this section, as that's where the
script looks for it).

.. runblock:: console

   $ sed -n 97,130p ../config_examples/template_config.yaml

As of now, ``run`` only supports simulating effects using the functions
in ``hera_sim``; however, we intend to provide support for using
different simulators in the future. If you would like more information
regarding the ``Tsky_mdl`` entry, please refer to the documentation for
the ``hera_sim.interpolators.Tsky`` class. Finally, note that the
``seed_redundantly`` parameter is specified for each entry in ``eor``
and ``foregrounds``; this parameter is used to ensure that baselines
within a redundant group all measure the same visibility, which is a
necessary feature for data to be absolutely calibrated. Please refer
to the documentation for ``hera_sim.eor`` and ``hera_sim.foregrounds``
for more information on the parameters and functions listed above.

Simulation
----------

This section is used to specify which of the simulation components to
include in or exclude from the simulation. There are only two entries
in this section: ``components`` and ``exclude``. The ``components``
entry should be a list specifying which of the groups from the ``sky``
and ``systematics`` sections should be included in the simulation. The
``exclude`` entry should be a list specifying which of the particular
models should not be simulated. Here's an example:

.. runblock:: console

   $ sed -n -e 137,138p -e 143,150p ../config_examples/template_config.yaml

The entries listed above would result in a simulation that includes all
models contained in the ``foregrounds``, ``noise``, ``eor``, ``rfi``,
and ``sigchain`` dictionaries, except for the ``sigchain_reflections``
and ``gen_whitenoise_xtalk`` models. So the simulation would consist of
diffuse and point source foregrounds, thermal noise, noiselike EoR, all
types of RFI modeled by ``hera_sim``, and bandpass gains, with the
effects simulated in that order. It is important to make sure that
effects which enter multiplicatively (i.e. models from ``sigchain``)
are simulated *after* effects that enter additively, since the order
that the simulation components are listed in is the same as the order
of execution.
