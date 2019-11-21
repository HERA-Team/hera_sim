==========================================
Running ``hera_sim`` from the command line
==========================================

As of v0.2.0 of ``hera_sim``, quick-and-dirty simulations can be run from 
the command line by creating a configuration file and using ``hera_sim``'s 
``run`` command to create simulated data in line with the configuration's 
specifications. The basic syntax of using ``hera_sim``'s command-line 
interface is: 

.. runblock:: console

   $ hera_sim run --help

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
and ``telescope``; these sections are used to initialize the ``Simulator`` 
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
for ``hera_sim.interpolators.Beam`` for more information.
