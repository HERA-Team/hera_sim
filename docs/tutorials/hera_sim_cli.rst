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

   $ cat ../config_examples/template_config.yaml


