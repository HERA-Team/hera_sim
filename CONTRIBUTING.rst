============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

Bug reports
===========

When `reporting a bug <https://github.com/HERA-Team/hera_sim/issues>`_ please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

Documentation improvements
==========================

``hera_sim`` could always use more documentation, whether as part of the
official ``hera_sim`` docs or in docstrings.

Feature requests and feedback
=============================

The best way to send feedback is to file an issue at https://github.com/HERA-Team/hera_sim/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions are welcome :)

Development
===========

To set up ``hera_sim`` for local development:

1. If you are *not* on the HERA-Team collaboration, make a fork of
   `hera_sim <https://github.com/HERA-Team/hera_sim>`_ (look for the "Fork" button).
2. Clone the repository locally. If you made a fork in step 1::

    git clone git@github.com:your_name_here/hera_sim.git

   Otherwise::

    git clone git@github.com:HERA-Team/hera_sim.git

3. Create a branch for local development (you will *not* be able to push to "master")::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. Make a development environment. We highly recommend using conda for this. With conda,
   just run::

    conda env create -n hera python=3
    pip install -e .[dev]
    pre-commit install

5. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

6. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests (run ``pytest``)
2. Update documentation when there's new API, functionality etc.
3. Add a note to ``CHANGELOG.rst`` about the changes.
4. Add yourself to ``AUTHORS.rst``.
