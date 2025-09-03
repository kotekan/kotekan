************
Actions
************

Running GitHub Actions Locally
------------------------------

When submitting pull requests, various checks are run using GitHub actions. These do not run quickly on GitHub's servers, so it may take some time to know if these checks pass.

With `act <https://github.com/nektos/act>`_ installed, you can run GitHub actions locally! You can run all workflows just by running the command ``act``. For GitHub authentication you may also need to install `GitHub CLI <https://cli.github.com/>`_.

The following command will authenticate (providing secrets with the ``-s`` flag), and run a basic 2204 build (use the ``-j`` flag to run a specific job).

.. code-block:: bash

    act -s GITHUB_TOKEN="$(gh auth token)" -j "build-base-2204"

