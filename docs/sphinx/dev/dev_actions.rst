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

Workflows
---------

The github actions workflow files live in ``.github/workflows/`` and are documented in ``.github/workflows/README.md``. Refer to those files for details.

Job Selection
-------------

On pull requests, ``tools/ci_select_jobs.py`` reads the list of changed files and decides which test jobs run. Pushes to ``develop`` run every job.

- The CPU and Intel jobs are skipped when every changed file is one those builds do not compile or read: ``lib/cuda``, ``lib/hip``, ``lib/opencl``, ``lib/gpu``, ``julia/``, ``config/ci-tests/gpu_batch/``, ``docs/``, and markdown files. These are listed in ``CPU_NEED``.
- The GPU job runs when a changed file matches ``GPU_NEED`` (GPU libraries, build and CI files), or when the GPU build reaches it through ``#include`` from ``lib/core``, ``lib/metadata``, the GPU libraries, the kotekan executable, or the stages used by ``config/ci-tests/gpu_batch``. The stage list is read from those configs. It also runs when a changed source uses ``float16_t``, ``KOTEKAN_FLOAT16`` or a ``WITH_CUDA``, ``WITH_HIP`` or ``WITH_OPENCL`` conditional: the CPU builds compile such a file with a different ``float16_t``.
- The PTX job runs when a file under ``lib/cuda/generated`` changes. It assembles every generated kernel with ``ptxas`` from the ``nvidia-cuda-nvcc-cu12`` wheel through ``tools/check_ptx.sh``, on a GitHub runner without a GPU. The kotekan build never compiles these kernels, and the GPU job runs only the ones ``config/ci-tests/gpu_batch`` uses.

``tools/ci_select_jobs.py --self-test`` checks a table of cases; the Lint job runs it. Update the patterns, and add a case, when:

- a CPU build or test starts using a file in one of the directories ``CPU_NEED`` skips, e.g. a boost test that compiles a source from ``lib/cuda``;
- GPU code moves to a new directory;
- the GPU tests start depending on files outside the ``#include`` scan, such as data files read at runtime.
