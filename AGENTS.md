This is a high-performance, soft-realtime radio astronomy pipeline code. This is an older codebase, maintained by a number of developers using different styles and conventions. Below are some guidelines for editing this code. Rules written as "must" are firm; "should" means a strong default.


Development
-----------

The code itself is intended to run on many nodes, cores, and GPUs simultaneously. It should avoid bottlenecks, race conditions, and memory leaks. Read the code you change and its callers before editing. Kotekan is run under a service daemon in production, so instances should prefer to FATAL_ERROR or abort rather than passing along suspect data; the daemon will restart the instance. `FATAL_ERROR` raises SIGTERM and then throws, so it must not be called from a destructor or a `noexcept` function.

When developing code, most edits and additions should be minimal, except when it comes to understanding code and validation/testing. Read the existing code and docs so you understand both what is implemented and its intent. Then follow this ladder to decide how to edit:
1. Does the feature need to exist at all, or is there a simplification that subsumes it?
2. Is the feature already in this codebase? If so, use it.
3. Are there native platform features that cover this? If so, use it.
4. Do existing dependencies solve it? Prefer against adding new dependencies.
5. Is it a fairly simple one-liner? Use one line.
6. Only then: proceed to edit. After ensuring you understand the changes needed, code edits should be minimal. That is: no boilerplate or scaffolding "for later", prefer deletion over addition, boring code over clever (e.g. avoid lambdas or nested structures where a named function or plain loop reads as well), minimal abstraction (e.g. no interfaces with one implementation, no factory for one product). New code should follow kotekan idioms, followed by native language or package idioms. However, code style may be chosen to reflect particular file(s) being edited due to kotekan's diverse base of maintainers.

Fix unrelated bugs in a separate PR. Report a latent bug you notice; fix it in the same PR only if the change already touches that code. Check with care to see if there are consequences of changes in other code or tests, since code paths may have developed assuming the bug's presence. Older code, or code following practices uncommon in C++, can be updated in the same PR. Changes should be checked for side effects and behavior changes, even minor, seemingly benign changes such as output frame ordering, since there may be downstream consequences.

Doxygen strings should exist where appropriate. Higher-level sphinx documentation is also available, and should be updated if appropriate. Code comments should generally be kept lean and explanatory, not verbose. Comments explaining past behavior, debugging steps, why code was removed, or previous functionality, are not generally needed, and can live in commit messages or PR descriptions.

When reviewing, attempt to understand both the intent of previously existing code, and incoming changes, before providing feedback. Rather than merely providing feedback, explicit suggestions for changes are preferred.


Writing
-------

Write code comments, commit messages, PR descriptions, and review comments the way a kotekan maintainer would: short, plain, active voice, and professional technical English. Use names and vocabulary already in the code and the README glossary, and keep kotekan's own acronyms (FRB, N2, RFI, FPGA). Do not coin new names, acronyms, or eponyms. Avoid restating the diff, headings and tables in short comments, stacked hedges, unsupported praise ("robust", "comprehensive"), and code comments that narrate the change. A PR description should say what changed, why, and how it was tested, in a few sentences or a short bullet list.

Lint scripts enforce code style and CI runs them. Further guidelines are in docs/sphinx/dev/dev_style_guide_code.rst.


Compiling
---------

Build with cmake, usually in the `build` directory. Specific `build-*` directories may be made to test clean builds with certain features, although these should be cleaned up afterwards and must not be committed. The CI containers are the exception: they expect the repo at /code/kotekan and the build directory named build-2404. Typically gcc (default compiler) is used, although the intel and clang compilers should be supported.

The kotekan binary can be compiled as a whole, or specific targets compiled, if relevant for specific debugging. If editing a specific piece of code, it may be faster to select a specific compile target rather than all of kotekan.

The kotekan binary lives at <build>/kotekan/kotekan (bare <build>/kotekan is a directory).


Testing
-------

Check syntax and logic in changed code before running tests.

Run the relevant tests before pushing a final PR version, not the full suite. Individual commits within a PR need not pass tests.

Python tooling may exist in /opt/kotekan_env; if so, activate it, or put its bin directory first on PATH.

There are three kinds of tests. Pick the lowest level that exercises the change:
1. Boost tests in tests/boost, built with -DWITH_BOOST_TESTS=ON and run with run_boost_tests.sh. Use these for code-level coverage: functions and classes, single stages, and a few stages chained together. They can read written files back, and can call an external reference tool where an independent check is needed, as test_timeUtil does with astropy.
2. Yaml configs in config/ci-tests/cpu_batch and gpu_batch, run through the kotekan executable by config/ci-tests/run_tests.sh. Use these for full-pipeline tests of one kotekan instance.
3. Shell scripts in tests/ci-scripts/push_pull. Use these to run several kotekan instances and test their interactions.

There are also pytest tests in the tests directory. New pytests are discouraged unless specifically requested: pytest adds a layer of abstraction that hides errors by default and can skip tests silently.

Many of these tests are run as part of github actions CI. We have a limited amount of hardware to run GPU tests locally, so a more minimal set of tests covers those runs.

If tests generate output files, this should be cleaned up, or directed to /tmp.

If changes affect config parameters, existing configs and tests may need to be updated.


Common Commands
---------------

The set of commands below is a quick reference that may be useful when working on hosts with the CHORD development environment (`/opt/kotekan_env` indicates the environment is present, and we are likely on one of those nodes).

<build> is the cmake build directory (`build` locally, `build-2404` in the CI containers).

```bash
# Configure and build (CI's gcc Test configuration; add -DUSE_CUDA=ON for GPU, -DCMAKE_BUILD_TYPE=Release for timing)
cmake -S . -B <build> -DCMAKE_BUILD_TYPE=Test -DWITH_BOOST_TESTS=ON -DWERROR=ON -DCCACHE=ON
cmake --build <build> -j$(expr $(nproc) / 2) [--target kotekan_core]

# Config checks: enough on their own for config- or comment-only edits
<build>/kotekan/kotekan --check-config config/<file>.yaml      # or --dry-run
config/ci-tests/run_tests.sh <build>/kotekan/kotekan 2m config/ci-tests/cpu_batch   # or gpu_batch

# Boost tests (PATH prefix needed: test_timeUtil shells out to python with astropy)
PATH=/opt/kotekan_env/bin:$PATH tests/boost/run_boost_tests.sh -v -t 30 <build>/tests/boost
<build>/tests/boost/test_<name>                                 # one test binary

# Pytest (KOTEKAN_BUILD_DIRNAME is needed unless the build directory is literally `build`;
#         in a container on a large host use -n 4, never -n auto: it exhausts the container pid limit)
KOTEKAN_BUILD_DIRNAME=<build> PYTHONPATH=$PWD/python pytest -v -x -rs -m serial tests/[test_<name>.py]
KOTEKAN_BUILD_DIRNAME=<build> PYTHONPATH=$PWD/python pytest -v -x -rs -n 4 --dist=loadfile -m 'not serial' tests/

# Lint (what CI runs: clang-format-18 + black, then git diff --exit-code)
tools/lint.sh
clang-format-18 -i <touched files>

# Reproduce CI in a container (podman; docker daemon is not available)
podman run --rm -it -v $PWD:/code/kotekan -w /code/kotekan localhost/u2404-cpu bash   # localhost/u2404 for GPU

# Git / GitHub policy
#   `develop` must change only via PRs; `chord` is the CHORD production branch.
git worktree add <scratch>/wt-<x> -b <user>/<x> origin/develop   # isolate work from the shared checkout
gh pr view <n> --json state,headRefOid                           # before every push to a PR branch
git push --force-with-lease=<branch>:<old-sha> origin <branch>
gh pr create --base develop --body-file <file>
gh run list --branch develop --workflow kotekan-ci-tests         # check develop itself before chasing a PR failure

# Runtime introspection (default REST port 12048)
curl -s localhost:12048/buffers | python3 -m json.tool
curl -s 'localhost:12048/buffer_frame?name=<buffer>'
```
