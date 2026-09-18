This is a high-performance, soft-realtime radio astronomy pipeline code. This is an older codebase, maintained by a number of developers using different styles and conventions. Below are some guidelines for editing this code. These are not firm rules.


Development
-----------

The code itself is intended to run on many nodes, cores, and GPUs simultaneously. It should avoid bottlenecks, race conditions, and memory leaks. Because this is a complex code, it is also best to fully understand code before editing. Kotekan is run under a service daemon in production, so instances should prefer to FATAL_ERROR or abort rather than passing along suspect data; the daemon will restart the instance.

When developing code, most edits and additions should be "lazy", except when it comes to understanding code and validation/testing. It is advisable to read through existing code and docs to ensure you understand both what has been implemented and its intent. Then, the ladder below can be followed heuristically to determine how to make edits:
1. Does the feature need to exist at all, or is there a simplification that subsumes it?
2. Is the feature already in this codebase? If so, use it.
3. Are there native platform features that cover this? If so, use it.
4. Do existing dependencies solve it? Prefer against adding new dependencies.
5. Is it a fairly simple one-liner? Use one line.
6. Only then: proceed to edit. After ensuring you understand the changes needed, code edits should aim to be "lazy". That is: no boilerplate or scaffolding "for later", prefer deletion over addition, boring code over clever (e.g. avoid lambdas or nested structures where a named function or plain loop reads as well), minimal abstraction (e.g. no interfaces with one implementation, no factory for one product). New code should follow kotekan idioms, followed by native language or package idioms. However, code style may be chosen to reflect particular file(s) being edited due to kotekan's diverse base of maintainers.

If there are latent bugs, these should be reported and corrected in a separate set of changes (separate PR), unless the bug is in code the change already touches. Check with care to see if there are consequences of changes in other code or tests, since code paths may have developed assuming the bug's presence. Older code, or code following practices uncommon in c++ (e.g. c), can and should be updated, but only if we're already touching that code. Changes should be checked for side effects and behavior changes, even minor, seemingly benign changes such as output frame ordering, since there may be downstream consequences.

Doxygen strings should exist where appropriate. Higher-level sphinx documentation is also available, and should be updated if appropriate. Code comments should generally be kept lean and explanatory, not verbose. Comments explaining past behavior, debugging steps, why code was removed, or previous functionality, are not generally needed, and can live in commit messages or PR descriptions.

When reviewing, attempt to understand both the intent of previously existing code, and incoming changes, before providing feedback. Rather than merely providing feedback, explicit suggestions for changes are preferred.

Review language, comments, and documentation should all use concise, professional technical English in the active voice. They should use existing kotekan vocabulary if relevant (see the glossary and repository layout in README.md), avoid introducing new terms if possible, and avoid "code-jockey" slang and buzzwords. Acronyms and eponyms are disfavored if a common generic name exists, including for variable and function names. Code style is enforced by a set of lint scripts, which the CI checks and runs. Some additional guidelines (these are not _firm_ rules) can be found in docs/sphinx/dev/dev_style_guide_code.rst.


Compiling
---------

cmake is used, generally from the `build` directory. Specific `build-*` directories may be made to test clean builds with certain features, although these should generally be cleaned up if made, and never committed. The CI containers are the exception: they expect the repo at /code/kotekan and the build directory named build-2404. Typically gcc (default compiler) is used, although the intel and clang compilers should be supported.

The kotekan binary can be compiled as a whole, or specific targets compiled, if relevant for specific debugging. If editing a specific piece of code, it may be faster to select a specific compile target rather than all of kotekan.

The kotekan binary lives at <build>/kotekan/kotekan (bare <build>/kotekan is a directory).


Testing
-------

Before running tests, a quick pass for syntactic or logical issues is appropriate.

Before pushing a final code version, it can be a good idea to run (only) select, relevant tests. The full test suite does not need to be run every time, and individual commits within a PR do not all need to pass tests.

Python tooling exists in /opt/kotekan_env; activate it, or put its bin directory first on PATH.

There are many boost tests in tests/boost, which are built with -DWITH_BOOST_TESTS=ON and run with run_boost_tests.sh.

There are pytest tests in the tests directory, however development of new pytests is discouraged in favor of yaml or boost tests. This is because of the extra layer of abstraction and care required in the pytest ecosystem -- errors hidden by default, tests may be silently skipped, etc.

Shell scripts in tests/ci-scripts/push_pull are run as part of CI tests, and are for testing more complex pipelines. The configs in config/ci-tests/gpu_batch and config/ci-tests/cpu_batch are also run with the kotekan executable as part of the tests. There is a script, config/ci-tests/run_tests.sh, that runs the latter of these.

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

# Pytest (in a container on a large host use -n 4, never -n auto: it exhausts the container pid limit on large hosts)
PYTHONPATH=$PWD/python pytest -v -x -rs -m serial tests/[test_<name>.py]
PYTHONPATH=$PWD/python pytest -v -x -rs -n 4 --dist=loadfile -m 'not serial' tests/

# Lint (what CI runs: clang-format-18 + black, then git diff --exit-code)
tools/lint.sh
clang-format-18 -i <touched files>

# Reproduce CI in a container (podman; docker daemon is not available)
podman run --rm -it -v $PWD:/code/kotekan -w /code/kotekan localhost/u2404-cpu bash   # localhost/u2404 for GPU
touch /.dockerenv            # inside: makes baseband tests self-skip as they do in CI

# Git / GitHub policy
#   `develop` branch changes via PRs; `chord` is the CHORD production branch.
git worktree add <scratch>/wt-<x> -b <user>/<x> origin/develop   # isolate work from the shared checkout
gh pr view <n> --json state,headRefOid                           # before every push to a PR branch
git push --force-with-lease=<branch>:<old-sha> origin <branch>
gh pr create --base develop --body-file <file>
gh run list --branch develop --workflow kotekan-ci-tests         # check develop itself before chasing a PR failure

# Runtime introspection (default REST port 12048)
curl -s localhost:12048/buffers | python3 -m json.tool
curl -s 'localhost:12048/buffer_frame?name=<buffer>'
```
