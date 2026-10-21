# gnss-broker

The CHORD GNSS broker and the Python around it: the broker (`python/scripts/gnss/gnss_broker/`,
run by `scripts/gnss/broker_multi.py`), the observables writers and the analysis, beam-cube and
nav-decoding modules (`python/scripts/gnss/`), the tools and gates (`scripts/gnss/`), the site
deployment with its systemd units, launchers and cron jobs (`scripts/gnss/site/`), the node
config generator (`config/gen_chord_gnss_config.py`, `scripts/gnss/gen_fleet.py`) and the
browser viewer (`python/scripts/js_viewer/`).

The GNSS stages are C++ in kotekan (`lib/stages/gnss/`). Kotekan also keeps the C++ harnesses
in its `scripts/gnss/` with their build script, and the twelve Python files that only check its
C++, listed in `scripts/gnss/run_tests.py`'s `KOTEKAN_ONLY`: the code-table cross-checks
`python/scripts/gnss/{b1i_b3i,b2b,e5b,e6,glonass_ca,l1cd,l2oc,l3oc}_code_check.py` and
`python/scripts/gnss/test_code_check_cpp.py`, and `scripts/gnss/trim_sign_gate.py`,
`scripts/gnss/test_trim_sign_gate.py` and `scripts/gnss/test_e2e_exact_seed.py`.
`python/scripts/gnss/kotekan_src.py` is in both repositories, since those checks find kotekan's
sources through it too.

## Layout

Every file keeps the path it had in kotekan, where this code grew. Its history came with it, so
`git log` and `git blame` reach back past the split, and the paths in commit messages, the docs
and the bug list still name real files. The modules also find each other by relative path from
their own location: `broker_multi.py` puts `python/scripts/gnss` on `sys.path`, and the
generator reads `config/` beside it.

The commit SHAs that the docs, the bug list and the comments cite from before the split are
kotekan's, and resolve in kotekan/kotekan. `docs/kotekan_commit_map.txt` maps each one that came
along to its commit here. `scripts/gnss/split_paths.txt` is the recipe the repository was
extracted with.

## Finding kotekan

The broker parses its signal table from kotekan's `lib/stages/gnss/gnssSignal.hpp` when it is
imported, the generator renders kotekan's `config/chord/pathfinder.j2` and reads its headers,
and the site scripts run kotekan's builds. All of them find the kotekan tree by one rule
(`python/scripts/gnss/kotekan_src.py`, and `scripts/gnss/site/kotekan_src.sh` for the shell),
the first of these that holds `lib/stages/gnss`:

    $KOTEKAN_SRC            when set, taken as it is
    <checkout>              kotekan and this code in one tree
    <checkout>/kotekan      a kotekan submodule
    <checkout>/../kotekan   a kotekan checkout beside this one

On the gnss VM this checkout is `/home/kvand/gnss/gnss-broker`, beside
`/home/kvand/gnss/kotekan`, so nothing needs to be set. With no kotekan tree, the broker and the
generator fail at import, and `gen_fleet.py` exits 2 when it needs kotekan, each naming
`KOTEKAN_SRC` and the places tried.

## Environments

Two venvs, as `scripts/gnss/site/stack_components.sh` runs the stack (`GNSS_VENV_FT`,
`GNSS_VENV`):

    uv python install 3.14t
    uv venv --python 3.14t venv-ft          # the broker: free-threaded CPython 3.14
    uv pip install --python venv-ft/bin/python -r requirements-broker-ft.txt -r requirements-dev.txt

    uv venv --python /usr/bin/python3.12 venv   # everything else
    uv pip install --python venv/bin/python -r requirements.txt -r requirements-dev.txt

The requirements files pin what the production venvs hold. `pyproject.toml` lists the same
dependencies as lower bounds, for `uv pip install -r pyproject.toml --extra tools --extra viewer`;
it is not an installable package, since the code runs from the checkout. `/usr/bin/python3` runs
`bad_inputs_cron.py`, `rail_watch.py` and `node_up.sh`'s `gen_fleet.py`, with apt's
`python3-yaml` and `python3-jinja2`. The tests also need node (apt `nodejs`), a C++ compiler and
git. `python/scripts/js_viewer/ccera_rest.py`, the viewer's upstream pointing helper, needs
tornado, which neither venv has: these environments do not support it.

## Tests

    /usr/bin/python3 scripts/gnss/run_tests.py -j 8 [--only REGEX] [--strict] [--out DIR]

runs every test, each under the interpreter it is written for (`GNSS_VENV`, `GNSS_VENV_FT` and
`GNSS_SYS_PY` override them), and prints a PASS/FAIL/SKIP/TIMEOUT table; its docstring has the
rules. Set `KOTEKAN_SRC` when kotekan is not beside the checkout.

## Docs

`docs/CHORD_GNSS_RUNBOOK.md` is the operations runbook; `docs/CHORD_BUGLIST.md` holds the open items;
`scripts/gnss/site/README.md` covers deployment. The other `docs/CHORD_*.md` and
`docs/gnss_*.md` each cover one subsystem.
