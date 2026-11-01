#!/bin/bash
# Run the Comet integration tests against a completed CPU build.
set -euo pipefail
BUILD="$(realpath "$1")"
SRC="$(cd "$(dirname "$0")/../../.." && pwd)"
for HELPER in dataset_broker_producer dataset_broker_producer2 dataset_broker_consumer; do
    test -x "$BUILD/tests/boost/$HELPER"
done
test -x "$BUILD/kotekan/kotekan"
command -v redis-server >/dev/null
WORK="$(mktemp -d "${RUNNER_TEMP:-/tmp}/comet.XXXXXXXX")"
trap 'rm -rf -- "$WORK"' EXIT
PYTHON="$(command -v python3)"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 12), "Comet tests require Python 3.12 or newer"'
"$PYTHON" -m venv --without-pip "$WORK/venv"
export PATH="$WORK/venv/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
"$PYTHON" -m pip --python "$WORK/venv" install -r "$SRC/tests/ci-scripts/comet/requirements.txt"
"$PYTHON" -m pip --python "$WORK/venv" check
python3 -c 'import comet, redis; print("Comet", comet.__version__)'
command -v comet >/dev/null
export PYTHONPATH="$SRC/python${PYTHONPATH:+:$PYTHONPATH}"
export KOTEKAN_BUILD_DIRNAME="$BUILD"
export PYTEST_TIMEOUT=120
cd "$WORK"
python3 -m pytest -v -x -rs --timeout=120 --basetemp "$WORK/pytest" --junitxml="$WORK/results.xml" \
    -o cache_dir="$WORK/pytest-cache" \
    "$SRC/tests/test_dataset_broker.py" \
    "$SRC/tests/test_freqslicer_broker.py" \
    "$SRC/tests/test_vissharedmem_validation.py" \
    "$SRC/tests/test_vissharedmemreaderwriter.py"

python3 - "$WORK/results.xml" <<'PYTEST_RESULT'
import sys
import xml.etree.ElementTree as ET

suites = ET.parse(sys.argv[1]).getroot().iter("testsuite")
counts = {"tests": 0, "skipped": 0}
for suite in suites:
    for name in counts:
        counts[name] += int(suite.get(name, 0))
assert counts["tests"] >= 4 and counts["skipped"] == 0, counts
PYTEST_RESULT
