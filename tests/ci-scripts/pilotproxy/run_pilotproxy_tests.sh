#!/bin/bash
# PilotProxy DTV detector checks: vendored-copy integrity, then the CUDA pipeline tests with the
# CHORD (64 samples per window) and CHIME (128) runtime bundles.
# Usage: run_pilotproxy_tests.sh <build dir>
set -euo pipefail
BUILD="$(realpath "$1")"
SRC="${GITHUB_WORKSPACE:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export PYTHONPATH="$SRC/python${PYTHONPATH:+:$PYTHONPATH}"
WORK="$(mktemp -d "${RUNNER_TEMP:-/tmp}/pilotproxy.XXXXXXXX")"
trap 'rm -rf -- "$WORK"' EXIT
export PYTHONDONTWRITEBYTECODE=1
# pilot-proxy comes from the environment (the CI image installs it), at the vendored commit.
python3 - "$SRC/external/pilotproxy" <<'PY'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "VENDOR.json").read_text())
for name, expected in manifest["files"].items():
    actual = "sha256:" + hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        sys.exit(f"vendored PilotProxy digest mismatch: {name}")
try:
    installed = importlib.metadata.distribution("pilot-proxy")
except importlib.metadata.PackageNotFoundError:
    sys.exit("pilot-proxy is not installed in this Python environment")
source = json.loads(installed.read_text("direct_url.json") or "{}")
url = source.get("url", "")
commit = manifest["upstream_commit"]
if commit not in url and source.get("vcs_info", {}).get("commit_id") != commit:
    sys.exit(f"pilot-proxy is installed from {url or 'an index'}, not at the vendored commit {commit}")
PY
PP_CONFIGS="$(python3 -c 'from pilot_proxy.paths import CONFIGS_DIR; print(CONFIGS_DIR)')"
for RECEIVER in chord chime; do
    pilot-proxy export-runtime-weight-bundle \
        --receiver-profile "$PP_CONFIGS/receiver_profiles/${RECEIVER}_dtv_fengine.json" \
        --detector-core-profile "$PP_CONFIGS/detector_core/pilotproxy_cuda_local_reference_power_ratio.json" \
        --weight-coordinate-system post_nyquist_zone_normalization \
        --physical-channel-range 14:36 --output-dir "$WORK/pilotproxy_bundle_$RECEIVER"
done
# Only the tests that run the CUDA pipeline: the template tests run in the CPU jobs, and the
# production geometry soak is run by hand (dev_pilotproxy_testing.rst).
export PILOTPROXY_TEST_BINARY="$BUILD/kotekan/kotekan"
status=0
PILOTPROXY_TEST_BUNDLE="$WORK/pilotproxy_bundle_chord" \
    timeout --signal=TERM --kill-after=10s 3m python3 -m pytest -v -rs --durations=0 --basetemp "$WORK/pytest_chord" -o cache_dir="$WORK/pytest-cache" "$SRC/tests/test_pilotproxy_integration.py" "$SRC/tests/test_dtv_rfi_mask.py" \
    -k 'not template and not chime and not soak' || status=1
PILOTPROXY_TEST_BUNDLE="$WORK/pilotproxy_bundle_chime" \
    timeout --signal=TERM --kill-after=10s 3m python3 -m pytest -v -rs --durations=0 --basetemp "$WORK/pytest_chime" -o cache_dir="$WORK/pytest-cache" "$SRC/tests/test_dtv_rfi_mask.py" \
    -k 'chime_detector_mask' || status=1
exit $status
