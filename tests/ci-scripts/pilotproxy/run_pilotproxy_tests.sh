#!/bin/bash
# PilotProxy DTV detector checks: vendored-copy integrity, the CHORD (64 samples per window) and
# CHIME (128) runtime bundles, the pipeline verifier and the integration tests.
# Usage: run_pilotproxy_tests.sh <build dir>
set -euo pipefail
BUILD="$(realpath "$1")"
SRC="${GITHUB_WORKSPACE:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export PYTHONPATH="$SRC/python${PYTHONPATH:+:$PYTHONPATH}"
WORK="$(mktemp -d "${RUNNER_TEMP:-/tmp}/pilotproxy.XXXXXXXX")"
trap 'rm -rf -- "$WORK"' EXIT
PYTHON="$(command -v python3)"
"$PYTHON" -m venv --without-pip "$WORK/venv"
export PATH="$WORK/venv/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_REQUIRE_VIRTUALENV=true
"$PYTHON" -m pip --python "$WORK/venv" install --no-deps -r "$SRC/tests/ci-scripts/pilotproxy/requirements.txt"
python3 - "$SRC/external/pilotproxy" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "VENDOR.json").read_text())
for name, expected in manifest["files"].items():
    actual = "sha256:" + hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        sys.exit(f"vendored PilotProxy digest mismatch: {name}")
PY
PP_COMMIT="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["upstream_commit"])' "$SRC/external/pilotproxy/VENDOR.json")"
PP_REPO="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["upstream_repo"])' "$SRC/external/pilotproxy/VENDOR.json")"
# The bundle exporter and the Python reference are installed from the pilot-proxy
# package (upstream_repo in VENDOR.json). The detector core is vendored in external/pilotproxy.
"$PYTHON" -m pip --python "$WORK/venv" install --no-deps --no-build-isolation "${PP_REPO}/archive/${PP_COMMIT}.tar.gz"
"$PYTHON" -m pip --python "$WORK/venv" check
PP_CONFIGS="$(python3 -c 'from pilot_proxy.paths import CONFIGS_DIR; print(CONFIGS_DIR)')"
for RECEIVER in chord chime; do
    pilot-proxy export-runtime-weight-bundle \
        --receiver-profile "$PP_CONFIGS/receiver_profiles/${RECEIVER}_dtv_fengine.json" \
        --detector-core-profile "$PP_CONFIGS/detector_core/pilotproxy_cuda_local_reference_power_ratio.json" \
        --weight-coordinate-system post_spectral_sense_normalization \
        --physical-channel-range 14:36 --output-dir "$WORK/pilotproxy_bundle_$RECEIVER"
    pilot-proxy validate-runtime-weight-bundle --bundle-dir "$WORK/pilotproxy_bundle_$RECEIVER"
done
PIPELINE_DIR="$WORK/pipeline"
mkdir -p "$PIPELINE_DIR/fake_data/pilotproxy_bundle" "$PIPELINE_DIR/fake_data/pilotproxy_verify"
cp "$WORK/pilotproxy_bundle_chord/"* "$PIPELINE_DIR/fake_data/pilotproxy_bundle/"
python3 "$SRC/tests/verify_pilotproxy_pipeline.py" --calibrate --bundle-dir "$PIPELINE_DIR/fake_data/pilotproxy_bundle"
(cd "$PIPELINE_DIR" && timeout --signal=TERM --kill-after=10s 2m "$BUILD/kotekan/kotekan" -b 127.0.0.1:0 --config "$SRC/config/tests/verify_pilotproxy_pipeline.yaml")
python3 "$SRC/tests/verify_pilotproxy_pipeline.py" --dump-dir "$PIPELINE_DIR/fake_data/pilotproxy_verify" --bundle-dir "$PIPELINE_DIR/fake_data/pilotproxy_bundle"
export PILOTPROXY_TEST_BINARY="$BUILD/kotekan/kotekan"
PILOTPROXY_TEST_BUNDLE="$WORK/pilotproxy_bundle_chord" \
    timeout --signal=TERM --kill-after=10s 20m python3 -m pytest -v -x --basetemp "$WORK/pytest_chord" -o cache_dir="$WORK/pytest-cache" "$SRC/tests/test_pilotproxy_integration.py" "$SRC/tests/test_dtv_rfi_mask.py"
PILOTPROXY_TEST_BUNDLE="$WORK/pilotproxy_bundle_chime" \
    timeout --signal=TERM --kill-after=10s 20m python3 -m pytest -v -x --basetemp "$WORK/pytest_chime" -o cache_dir="$WORK/pytest-cache" "$SRC/tests/test_pilotproxy_integration.py" "$SRC/tests/test_dtv_rfi_mask.py" \
    -k 'receiver_geometry or receiver_namespace or archive_review or chime_detector or production_geometry_soak'
