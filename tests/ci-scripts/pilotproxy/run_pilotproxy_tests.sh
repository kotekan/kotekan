#!/bin/bash
# PilotProxy DTV detector checks: vendored-copy integrity, the CHORD (64 samples per window) and
# CHIME (128) runtime bundles, the pipeline verifier and the integration tests.
# Usage: run_pilotproxy_tests.sh <build dir>
set -euo pipefail
BUILD="$1"
SRC="${GITHUB_WORKSPACE:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export PYTHONPATH="$SRC/python${PYTHONPATH:+:$PYTHONPATH}"
python3 "$SRC/tools/check_vendored_pilotproxy.py" --offline
PP_COMMIT="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["upstream_commit"])' "$SRC/external/pilotproxy/VENDOR.json")"
PP_REPO="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["upstream_repo"])' "$SRC/external/pilotproxy/VENDOR.json")"
# The bundle exporter and the Python reference come from the pilot-proxy package
# (upstream_repo in VENDOR.json); the detector core itself is vendored here.
python3 -m pip install --upgrade "git+${PP_REPO}@${PP_COMMIT}"
PP_CONFIGS="$(python3 -c 'from pilot_proxy.paths import CONFIGS_DIR; print(CONFIGS_DIR)')"
for RECEIVER in chord chime; do
    pilot-proxy export-runtime-weight-bundle \
        --receiver-profile "$PP_CONFIGS/receiver_profiles/${RECEIVER}_dtv_fengine.json" \
        --detector-core-profile "$PP_CONFIGS/detector_core/pilotproxy_cuda_local_reference_power_ratio.json" \
        --weight-coordinate-system post_spectral_sense_normalization \
        --physical-channel-range 14:36 --output-dir "$BUILD/pilotproxy_bundle_$RECEIVER"
    pilot-proxy validate-runtime-weight-bundle --bundle-dir "$BUILD/pilotproxy_bundle_$RECEIVER"
done
PIPELINE_DIR="${RUNNER_TEMP:-/tmp}/pilotproxy_pipeline_test"
mkdir -p "$PIPELINE_DIR/fake_data/pilotproxy_bundle" "$PIPELINE_DIR/fake_data/pilotproxy_verify"
cp "$BUILD/pilotproxy_bundle_chord/"* "$PIPELINE_DIR/fake_data/pilotproxy_bundle/"
python3 "$SRC/tests/verify_pilotproxy_pipeline.py" --calibrate --bundle-dir "$PIPELINE_DIR/fake_data/pilotproxy_bundle"
(cd "$PIPELINE_DIR" && "$BUILD/kotekan/kotekan" --config "$SRC/config/tests/verify_pilotproxy_pipeline.yaml")
python3 "$SRC/tests/verify_pilotproxy_pipeline.py" --dump-dir "$PIPELINE_DIR/fake_data/pilotproxy_verify" --bundle-dir "$PIPELINE_DIR/fake_data/pilotproxy_bundle"
export PILOTPROXY_TEST_BINARY="$BUILD/kotekan/kotekan"
PILOTPROXY_TEST_BUNDLE="$BUILD/pilotproxy_bundle_chord" \
    python3 -m pytest -v "$SRC/tests/test_pilotproxy_integration.py" "$SRC/tests/test_dtv_rfi_mask.py"
PILOTPROXY_TEST_BUNDLE="$BUILD/pilotproxy_bundle_chime" \
    python3 -m pytest -v "$SRC/tests/test_pilotproxy_integration.py" "$SRC/tests/test_dtv_rfi_mask.py" \
    -k 'receiver_geometry or receiver_namespace or archive_review or chime_detector'
