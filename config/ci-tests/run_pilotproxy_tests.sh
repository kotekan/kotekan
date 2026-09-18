#!/bin/bash
# PilotProxy DTV detector checks: vendored-copy integrity, the runtime bundle, the pipeline
# verifier and the integration tests. Usage: run_pilotproxy_tests.sh <build dir>
set -euo pipefail
BUILD="$1"
SRC="${GITHUB_WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}"
python3 "$SRC/tools/check_vendored_pilotproxy.py" --offline
PP_COMMIT="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["upstream_commit"])' "$SRC/external/pilotproxy/VENDOR.json")"
python3 -m pip install --upgrade "git+https://github.com/WVURAIL/pilot-proxy@${PP_COMMIT}"
PP_CONFIGS="$(python3 -c 'from pilot_proxy.paths import CONFIGS_DIR; print(CONFIGS_DIR)')"
PP_BUNDLE="$BUILD/pilotproxy_bundle"
pilot-proxy export-runtime-weight-bundle \
    --receiver-profile "$PP_CONFIGS/receiver_profiles/chord_dtv_fengine.json" \
    --detector-core-profile "$PP_CONFIGS/detector_core/pilotproxy_cuda_local_reference_power_ratio.json" \
    --weight-coordinate-system post_spectral_sense_normalization \
    --physical-channel-range 14:36 --output-dir "$PP_BUNDLE"
pilot-proxy validate-runtime-weight-bundle --bundle-dir "$PP_BUNDLE"
PIPELINE_DIR="${RUNNER_TEMP:-/tmp}/pilotproxy_pipeline_test"
mkdir -p "$PIPELINE_DIR/fake_data/pilotproxy_bundle" "$PIPELINE_DIR/fake_data/pilotproxy_verify"
cp "$PP_BUNDLE/"* "$PIPELINE_DIR/fake_data/pilotproxy_bundle/"
(cd "$PIPELINE_DIR" && "$BUILD/kotekan/kotekan" --config "$SRC/config/tests/verify_pilotproxy_pipeline.yaml")
python3 "$SRC/tools/verify_pilotproxy_pipeline.py" --dump-dir "$PIPELINE_DIR/fake_data/pilotproxy_verify" --bundle-dir "$PIPELINE_DIR/fake_data/pilotproxy_bundle"
PILOTPROXY_TEST_BINARY="$BUILD/kotekan/kotekan" PILOTPROXY_TEST_BUNDLE="$PP_BUNDLE" \
    python3 -m pytest -v "$SRC/tests/test_pilotproxy_integration.py" "$SRC/tests/test_dtv_rfi_mask.py"
