.. _dev_pilotproxy_testing:

*********************
PilotProxy validation
*********************


Build Kotekan with ``USE_CUDA=ON`` and ``WITH_TESTS=ON``, and install its Python test
dependencies. The CUDA device must support the vendored detector core.
The runtime bundle is a separate input. From the repository root, install the
matching PilotProxy package, export the CHORD bundle and run the tests:

.. code-block:: sh

    PP_COMMIT="$(python3 -c 'import json; print(json.load(open("external/pilotproxy/VENDOR.json"))["upstream_commit"])')"
    python3 -m pip install "git+https://github.com/WVURAIL/pilot-proxy@${PP_COMMIT}"
    PP_CONFIGS="$(python3 -c 'from pilot_proxy.paths import CONFIGS_DIR; print(CONFIGS_DIR)')"
    export PILOTPROXY_TEST_BINARY="$PWD/build/kotekan/kotekan"
    export PILOTPROXY_TEST_BUNDLE="$PWD/build/pilotproxy_bundle"
    pilot-proxy export-runtime-weight-bundle \
        --receiver-profile "$PP_CONFIGS/receiver_profiles/chord_dtv_fengine.json" \
        --detector-core-profile "$PP_CONFIGS/detector_core/pilotproxy_cuda_local_reference_power_ratio.json" \
        --weight-coordinate-system post_spectral_sense_normalization \
        --physical-channel-range 14:36 --output-dir "$PILOTPROXY_TEST_BUNDLE"
    pilot-proxy validate-runtime-weight-bundle --bundle-dir "$PILOTPROXY_TEST_BUNDLE"
    python3 -m pytest -v tests/test_pilotproxy_pipeline.py \
        tests/test_pilotproxy_integration.py tests/test_dtv_rfi_mask.py


For telescope runs, set ``dtv_runtime_bundle_dir`` to a CHORD bundle calibrated
for the active inputs. The exported development bundle is for testing.

The integration tests compare every mask, coarse power and FPGA timestamp
against the CPU reference using synthetic calibration.

For a sustained workload, enable the larger geometries:

.. code-block:: sh

    PILOTPROXY_SOAK_FRAMES=2048 python3 -m pytest -v -s \
        tests/test_pilotproxy_integration.py -k production_geometry_soak

The mask stage's own tests need the same two variables:

.. code-block:: sh

    python3 -m pytest tests/test_dtv_rfi_mask.py -q
