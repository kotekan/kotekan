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
    PP_REPO="$(python3 -c 'import json; print(json.load(open("external/pilotproxy/VENDOR.json"))["upstream_repo"])')"
    python3 -m pip install "git+${PP_REPO}@${PP_COMMIT}"
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


Fine support and channel threshold limits
========================================

The optional ``dtv_fine_support_name`` detector output is an ``int32[F, 2]``
product named ``dtv_fine_support``. Each row contains the rank-valid flag and
the number of usable positive-denominator reference bins. A valid fine test
has flag 1; insufficient rank support has flag 0. The pair ``[-1, -1]`` means
that no fine test was evaluated, including an unbound, coarse, or permanently
masked frequency. This support product does not establish packet-loss,
input-health, target-support, or scientific-calibration validity.

The F-engine include connects this product at full rate to ``DtvRfiMask``.
That stage intersects the existing RFI mask with the DTV decision and excludes
invalid fine support before the same combined mask reaches correlation and
valid-sample counts. It refuses mismatched timestamps, frequency identities,
layouts, and malformed support rows. Legacy graphs that omit the optional
support buffer retain their previous behavior.

Two explicit receiver-frequency lists configure the follow-up path:

* ``dtv_require_fine_freq_ids`` requires an evaluated fine test for each listed
  frequency. An unevaluated row stops processing rather than being treated as
  a successful test.
* ``dtv_permanent_mask_freq_ids`` encodes the reject-all limit of the threshold
  family. The detector bypasses its calibration and evaluation for these
  frequencies, and ``DtvRfiMask`` clears every sample regardless of the raw
  detector mask. A numeric Q16 multiplier of zero is not this limit: the
  comparison is strict, so zero scores would remain unflagged.

Both lists contain unique exact integer receiver IDs in the active node's
frequency assignment, not physical television-channel numbers. They must be
disjoint. The detector and mask stage receive the same permanent-mask list.
An incomplete assignment does not imply permanent masking. A reviewed policy
must state each assignment's reason, era, frequency coverage, and evidence.
This configuration implements threshold assignments; it does not establish
their commissioning or BAO qualification.

The integration tests include a real CUDA fine decision with invalid support
and a permanent-mask limit, then verify the resulting visibilities and sample
counts. Keep synthetic plumbing validation separate from telescope acceptance.
