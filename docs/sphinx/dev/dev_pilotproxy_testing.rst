.. _dev_pilotproxy_testing:

*********************
PilotProxy validation
*********************

What the stage does and how it is configured on the telescope is in
:ref:`pilotproxy`.

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


The exported development bundle is for testing and is not calibrated.

The integration tests compare every mask, coarse power and FPGA timestamp
against the CPU reference using synthetic calibration.

For a sustained workload, enable the larger geometries after checking the
test process's soft and hard locked-memory limits (``ulimit -Sl`` and
``ulimit -Hl``). Each geometry has four 384 MiB host voltage frames, so its
locked-memory allowance must exceed 1.5 GiB plus the other buffers. A 64 MiB
default fails before detector execution. Use the host's approved mechanism
to raise the dedicated test process's limit; record the effective user and
limits. Preserve the usual locked-buffer configuration.

.. code-block:: sh

    PILOTPROXY_SOAK_FRAMES=2048 PILOTPROXY_SOAK_TIMEOUT_SECONDS=1800 \
        python3 -m pytest -v -s \
        tests/test_pilotproxy_integration.py -k production_geometry_soak

The two cases use 64 dishes with 384 frequencies and 512 dishes with 48
frequencies, each with two polarizations. All 23 bound pilots are evaluated.
The full-rate fine-support product is compared with the independent CPU
reference at every block, along with masks, coarse powers, and FPGA timing;
unbound frequencies must retain the ``[-1, -1]`` support sentinel.

Each 2,048-frame case processes 768 GiB of voltage through a four-frame ring.
Only the four repeated seed voltage frames are saved, together with every
mask, power, and support output. Allow about 2.4 GiB of scratch space per case
and retain failed fixtures until reviewed. The default per-pipeline timeout
is 1,800 seconds. Start with ``PILOTPROXY_SOAK_FRAMES=8`` to verify local
allocation and geometry before a sustained run.

At the configured 5.12 microsecond sample interval, 2,048 blocks represent
85.89934592 seconds of data per case. Report measured wall time separately:
it includes synthetic seed generation, GPU work, and file output. Reused
synthetic inputs and file-I/O stress do not establish production cadence,
processing headroom, or telescope acceptance.

The mask stage's own tests need the same two variables:

.. code-block:: sh

    python3 -m pytest tests/test_dtv_rfi_mask.py -q


Fine support and threshold limits
=================================

Record the Kotekan revision, vendor manifest, runtime bundle, and configuration
when validating fine support and frequency policies. The support product and
the two frequency lists are described in :ref:`pilotproxy`.

The integration tests include a real CUDA fine decision with invalid support
and a permanent-mask limit, then verify the resulting visibilities and sample
counts. Keep synthetic plumbing validation separate from telescope acceptance.
