.. _dev_pilotproxy_testing:

*********************
PilotProxy validation
*********************

What the stage does and how it is configured on the telescope is in
:ref:`pilotproxy`.

Build Kotekan with ``USE_CUDA=ON`` and ``WITH_TESTS=ON``. The CUDA device must
support the vendored detector core. The CI harness creates a temporary virtual
environment with pinned dependencies, exports both receiver bundles, and cleans
up its generated files on exit:

.. code-block:: sh

    bash tests/ci-scripts/pilotproxy/run_pilotproxy_tests.sh "$PWD/build"

For individual tests, use a separate Python environment with the dependencies
in ``tests/ci-scripts/pilotproxy/requirements.txt``.
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

The integration tests compare every mask, frame power and FPGA timestamp
against the CPU reference using synthetic calibration.

For a sustained workload, enable the larger geometries after checking the
test process's soft and hard locked-memory limits (``ulimit -Sl`` and
``ulimit -Hl``). The CHORD geometries have four 384 MiB host voltage frames;
CHIME has four 512 MiB frames. The locked-memory allowance must exceed 1.5 GiB
for CHORD or 2 GiB for CHIME, plus the other buffers. A 64 MiB
default fails before detector execution. Use the host's approved mechanism
to raise the dedicated test process's limit; record the effective user and
limits. Preserve the usual locked-buffer configuration.

.. code-block:: sh

    PILOTPROXY_SOAK_FRAMES=2048 PILOTPROXY_SOAK_TIMEOUT_SECONDS=1800 \
        python3 -m pytest -v -s \
        tests/test_pilotproxy_integration.py -k production_geometry_soak

The bundle selects the receiver; cases for the other receiver skip. Export a
``chime_dtv_fengine.json`` bundle to run CHIME. CHORD uses 64 dishes with 384
frequencies or 512 dishes with 48 frequencies, each with two polarizations and
8192 samples per block. Each CHORD case evaluates all 23 bound pilots.

CHIME uses 1024 dual-polarization feeds (2048 inputs), 16 frequencies and
16384 samples per block, matching ``config/chime_upgrade_n2.j2``. Two cases
cover pilots 14--28 and 29--36, with unbound channels filling each 16-frequency
frame. They use CHIME profiles, weights and channel IDs with the 128-sample
core. Synthetic packed voltage enters at the detector ring; these cases do
not exercise packet reception, the upstream transpose or the live correlator.

The full-rate fine-support product is compared with the independent CPU
reference at every block, along with masks, frame powers, and FPGA timing;
unbound frequencies must retain the ``[-1, -1]`` support sentinel.

Each 2,048-frame CHORD case processes 768 GiB of voltage through a four-frame
ring; each CHIME case processes 1024 GiB. Only the four repeated seed voltage
frames are saved, together with every mask, power, and support output. Allow
3 GiB of scratch space per case
and retain failed fixtures until reviewed. The default per-pipeline timeout
is 1,800 seconds. Start with ``PILOTPROXY_SOAK_FRAMES=8`` to verify local
allocation and geometry before a sustained run.

The sample intervals are 5.12 microseconds for CHORD and 2.56 microseconds
for CHIME. With their respective block sizes, 2,048 blocks represent
85.89934592 seconds of data per case. Report measured wall time separately:
it includes synthetic seed generation, GPU work, and file output. Reused
synthetic inputs and file-I/O stress do not establish production cadence,
processing headroom, or telescope acceptance.

The mask stage also has CPU-only Boost coverage, built with
``WITH_BOOST_TESTS=ON`` and run in the CPU jobs:

.. code-block:: sh

    build/tests/boost/test_DtvRfiMask

It checks both block sizes across buffer wrap, mask intersection, invalid fine
support, permanent masks, metadata preservation and malformed inputs. The
CUDA visibility and count integration tests use the two variables above:

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
