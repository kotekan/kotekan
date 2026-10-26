.. _dev_pilotproxy_testing:

*********************
PilotProxy validation
*********************

This page describes the tests of the PilotProxy detector and the DTV mask. The
stage and its configuration on the telescope are described in
:ref:`pilotproxy`.

The tests need Kotekan built with ``USE_CUDA=ON`` and ``WITH_TESTS=ON``, and a
CUDA device that supports the vendored detector core. The CI harness creates a
temporary virtual environment with pinned dependencies, exports both receiver
bundles, and removes its generated files on exit:

.. code-block:: sh

    bash tests/ci-scripts/pilotproxy/run_pilotproxy_tests.sh "$PWD/build"

Individual tests need their own Python environment with the dependencies in
``tests/ci-scripts/pilotproxy/requirements.txt``. A runtime bundle is also
needed, and it is exported with the PilotProxy tools. From the repository root,
the matching PilotProxy package is installed, the CHORD bundle is exported, and
the tests are run:

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


An exported development bundle is for testing only and is not calibrated.

In the integration tests, every mask, frame power, and FPGA timestamp is
compared with the CPU reference, using a synthetic calibration.

For a sustained workload, the larger geometries are enabled after the soft and
hard locked-memory limits of the test process are checked (``ulimit -Sl`` and
``ulimit -Hl``). Each CHORD geometry has four host voltage frames of 384 MiB,
and CHIME has four frames of 512 MiB. Locked memory must therefore exceed
1.5 GiB for CHORD or 2 GiB for CHIME, plus the other buffers. With a 64 MiB
default, the run fails before the detector executes. The limit is raised only
for the dedicated test process, with the host's approved mechanism, and the
usual locked-buffer configuration stays as it is. The effective user and limits
must be recorded.

.. code-block:: sh

    PILOTPROXY_SOAK_FRAMES=2048 PILOTPROXY_SOAK_TIMEOUT_SECONDS=1800 \
        python3 -m pytest -v -s \
        tests/test_pilotproxy_integration.py -k production_geometry_soak

The bundle selects the receiver, and cases for the other receiver are skipped.
The CHIME cases need a bundle exported with the ``chime_dtv_fengine.json``
receiver profile.
CHORD uses 64 dishes with 384 frequencies or 512 dishes with 48 frequencies,
each with two polarizations and 8192 samples per block. Each CHORD case
evaluates all 23 bound pilots.

CHIME uses 1024 dual-polarization feeds (2048 inputs), 16 frequencies, and
16384 samples per block, as in ``config/chime_upgrade_n2.j2``. Two cases cover
pilots 14--28 and 29--36, and unbound channels fill the rest of each
16-frequency frame. These cases use the CHIME profiles, weights, and channel
IDs with the 128-sample core. Synthetic packed voltage is written at the
detector ring, so packet reception, the upstream transpose, and the live
correlator are not covered.

At every block, the full-rate fine-support product is compared with the
independent CPU reference, along with the masks, frame powers, and FPGA
timing. Unbound frequencies must keep the ``[-1, -1]`` support sentinel.

Each 2048-frame CHORD case passes 768 GiB of voltage through a four-frame
ring, and each CHIME case passes 1024 GiB. Only the four repeated seed voltage
frames are saved, together with every mask, power, and support output. Each
case needs 3 GiB of scratch space, and failed fixtures should be kept until
they are reviewed. By default, each pipeline has a timeout of 1800 seconds. A
first run with ``PILOTPROXY_SOAK_FRAMES=8`` checks the local allocation and
geometry before a sustained run.

The sample interval is 5.12 microseconds for CHORD and 2.56 microseconds for
CHIME. With their block sizes, 2048 blocks hold 85.89934592 seconds of data in
each case. Measured wall time should be reported apart from this, since it
covers synthetic seed generation, GPU work, and file output. These cases reuse
synthetic inputs and stress the file I/O, and they do not cover production
cadence, processing headroom, or telescope acceptance.

For the mask stage, CPU-only Boost coverage is built with
``WITH_BOOST_TESTS=ON`` and run in the CPU jobs:

.. code-block:: sh

    build/tests/boost/test_DtvRfiMask

It covers both block sizes across buffer wrap, the mask intersection, invalid
fine support, permanent masks, metadata preservation, and malformed inputs. For
the CUDA integration tests of visibilities and counts, the two environment
variables above are used:

.. code-block:: sh

    python3 -m pytest tests/test_dtv_rfi_mask.py -q


Fine support and threshold limits
=================================

When fine support and the two frequency lists are validated, the Kotekan
revision, vendor manifest, runtime bundle, and configuration must be recorded.
The support product and the frequency lists are described in
:ref:`pilotproxy`.

In the integration tests, a CUDA fine decision is run with invalid support and
a permanent-mask limit, and the resulting visibilities and sample counts are
checked. These tests validate the data path with synthetic inputs and do not
cover telescope acceptance.
