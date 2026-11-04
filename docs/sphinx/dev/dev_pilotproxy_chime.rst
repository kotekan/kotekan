CHIME detector validation
=========================

For CHIME, the detector uses 128 samples per window and 128 windows per block
(16384 samples), and for CHORD it uses 64 samples per window. A CUDA build
contains both, and ``detector_window_samples: 128`` on the detector selects the
CHIME geometry. Each bundle must match the configured window size.

On the detector, ``receiver_channel_id_namespace`` must also be set to
``chime_freq_id``. The bundle must declare this namespace at its top level and
in each profile, with ``receiver_channel_id`` values that match the metadata's
``coarse_freq`` IDs. When ``chime_channel_id`` is present, it
must agree with the generic ID. Legacy CHORD-only bundles are still accepted
with the default ``chord_freq_id`` setting.

``config/include/dtv_chime_upgrade.j2`` is an optional diagnostic template for
the packed ``xposed_voltage`` ring. It uses streams 3 and 4 and writes the
mask, power, and fine-support products. It is not part of the production
template, and no visibility or count consumer is routed through the detector.
Before the file is used, the stream allocation and metadata of the target
configuration must be reviewed.

``DtvRfiMask`` accepts 8192- and 16384-sample blocks, and its RFImask has the
shape ``[num_times/1024, F, 128]``. Both the visibility and count consumers
must use the same mask from this stage. A mismatch of time, frequency, or
support identity stops the pipeline. Failure recovery and pairwise masking are
not covered by this stage.

Profiles marked ``archive_review_only`` are refused for active fine
evaluation, even when their arithmetic fields have status ``calibrated``. A
permanent reject-all binding bypasses detector evaluation and is accepted for
these profiles. Such candidate thresholds are for archive review only, and
telescope use requires qualified thresholds.

Validation
----------

A binary and a matching bundle are read from ``PILOTPROXY_TEST_BINARY`` and
``PILOTPROXY_TEST_BUNDLE``, and the current PilotProxy source and the
``python`` directory of this checkout must be on ``PYTHONPATH``. In the
integration fixture, the bundle is copied and its fine calibration is replaced
with synthetic test thresholds, so the source bundle is never changed.

For either geometry, ``tests/test_pilotproxy_integration.py`` is run with
``-k 'receiver_geometry or receiver_namespace or archive_review'``. These tests
compare the masks, frame powers, and fine support with integer CPU references,
and they cover window reversal, blocks that cross input frames, ring reuse,
namespace refusal, and candidate refusal. The rest of the CHORD integration
suite applies to 64-sample bundles.

With a CHIME bundle and ``PILOTPROXY_SOAK_FRAMES`` set,
``-k production_geometry_soak`` runs the full CHIME detector geometry. A run of
eight frames should come before the 2048-frame run. Both cases use 2048
inputs, 16 frequencies, and 16384-sample blocks, and they cover all 23 pilots
in two frequency assignments. Memory limits, reference checks, and the scope of
these synthetic tests are described in :ref:`dev_pilotproxy_testing`.

``tests/test_dtv_rfi_mask.py::test_chime_mask_visibility_and_counts_across_ring_wrap``
checks the 16384-sample adapter with synthetic decisions, packet loss, invalid
rank support, and permanent masks. The GPU visibilities, the GPU counts, and
the RFImask counts are compared for the same accepted samples. This test does
not depend on the detector window size.

``tests/test_dtv_rfi_mask.py::test_chime_detector_mask_visibility_and_counts_across_ring_wrap``
runs the K128 detector ahead of the adapter and both GPU correlators, with
synthetic calibration, supported finite-threshold acceptance and rejection,
invalid rank support, and permanent masks. Over eight frames with ring reuse,
the visibilities and counts are checked for agreement. This test is a simple
correctness check on a small array, and it does not cover full CHIME
throughput or science qualification.

The frozen CHIME accumulation configuration uses 2560 ns per FPGA sequence
tick, 16384 samples per frame, and zero-origin pairs of adjacent even and odd
frames. ``N2Accumulate`` requires ``seq % 16384 == 0`` and pairs frames by
``(seq // 16384) % 2``. Shared pair acceptance is not enforced by the detector
adapter above. Deployed overrides, and the mapping of archived captures onto
this grid, are not covered by these tests and must be verified before use.
