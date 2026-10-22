CHIME detector validation
=========================

The CHIME detector uses 128 samples per window and 128 windows per block
(16384 samples); CHORD uses 64. A CUDA build contains both. Set
``detector_window_samples: 128`` on the detector. A bundle must match the
configured window size.

Set ``receiver_channel_id_namespace: chime_freq_id`` on the detector. The bundle
must explicitly declare this namespace at its top level and in each profile,
with ``receiver_channel_id`` values matching the metadata ``coarse_freq`` IDs.
When present, ``chime_channel_id`` must agree with the generic ID. Legacy
CHORD-only bundles remain accepted with the default ``chord_freq_id`` setting.

``config/include/dtv_chime_upgrade.j2`` is an optional diagnostic include for
the packed ``xposed_voltage`` ring. It uses streams 3 and 4 and writes mask,
power, and fine-support products. It is not included by the production template
and does not route any visibility or count consumer through the detector.
Review the actual stream allocation and metadata before including it.

``DtvRfiMask`` accepts 8192- and 16384-sample blocks. Its RFImask shape is
``[num_times/1024, F, 128]``; both visibility and count consumers must use the
same combined mask. Mismatched time, frequency, or support identity stops the
pipeline. This change does not add failure recovery or pairwise masking.

Profiles explicitly marked ``archive_review_only`` are refused for active
fine evaluation, even when their arithmetic fields have status ``calibrated``.
A permanent reject-all binding bypasses detector evaluation as before.
These candidate thresholds do not establish qualification for telescope use.

Validation
----------

Set ``PILOTPROXY_TEST_BINARY`` and ``PILOTPROXY_TEST_BUNDLE`` to a matching
binary and bundle, and include the current PilotProxy source and this checkout's
``python`` directory on ``PYTHONPATH``. The integration fixture copies the bundle
and replaces its fine calibration with explicitly synthetic test thresholds;
it never changes the source bundle.

Run ``tests/test_pilotproxy_integration.py`` with
``-k 'receiver_geometry or receiver_namespace or archive_review'`` for either
geometry. These tests compare masks, frame powers, and fine support
with integer CPU references, including window reversal, blocks crossing input
frames, ring reuse, namespace refusal, and candidate refusal. The original
CHORD integration suite applies to 64-sample bundles.

For the full CHIME detector geometry, use a CHIME bundle and enable
``PILOTPROXY_SOAK_FRAMES`` with ``-k production_geometry_soak``. Start with
eight frames before the 2048-frame run. The two cases use 2048 inputs,
16 frequencies and 16384-sample blocks, covering all 23 pilots across two
frequency assignments. See :ref:`dev_pilotproxy_testing` for memory limits,
reference checks and the scope of these synthetic tests.

``tests/test_dtv_rfi_mask.py::test_chime_mask_visibility_and_counts_across_ring_wrap``
checks the 16384-sample adapter using synthetic decisions, packet loss, invalid
rank support, and permanent masks. It compares GPU visibilities and counts and
RFImask counts with the same accepted samples. This test is independent of the
detector window size.

``tests/test_dtv_rfi_mask.py::test_chime_detector_mask_visibility_and_counts_across_ring_wrap``
uses the actual K128 detector with synthetic calibration, supported finite-threshold
acceptance and rejection, invalid rank support, and permanent masks before the
adapter and both GPU correlators. It checks
visibility and count agreement across eight frames and ring reuse. This is a
small-array correctness test, not a full CHIME throughput or science qualification.

The frozen CHIME accumulation configuration uses 2560 ns per FPGA sequence
tick, 16384 samples per frame, and zero-origin pairs of adjacent even/odd frames.
``N2Accumulate`` requires ``seq % 16384 == 0`` and pairs by
``(seq // 16384) % 2``. The detector adapter above does not enforce shared pair
acceptance. Deployed overrides and the mapping of archived captures onto this
grid require separate verification.
