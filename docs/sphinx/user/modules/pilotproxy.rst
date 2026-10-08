.. _pilotproxy:

********************************
DTV Pilot Detection (PilotProxy)
********************************

PilotProxy detects ATSC digital television (DTV) on a CHORD GPU node, next to
the n2k correlator, and can mask the affected coarse frequency. It is off by
default.

An 8-VSB DTV channel is 6 MHz wide and carries a steady pilot tone
309.441 kHz above its lower edge. For each ATSC channel from 14 to 36 (470 to
608 MHz) on a node, the stage watches the coarse frequency that holds the
pilot and decides once per 8192-sample block (41.94304 ms, one n2k
integration) whether the pilot is there.

The tests are described in :ref:`dev_pilotproxy_testing`.


Components
==========

``cudaPilotProxyDetector`` (``lib/cuda``)
    A ``cudaCommand`` that reads the same voltage ring as the n2k correlator.
    For each bound frequency it packs one block from every input into 128
    windows of 64 samples and runs the detector core. Per local frequency and
    block it writes ``dtv_mask`` (``int8[F]``, 1 rejects the block),
    ``dtv_powers`` (``uint64[F, 3]``, the target and the two reference coarse
    powers) and, optionally, ``dtv_fine_support`` (``int32[F, 2]``, see below).
    Frequencies without a pilot get zero masks and powers. The detector sums
    every input in the ring, including inputs the correlator flags as bad.

``external/pilotproxy``
    The CUDA detector core, copied unchanged from the PilotProxy project.
    ``VENDOR.json`` pins the upstream revision and the hash of each file, and
    ``tools/check_vendored_pilotproxy.py`` checks them. Changes to the core are
    made upstream and re-vendored, not edited here.

``DtvRfiMask`` (``lib/stages``)
    Combines the DTV decision with the SK mask ``rfi_RFImask``. A rejected
    frequency has every sample of the block cleared; a kept one keeps its SK
    mask. The result, ``dtv_RFImask``, replaces ``rfi_RFImask`` for n2k
    correlation, n2k counts and ``RfiMaskSum``.

For each of 256 fine bins the core forms the ratio of the power in a target
cell at the pilot to the power in two reference cells beside it. The block is
rejected when a bin at the pilot exceeds a calibrated multiple of a rank
statistic of that ratio over the bundle's bulk bins. The arithmetic is exact
integer arithmetic, and the GPU result matches the CPU reference bit for bit.
``dtv_fine_support`` says whether each test was valid: ``(1, n)`` is a valid
test using ``n`` bulk bins; ``(0, n)`` means too few usable bulk bins, with mask
0, which is not a pass; ``(-1, -1)`` means no test was run (no pilot,
permanently masked, or detector disabled). It does not show packet loss, input
health or calibration quality.


Runtime bundle
==============

``dtv_runtime_bundle_dir`` names a directory holding ``pilot_profiles.json``
and ``weights.bin``, exported with the PilotProxy Python tools, which are not
part of Kotekan. Each profile gives the ATSC channel, the receiver
``coarse_freq`` ID it binds to (``chord_channel_id``), its weights, and its
fine calibration, whose status is ``pending_campaign`` (not calibrated) or
``calibrated``.

The stage checks the bundle's structure at startup, but not the quality of the
calibration. Each node running the stage loads the bundle and creates its
detector handles at startup, even a node with no pilot. Channels are bound from
the first voltage frame's ``coarse_freq`` metadata and stay fixed for the run;
a new bundle needs a restart.


Configuration
=============

``config/fengine/chord.j2`` includes ``include/dtv_chord.j2`` when either
switch is true. Both default to false.

``dtv_enabled: true`` (record only)
    The detector runs and its mask and powers are written to HDF5. n2k still
    reads the SK mask, so the visibilities do not change.

``dtv_apply_mask: true``
    ``DtvRfiMask`` builds ``dtv_RFImask`` and n2k reads it, so a rejected block
    leaves the visibilities and the valid-sample counts together. A frequency
    with support ``(0, n)`` is also cleared for that block. The combined mask is
    formed on the host and copied back to the GPU, and is written to HDF5.

Other keys:

``dtv_runtime_bundle_dir``
    The bundle directory (default ``data/pilotproxy_bundle``).

``dtv_permanent_mask_freq_ids``
    Receiver IDs that the detector skips. With ``dtv_apply_mask`` every sample
    of these frequencies is cleared; in record-only mode they are just not
    evaluated.

``dtv_require_fine_freq_ids``
    Receiver IDs that must have a fine test in every block; a row of
    ``(-1, -1)`` stops the run instead of counting as a pass. ``DtvRfiMask``
    enforces this, so it applies only with ``dtv_apply_mask``.

Both lists hold receiver IDs (``coarse_freq``), not ATSC channel numbers. They
must not overlap, and every listed ID must be on the node, so they are set per
node.


GPU streams
===========

Every ``cudaProcess`` on a GPU shares that GPU's CUDA streams. The F-engine and
n2k stages use the defaults: copies to the GPU on stream 0, copies to the host
on stream 1 and kernels on stream 2. ``dtv_chord.j2`` gives the detector stage
``num_cuda_streams: 5``, runs its kernels on stream 3 and copies its products to
the host on stream 4, so the detector never queues behind the correlator, and
the correlator never queues behind it. The streams let the two overlap; they
still share the GPU's cores and memory bandwidth, at the same priority.

The voltage ring orders the two stages: the detector reads a block only after
the copy that wrote it has finished on the GPU, and the block is overwritten
only after the detector's last copy of it has finished. In record-only mode n2k
never waits for the detector. With ``dtv_apply_mask`` n2k reads each block's
``dtv_RFImask``, so it waits for that block's detector run, the host combine
and the copy back (on stream 0, like the other copies to the GPU).


Pilot frequency only
====================

The stage masks only the coarse frequency that holds each pilot. A DTV channel
spans about 30 CHORD coarse frequencies, spread across GPU nodes, and nothing
carries a pilot decision to the others. To remove a whole DTV channel, set
``dtv_apply_mask`` and list its receiver IDs in each node's
``dtv_permanent_mask_freq_ids``.


What stops the run
==================

These stop the whole Kotekan instance, the correlator included:

* a malformed bundle or configuration, at startup;
* at the first frame: missing or miscounted ``coarse_freq`` metadata, a listed
  receiver ID that is not on the node, or a bound channel that is not
  calibrated;
* a ``coarse_freq`` change after binding, or a CUDA or detector error;
* in ``DtvRfiMask``: mismatched timestamps, frequencies or layouts, a malformed
  support row, or an unevaluated row for a required frequency.

So an uncalibrated channel has to be left out of the bundle, or listed in
``dtv_permanent_mask_freq_ids``, before the run starts. A node with no bundle
pilot runs with the detector disabled and writes all-zero masks, which means
"not tested", not clean data.


Monitoring
==========

* At the first frame the log has a ``bound ATSC physical channel`` line for
  each pilot, a ``reject-all limit`` line for each permanently masked ID, or a
  ``detector DISABLED`` line. Check them against the node's frequency map.
* ``output_chord.j2`` writes ``dtv_mask`` and ``dtv_powers``, and with
  ``dtv_apply_mask`` also ``dtv_RFImask``. The ``RfiMaskSum`` counts then
  include the DTV rejections.
* ``dtv_powers`` follows the pilot and reference powers in every block, so a
  moved pilot or a new line near a reference bin can show up there even when
  the mask does not fire. Either is a reason to recalibrate.
