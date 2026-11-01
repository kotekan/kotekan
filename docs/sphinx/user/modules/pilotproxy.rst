.. _pilotproxy:

********************************
DTV Pilot Detection (PilotProxy)
********************************

PilotProxy detects ATSC digital television (DTV) on a CHORD GPU node, next to
the n2k correlator, and can mask the affected coarse frequency. It is off by
default.

An 8-VSB DTV channel is 6 MHz wide and has a steady pilot tone 309.441 kHz
above its lower edge. For each ATSC channel from 14 to 36 (470 to 608 MHz) on a
node, the stage monitors the coarse frequency that holds the pilot. The
presence or absence of the pilot is decided once per 8192-sample block
(41.94304 ms, one n2k integration).

The tests are described in :ref:`dev_pilotproxy_testing`.


Components
==========

``cudaPilotProxyDetector`` (``lib/cuda``)
    A ``cudaCommand`` that reads the same voltage ring as the n2k correlator.
    For each bound frequency, one block from every input is packed into 128
    windows and passed to the detector core. ``detector_window_samples``
    selects 64 samples per window for CHORD or 128 for CHIME. For each local
    frequency and block, the command writes ``dtv_mask`` (``int8[1, F]``, 1
    rejects the block), ``dtv_powers`` (``uint64[1, F, 3]``, the target and the
    two reference frame powers, each the sum of the 256 fine powers of that
    term), and optionally ``dtv_fine_support`` (``int32[1, F, 2]``, described
    below). Frequencies without a pilot get zero masks and powers. The
    detector sums every input in the ring, even the inputs that the correlator
    flags as bad.
    The leading ``T`` axis holds one block, scaled by its FPGA sequence period,
    so ``hdf5FileWrite`` can concatenate blocks with ``frames_per_file``.

``external/pilotproxy``
    The CUDA detector core, copied unchanged from the PilotProxy project.
    ``VENDOR.json`` pins the upstream revision and the hash of each file, and
    the PilotProxy CI script checks them. Changes to the core are made
    upstream and then re-vendored. Core 4 uses the ``PilotProxy_*`` C API and
    the ``pilotproxy_detector.*`` sources, with the same integer arithmetic and
    runtime bundle format as the earlier core.

``DtvRfiMask`` (``lib/stages``)
    Applies the DTV decision to the SK mask ``rfi_RFImask``. Every sample of
    the block is cleared for a rejected frequency, and a kept frequency keeps
    its SK mask. The result, ``dtv_RFImask``, replaces ``rfi_RFImask`` for the
    n2k correlation, the n2k counts, and ``RfiMaskSum``.

For each of the 256 fine bins, the core divides the power in a target cell at
the pilot by the mean power in two reference cells beside it. A rank statistic
of this ratio is taken over the bundle's bulk bins, and the block is
rejected when the ratio in a bin at the pilot exceeds a calibrated multiple of
that statistic. All of the arithmetic is exact integer arithmetic, and the GPU
result matches the CPU reference bit for bit.

``dtv_fine_support`` records the validity of each test. A row of ``(1, n)`` is
a valid test with ``n`` bulk bins. A row of ``(0, n)`` has too few usable bulk
bins, and its mask of 0 does not mark the block as clean. A row of
``(-1, -1)`` means that no test was run, because the frequency has no pilot, is
permanently masked, or is on a node with the detector disabled. The support
product does not cover packet loss, input health, or calibration quality.


Runtime bundle
==============

``dtv_runtime_bundle_dir`` names a directory that holds ``pilot_profiles.json``,
``weights.bin``, ``weights.manifest.json``, ``detector_contract.json``, and
``sha256sums.txt``. These files are exported with the PilotProxy Python tools,
which are not part of Kotekan. Each profile gives the ATSC channel, the
receiver ``coarse_freq`` ID that it binds to (``chord_channel_id``), its
weights, and its fine calibration. The status of the fine calibration is
``pending_campaign`` (not calibrated) or ``calibrated``.

At startup, the bundle's digests are checked as in the PilotProxy validator.
``sha256sums.txt`` must list the other four files with their SHA-256, and
``weights_sha256`` in ``pilot_profiles.json`` and ``weights.manifest.json``
must match ``weights.bin``. This check requires Kotekan to be built with
OpenSSL.

The stage also checks the bundle's structure at startup, but the quality of
the calibration is not checked. Each node that runs the stage loads the bundle
and creates its detector handles at startup, even when the node has no pilot.
Channels are bound from the ``coarse_freq`` metadata of the first voltage frame
and stay fixed for the run, so a new bundle requires a restart.


Configuration
=============

``config/fengine/chord.j2`` loads ``include/dtv_chord.j2`` when either switch
below is true. Both switches default to false.

``dtv_enabled: true`` (record only)
    The detector runs, and its mask and powers are written to HDF5. n2k still
    reads the SK mask, so the visibilities do not change.

``dtv_apply_mask: true``
    ``DtvRfiMask`` builds ``dtv_RFImask`` and n2k reads it, so a rejected block
    is removed from the visibilities and the valid-sample counts together. A
    frequency with support ``(0, n)`` is also cleared for that block.
    ``dtv_RFImask`` is formed on the host, copied back to the GPU, and written
    to HDF5.

The other keys are listed below.

``dtv_runtime_bundle_dir``
    The bundle directory (default ``data/pilotproxy_bundle``).

``dtv_permanent_mask_freq_ids``
    Receiver IDs that the detector skips. With ``dtv_apply_mask``, every sample
    of these frequencies is cleared. In record-only mode, these frequencies are
    not evaluated and nothing is cleared.

``dtv_require_fine_freq_ids``
    Receiver IDs that must have a fine test in every block. A row of
    ``(-1, -1)`` for one of these IDs stops the run instead of counting as a
    pass. The list is enforced by ``DtvRfiMask`` and so applies only with
    ``dtv_apply_mask``.

Both lists hold receiver ``coarse_freq`` IDs. The ATSC channel number of a
pilot is given only in the bundle. The two lists must not overlap, and every
listed ID must be on the node, so the lists are set for each node.


Pathfinder recording
--------------------

``config/chord/pathfinder.j2`` uses the same ``dtv_enabled``, ``dtv_apply_mask``
and ``dtv_runtime_bundle_dir`` switches. It records masks, powers and fine
support in 1024-block files (about 43 seconds), under ``dtv_output_dir``
(default ``/scratch/dtv``).

While running, POST ``{"enabled": false}`` to ``/dtv_write_enabled`` to stop
recording. POST ``{"enabled": false, "valid_at_time_ns": 0}`` to
``/dtv_detector_enabled`` to stop detection at the next block. Set ``enabled``
to true to resume; recording starts a new file. A nonzero ``valid_at_time_ns``
applies the detector change at the first block starting at or after that
instrument time. Both switches start enabled.


GPU streams
===========

Every ``cudaProcess`` on a GPU shares that GPU's CUDA streams. The
F-engine and n2k stages use the default streams, with copies to the GPU on
stream 0, copies to the host on stream 1, and kernels on stream 2.
``dtv_chord.j2`` sets ``num_cuda_streams: 5`` for the detector stage. The
detector kernels are run on stream 3, and the detector products are copied to
the host on stream 4. Neither the detector nor the correlator queues behind
the other. These streams allow for the two stages to overlap, but the stages
still share the GPU's cores and memory bandwidth at the same priority.

The two stages are ordered by the voltage ring. A block is read by the
detector only after the copy that wrote it has finished on the GPU, and the
block is overwritten only after every reader has released it. In record-only
mode, n2k does not use the detector output, but a slow detector can hold the
voltage ring and back-pressure its producer, which delays n2k. With
``dtv_apply_mask``, n2k reads each block's ``dtv_RFImask``, so it waits for
that block's detector run, the ``DtvRfiMask`` step on the host, and the copy
back to the GPU (on stream 0, like the other copies to the GPU).


Pilot frequency only
====================

The stage masks only the coarse frequency that holds each pilot. A DTV channel
spans about 30 CHORD coarse frequencies on several GPU nodes, and the pilot
decision is not passed to the other coarse frequencies. A whole DTV channel is
removed by setting ``dtv_apply_mask`` and listing its receiver IDs in each
node's ``dtv_permanent_mask_freq_ids``.


Fatal errors
============

The following errors stop the whole Kotekan instance, and the correlator with
it:

* at startup, a bundle that fails its digest checks, a malformed bundle, or a
  malformed configuration
* at the first frame, missing or miscounted ``coarse_freq`` metadata, a listed
  receiver ID that is not on the node, or a bound channel that is not
  calibrated
* during the run, a ``coarse_freq`` change after binding, or a CUDA or
  detector error
* in ``DtvRfiMask``, mismatched timestamps, frequencies, or layouts, a malformed
  support row, or an unevaluated row for a required frequency

An uncalibrated channel must therefore be removed from the bundle or listed in
``dtv_permanent_mask_freq_ids`` before the run starts. A node with no bundle
pilot runs with the detector disabled and writes all-zero masks. These zero
masks mean that the blocks were not tested.


Monitoring
==========

* At the first frame, the log has a ``bound ATSC physical channel`` line for
  each pilot, a ``reject-all limit`` line for each permanently masked ID, or a
  ``detector DISABLED`` line. These lines should be checked against the node's
  frequency map.
* ``output_chord.j2`` writes ``dtv_mask`` and ``dtv_powers``, and with
  ``dtv_apply_mask`` it also writes ``dtv_RFImask``. The ``RfiMaskSum`` counts
  then cover the DTV rejections as well.
* ``dtv_powers`` holds the pilot and reference powers of every block, so a
  moved pilot or a new line near a reference bin can be seen there even when
  the mask does not fire. Either one requires a recalibration.
