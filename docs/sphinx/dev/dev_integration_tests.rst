.. _dev_integration_tests:

*****************************
X-engine integration tests
*****************************

End-to-end tests of one X-engine node for the CHORD pathfinder, CHORD and CHIME. They check that
the stages work in a production-like setup, that they interact correctly, and that the output of
the node is correct. They are run by hand, not in CI.

Scope
=====

* One X-engine node in isolation, with simulated inputs. Outputs are checked per node, without
  combining frequencies and without the FRB back-end.
* Subsystems: N2 correlator, upchannelizer, baseband beamformer (BB), FRB beamformer, and for
  CHIME also HFB.
* The tests exercise the stages that call the kernels and the interfaces between stages. Kernels
  are only spot-checked (phases, complex conjugation).
* Input enters as voltage, packet-loss (PL) mask and bad feed mask frames; there is no input
  networking. Output leaves as frames written to HDF5; there is no output networking.
  Intermediate quantities (e.g. upchannelized voltages) are written as well.
* Timing numbers are meaningless; the tests do not run in real time.

Test levels:

#. Qualitative (correctness). This is the only level implemented so far.
#. Quantitative (sensitivity): SNR, gain settings, PFB effects, fringe stopping.
#. RFI handling: fast and slow RFI masks.
#. Astronomy: pulsar, FRB and HI-like inputs.

Environment
===========

* CHORD and the pathfinder run on cx67/cx68 (NVIDIA A40); CHIME runs on upgrade-gpu1 (NVIDIA
  L40S). These nodes are shared: check ``nvidia-smi`` before a run and select an idle GPU with
  ``CUDA_VISIBLE_DEVICES``.
* Use a Release build with the test stages:
  ``-DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DWITH_TESTS=ON``.
* About 1 TByte of disk is available for all tests combined. Outputs of a passing test are
  deleted.
* Each test should take about an hour or less.

Layout
======

``config/integration-tests/``
   Top-level configs ``pathfinder.j2``, ``chord.j2`` and ``chime.j2``. Jinja variables, passed
   with ``kotekan --variables '<json>'``, select the test signal, schedules and outputs. The
   directory is self-contained: ``include/`` holds copies of the ``config/fengine`` includes,
   changed as described below, plus the test input and output includes.

``lib/testing/``
   The test input generators ``inventVoltage``, ``inventVoltageCHIME``, ``inventPLMask`` and
   ``inventBFMask``, extended for these tests.

``python/integration-tests/``
   ``run.py`` runs kotekan and the check for one test; ``kit.py`` holds the shared HDF5 reading,
   decoding and models; ``check_<test>.py`` is one check per test.

Differences from ``config/fengine``:

* ``poison_buffers`` is left at its default (``false``). Only the buffers a test checks are
  copied to the host and written.
* Inputs come from test generators instead of ``hdf5FileReadSingleFile``.
* ``N2Accumulate`` reads the bad feed mask (``in_bad_feed_mask_buf``), as in
  ``config/chord/pathfinder.j2``. This requires one mask frame per correlation frame, so
  ``bf_mask_lifetime_in_samples`` is ``num_times``. A ``bufferBadInputs`` stage is available for
  the REST test.
* Lifetimes of slowly varying inputs are jinja variables; production values are the defaults.

Running
=======

From the repository root, with a Python that has ``numpy``, ``h5py``, ``hdf5plugin``, ``jinja2``
and ``pyyaml`` (e.g. ``/opt/kotekan_env``)::

    CUDA_VISIBLE_DEVICES=0 python3 python/integration-tests/run.py \
        --tel pathfinder --test timing \
        --kotekan build-release/kotekan/kotekan \
        --out /scratch/$USER/kotekan-integration-tests

``run.py`` writes into ``<out>/<tel>_<test>/`` and the kotekan log into
``<out>/<tel>_<test>.log``, prints ``PASS`` or ``FAIL`` with the failing quantity, and deletes the
output after a pass unless ``--keep`` is given. ``--check-only`` re-checks existing output.

Each writer stops after a given number of frames, and kotekan exits once all writers are done.
Every writer has a non-writing drain on the same buffer, so that a finished writer does not
stall the GPU pipeline.

Test inputs
===========

Inputs are generated on the fly; nothing is read from disk.

* ``inventVoltage`` (``input_kind: sky``) and ``inventVoltageCHIME`` produce point-source tones
  with per-dish geometric phases, noise, time windows, and one-hot or few-hot inputs. Tones have
  an integer number of cycles per frame, so frames repeat; the generator copies frames from a
  precomputed pool and changes only the time-limited parts. Dish positions are computed from the
  config's ``telescope.dish_inputs``, not from ``Telescope``, so that a positioning bug does not
  cancel between generator and beamformer.
* ``inventPLMask`` produces random packet loss at a given rate (deterministic, reproducible in
  Python) and loss boxes for edge cases.
* ``inventBFMask`` produces bad feed schedules: random bad feeds that change every few frames, and
  for CHORD and the pathfinder an array that is only partly filled, starting from the south;
  missing dishes are marked bad.
* All generators take ``first_fpga_seq_num``. Tests start at a large value that is not aligned to
  a frame or lifetime boundary (default ``2**33 + 12345``).
* Voltages of lost packets and bad feeds can be poisoned with the int4 no-data value ``-8-8i``.

Level 1 tests
=============

Every check also audits the metadata of every output: contiguous ``fpga_seq_num``, a common
upchannelizer output start ``T*``, ``coarse_freq`` / ``freq_upchan_factor`` /
``freq_upchan_index`` against the config, and no NaN or Inf in float outputs.

.. list-table::
   :header-rows: 1
   :widths: 10 25 20 30 15

   * - Test
     - Input
     - Outputs
     - Check
     - Tolerance
   * - T1 Timing
     - Time-limited tone, starting mid-frame. Run with seq 0 and with a large start.
     - upchan voltages (all U), FRB1, FRB2, FRB3, BB, N2, PL and RFI masks
     - Signal edges at the predicted ``fpga_seq_num`` in every output.
     - BB exact; upchan ±U/4; FRB ±1 sample; N2 by integration
   * - T2 Phases
     - Point-source tone from three directions, no noise.
     - N2, upchan, BB, FRB1, FRB2
     - Visibility phases match geometry (sign, conjugation). The tone is in the predicted fine
       channel with the predicted phase for each U. CHIME: U=16 and U=128 agree. Beams peak in
       the right place.
     - N2 2°; upchan 5°; peak index exact; beam pattern 10%
   * - T3 Quantization
     - Noise (σ = 2.8 LSB) plus a weak tone.
     - upchan, BB, FRB3 vs FRB2
     - int4 rms, zero and saturated fractions. FRB3 dequantized vs FRB2.
     - Fail if rms < 0.5 or > 3.5 LSB or > 5% saturated; FRB3 ≤ half a step
   * - T4 Overflow
     - Full-scale coherent input; full-scale tone from a beam direction.
     - all
     - No NaN or Inf; N2 equals the exact integer prediction; saturation without sign flips;
       FRB1 below its bound.
     - exact
   * - T5 Gains
     - Noise plus tone; reference and scaled runs.
     - upchan, BB, FRB1
     - Per-fine-channel ``upchan_gain``, ``bb_scale`` and ``frb1_input_scale`` act as configured.
     - 3% (int4), 1% (FRB1)
   * - T6 Dish locations
     - One-hot dishes; two-hot pairs; half-filled array.
     - N2, BB, FRB1, FRB2
     - The right N2 product is non-zero; BB phase per dish matches its position; FRB1 fringes
       match the dish separation; beams peak in the right place.
     - exact; BB phase 5°
   * - T7 Beam pointings
     - Point-source tone on different configured beams; short lifetimes.
     - BB, FRB2, beam positions and ids
     - The peak is in the beam configured at the source; positions and ids match the config;
       no step at lifetime boundaries.
     - exact
   * - T8 Packet loss
     - Noise plus tone; loss rates 0.01% and 1%, plus loss boxes at frame and ``T*`` edges; lost
       voltages poisoned; a reference run without loss.
     - PL masks (expanded, upchan, FRB), FRB3, N2 counts and visibilities
     - Masks equal the propagated injected loss; valid data equal the reference run.
     - masks and valid data exact; FRB3 one step; N2 1e-6
   * - T9 Bad feed mask
     - Random bad feeds changing every 3 frames; half-filled array; bad feeds poisoned; a
       reference run without poison.
     - RFI SK, N2
     - Only good feeds enter; flags switch at the exact frame; good data equal the reference.
     - exact
   * - T9r Bad feed REST
     - Static mask, then a POST to ``/updatable_config/bad_inputs``.
     - N2
     - The mask changes at one boundary after the POST.
     - exact
   * - T10 Long run
     - Noise plus tone, 0.01% loss, changing bad feeds, short lifetimes, 10 minutes.
     - N2, FRB PL mask, FRB2 (decimated)
     - No stalls, contiguous ``fpga_seq_num``, all switches on time, loss rate as configured.
     - exact; loss rate 3σ

The pathfinder is tested first, then CHORD, then CHIME.

Missing features (known gaps)
=============================

These features are missing in kotekan. Add the tests once they exist.

* **Bad feed mask in BB and FRB.** ``calcBBPhase``, ``setFRB1Phase`` and the FRB2 weights ignore
  the bad feed mask; only the RFI SK kernels and ``N2Accumulate`` use it. Future test: T9 with BB,
  FRB1 and FRB2 outputs, which must not change when bad feeds are poisoned.
* **Changing beam pointings at runtime.** ``setBBBeams`` and ``setFRBBeams`` serve only GET
  ``/<stage>/beams``. Future test: re-point the beams every few frames and check that the peak
  moves exactly at the lifetime boundary.
* **BB packet-loss mask.** Not on develop yet (draft PR #1766). Future test: T8 with the BB mask
  and BB data.
* **Time-varying gains.** The upchannelizer gain is ``do_once``, and the CHORD configs have no
  per-dish gains. Future test: T5 with gains that change every few frames.

Expected finding: the upchannelizer scales its PFB window by 1/U and all configs use
``upchan_gain`` 1, so for noise input the int4 output rms is about 0.89 σ/√U. T3 is likely to
flag this at large U.

Data volume and run time
========================

* Input frames are 384 MiB (pathfinder, CHORD) or 512 MiB (CHIME) per 41.9 ms; they are not
  written.
* A test writes at most about 10 GiB (8 frames of upchannelized voltages and BB beams, a few N2
  integrations).
* A run simulates 2 to 4 seconds and takes a few minutes; T1 to T9 take about 45 minutes per
  telescope, T10 about 12 minutes.

Future ideas
============

* Run the tests with ``poison_buffers: true``.
* Run the tests on a Debug build, or a Release build with asserts enabled, with shorter runs.
* Add the tests listed under missing features.
* Combine the output of several X-engine nodes to test all frequencies.
* Test the FRB back-end.
