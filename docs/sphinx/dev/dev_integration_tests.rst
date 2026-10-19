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
   decoding and models; ``check_<test>.py`` is one check per test. A check either has
   ``VARIANTS``, each run and checked on its own, or ``RUNS``, which are all run before one
   check compares them (T5).

``docs/integration-tests/``
   One Markdown description per test (``T<n>-<name>.md``): the errors the test catches, and the
   mechanism and formulae it relies on.

Differences from ``config/fengine``:

* ``poison_buffers`` is left at its default (``false``). Only the buffers a test checks are
  copied to the host and written.
* Inputs come from test generators instead of ``hdf5FileReadSingleFile``.
* ``N2Accumulate`` reads the bad feed mask (``in_bad_feed_mask_buf``), as in
  ``config/chord/pathfinder.j2``. It needs one mask frame per correlation frame, while BB and FRB1
  need the mask lifetime to equal their phase lifetimes, so it gets its own one-frame mask stream
  with the same content (see Findings).
* The bad feed mask lifetime (``bf_mask_lifetime_frames``, default 24), ``do_fringestop``,
  ``rfi_excision_enabled``, the gains ``upchan_gains`` (a dict from U to the U fine-channel
  gains), ``bb_gain`` and ``frb1_input_scale``, and the test outputs are jinja variables;
  production values are the defaults.
* ``half_filled`` (default ``false``) marks the northern half of the pathfinder dishes (32 to
  63; north is +y) as ``Missing``.

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
  cancel between generator and beamformer. Missing dishes receive nothing. Each source can be
  restricted to a list of inputs.
* ``inventPLMask`` produces random packet loss at a given rate (deterministic, reproducible in
  Python) and loss boxes for edge cases.
* ``inventBFMask`` produces bad feed schedules: random bad feeds that change every few frames, and
  for CHORD and the pathfinder an array that is only partly filled, starting from the south;
  missing dishes are marked bad.
* All generators take ``first_fpga_seq_num``. Tests start at a large value,
  ``2**33 + 7 * num_times``: frame-aligned, as ``N2Accumulate`` requires, but aligned to no
  lifetime or integration boundary.
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
     - Point-source tone from three directions, no noise; for N2 a weaker tone in Gaussian
       noise (a pure tone is flagged as RFI), fringe stopping off.
     - N2, upchan, BB, FRB1, FRB2
     - Visibility phases match geometry (sign, conjugation). The tone is in the predicted fine
       channel with the predicted phase for each U. CHIME: U=16 and U=128 agree. Beams peak in
       the right place.
     - N2 2°; upchan 2° mean, 15° 99th percentile; BB/FRB1 peak index exact; FRB2 source beam
       within 0.5% of the peak (grating-lobe ties)
   * - T3 Quantization
     - Noise (σ = 2.8 LSB) plus a weak tone.
     - upchan (all U), BB, FRB2, FRB3 with offset and scale, FRB PL mask
     - int4 rms, zero, saturated and -8 fractions against a Gaussian noise model. FRB3
       dequantized vs FRB2; -8 exactly where the PL mask is invalid.
     - Fail if rms < 0.5 or > 3.5 LSB, > 5% saturated, or any -8; FRB3 ≤ half a step plus
       float16 rounding
   * - T4 Overflow
     - 7 + 7i on every input; an amplitude-7 tone from a BB beam. No noise, RFI excision off
       (a constant-power signal has SK 0).
     - correlator (``n2k_correlation``), upchan (all U), BB with phases and shifts, FRB1, FRB2,
       FRB3 with offset and scale
     - No NaN or Inf, no -8. The correlator equals Σ E_i conj(E_j); BB equals the integer
       beamformer; upchan saturates with the predicted sign and equals the rounded prediction;
       FRB1 is below its bound for int4 input.
     - correlator and BB exact; upchan 0.02 LSB around rounding thresholds; FRB1 1%
   * - T5 Gains
     - Noise (σ = 1.2 LSB) plus a weak tone; a reference run with the production gains and a
       scaled run: a distinct factor in [1, 2) per fine channel, ``bb_gain`` and
       ``frb1_input_scale`` doubled.
     - upchan (all U), BB, FRB1
     - Per-fine-channel ``upchan_gain``, ``bb_gain`` and ``frb1_input_scale`` act as configured:
       the unrounded noise σ, inferred from the int4 rms with the rounded-Gaussian model, scales
       with the gain; the FRB1 intensity by 4 times the upchannelized power ratio.
     - 3% (int4), 1% (FRB1)
   * - T6 Dish locations
     - Eight one-hot inputs, each with a tone in its own window; six pairs with a zenith tone
       each; T2's ``frb2`` input on a half-filled array. No noise and RFI excision off except
       for the half-filled array.
     - N2 (with dish attributes), BB, FRB1; for the half-filled array as T2
     - Only the one-hot autocorrelations are non-zero; BB phase per dish matches its position;
       FRB1 fringes match the pair's grid separation; the half-filled array passes T2 with
       predictions from the present dishes.
     - N2 exact; BB phase 5°; FRB1 peak bins exact, fringe 0.01
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

* **Changing beam pointings at runtime.** ``setBBBeams`` and ``setFRBBeams`` serve only GET
  ``/<stage>/beams``. Future test: re-point the beams every few frames and check that the peak
  moves exactly at the lifetime boundary.
* **Time-varying gains.** The upchannelizer gain is ``do_once``, and the CHORD configs have no
  per-dish gains. Future test: T5 with gains that change every few frames.

Two features that were missing when this plan was written are now on develop, and their tests
belong in T8 and T9:

* **Bad feed mask in BB and FRB1** (#1778): ``calcBBPhase`` and ``setFRB1Phase`` zero the phases
  and weights of bad feeds. T9 adds BB and FRB1 outputs, which must not change when bad feeds
  are poisoned.
* **BB packet-loss mask** (#1766): T8 adds the BB mask (the AND over dishes of the expanded
  mask) and BB data.

Findings
========

* **FRB1 was mirrored along M (grid x)** (T2). The FRB1 beam index along M peaked at
  ``-2θ_M mod 2M`` instead of ``2θ_M``, while N (grid y) was correct, and FRB2 inherited the
  mirror. The second FFT in ``julia/kernels/frb.jl`` had the wrong sign; fixed in #1779.
* **The upchannelizer and pathfinder BB outputs were under-loaded** (T3). The upchannelizer
  window is scaled by 1/U and all configs used ``upchan_gain`` 1, so for 2.8 LSB input noise the
  int4 rms fell to 0.34 LSB (89% zeros) at U = 64. The pathfinder ``bb_scale`` was 12 instead of
  10, giving a BB rms of 0.75 LSB. #1784 sets ``upchan_gain`` to 0.8 √U (2 LSB output noise) and
  replaces ``bb_scale`` by ``bb_gain = 1/√D``.
* **A grid of one beam had NaN positions**, found while fixing the loading: ``setBBBeams`` and
  ``setFRBBeams`` divided by ``num_x - 1``, so every CHIME BB beam (``bb_num_beams_x: 1``) was
  zero. Fixed in #1785.
* **The bad feed mask lifetime has conflicting requirements.** ``N2Accumulate`` needs one mask
  frame per correlation frame (8192 samples). Since #1778, BB and FRB1 need the mask lifetime to
  equal their phase lifetimes, a multiple of the FRB processing quantum (192 U samples, 12288
  for U = 64). No single stream satisfies both. The test configs use two mask streams with the
  same content; a production config that feeds the mask to ``N2Accumulate`` and to BB/FRB1 (as
  ``config/chord/pathfinder.j2`` would) needs the same.
* ``N2Accumulate`` requires correlation frames to start at multiples of the frame length. The
  tests assume that the receivers guarantee this.
* FRB3 encodes FRB2 within half a step plus float16 rounding (T3).
* **FRB1 at U = 1 overestimates a constant intensity by 0.64%** (T4). The kernel accumulates
  Tds = 192 float16 terms per output; full-scale input gives 2104 instead of the exact bound
  2090.7. This is within the 1% allowance of ``frb1_intensity_limit``. At U ≥ 16 (12 or fewer
  terms) the saturated input gives 2090.
* The correlator, the upchannelizer and BB saturate exactly as predicted at full scale (T4). The
  correlator output stays far from the int32 limit (802816 for 8192 samples).
* A constant-power test signal is flagged entirely by first-stage RFI excision (SK = 0), so
  the correlator sums nothing; T4 switches excision off. FRB3 chunks of zero variance store
  scale 0 and q = 0, without NaN (T4).
* ``upchan_gain`` (per fine channel), ``bb_gain`` and ``frb1_input_scale`` act as documented
  (T5). The measured ratios agree with the predictions within 0.2%. Int4 rounding biases a raw
  rms ratio by up to 3.7% at the 0.9 LSB of the reference run, so the check corrects for it.
* Every subsystem places the dishes as configured (T6): only the predicted N2 products are
  non-zero, the BB phase of every dish matches its position within 2.2°, FRB1 shows the fringes
  of the dish separations, and a half-filled array passes T2.
* **Fringe-stopped N2 visibilities are truncated** (T6). ``N2Accumulate`` adds the
  fringe-stopped complex float values into its ``int32`` accumulator
  (``lib/stages/N2Accumulate.cpp:989``), which truncates towards zero and loses up to one count
  per component and sub-integration. T6 sees autocorrelations one count low.
* ``setUpchanGain`` checks the number of gains (one per fine channel) only with ``assert``
  (seen while writing T5, not tested), which a Release build drops: extra gains are ignored,
  and too few throw ``std::out_of_range``.

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
