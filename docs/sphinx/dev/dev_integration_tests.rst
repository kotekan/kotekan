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
* Subsystems: N2 correlator, upchannelizer, baseband beamformer (BB), FRB beamformer and, on
  CHIME, the HFB upchannelizer (U = 128), HFB1 and HFB2.
* The tests exercise the stages that call the kernels and the interfaces between stages. Kernels
  are only spot-checked (phases, complex conjugation).
* Input enters as voltage, packet-loss (PL) mask and bad feed mask frames; there is no input
  networking. Output leaves as frames written to HDF5; there is no output networking.
  Intermediate quantities (e.g. upchannelized voltages) are written as well.
* Timing numbers are meaningless; the tests do not run in real time.

Test levels:

#. Qualitative (correctness). This is the only level implemented so far.
#. Quantitative (sensitivity): SNR, gain settings, PFB effects, fringe stopping. Planned; see
   Level 2 tests.
#. RFI handling: fast and slow RFI masks.
#. Astronomy: pulsar, FRB and HI-like inputs.

Environment
===========

* CHORD and the pathfinder run on cx67/cx68 (NVIDIA A40). CHIME runs on upgrade-gpu1 (NVIDIA
  L40S); while that node is unavailable, the CHIME tests run on cx67/cx68, assuming that the A40
  and the L40S compute the same results. These nodes are shared: check ``nvidia-smi`` before a run
  and select an idle GPU with ``CUDA_VISIBLE_DEVICES``.
* Use a Release build with the test stages:
  ``-DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DWITH_TESTS=ON``.
* About 1 TByte of disk is available for all tests combined. Outputs of a passing test are
  deleted.
* Each test should take about an hour or less.

Layout
======

``config/integration-tests/``
   Top-level configs ``pathfinder.j2``, ``chord.j2`` and ``chime.j2``. Jinja variables select
   the test signal, schedules and outputs. ``run.py`` renders the config with them into
   ``<out>/<tel>_<test>_<variant>.yaml``, next to the log, and passes that file to kotekan: a
   command-line argument is limited to 128 KiB, which T7's manual CHORD beams exceed. The
   directory is self-contained: ``include/`` holds copies of the ``config/fengine`` includes
   (for CHIME, the setup that follows ``config/chime_upgrade_frb.j2``, #1804), changed as
   described below and listed in each copy's header, plus the test input and output includes.

``lib/testing/``
   The test input generators ``inventVoltage``, ``inventPLMask`` and ``inventBFMask``, extended
   for these tests.

``python/integration-tests/``
   ``run.py`` runs kotekan and the check for one test; ``kit.py`` holds the shared HDF5 reading,
   decoding and models; ``check_<test>.py`` is one check per test. A check either has
   ``VARIANTS``, each run and checked on its own, or ``RUNS``, which are all run before one
   check compares them (T5). A variant may also POST to kotekan's REST server while it runs
   (``posts``, T9r). With ``MEMORY_INTERVAL``, ``run.py`` samples kotekan's resident and GPU
   memory while it runs (T10).

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
* ``half_filled`` (default ``false``) marks the northern half of the dishes as ``Missing``
  (north is +y): pathfinder dishes 32 to 63, CHORD dishes 256 to 511, and on CHIME the feeds with
  ``grid_y_idx`` 128 to 255 of every cylinder.
* ``bf_source: rest`` (default ``invent``) produces the bad feed masks with ``bufferBadInputs``
  from POSTs to ``/updatable_config/bad_inputs``, as ``config/chord/pathfinder.j2`` does.
* ``rfi_feed_averaged_min_good_frac`` (default 0.8, as ``config/fengine``) is a jinja variable.
* ``gps_time`` fixes the telescope time origin (jinja variable ``gps_frame0_nano``, default
  2026-01-01), so that fringe stopping is the same in every run.
* ``bb_beam_layout`` and ``frb2_beam_layout`` replace the keys of ``setBBBeams`` and
  ``setFRBBeams`` that place the beams (``fixed_mode`` / ``mode`` and its keys; default: the
  production grids), and ``frb2_weights_lifetime_in_samples`` the FRB2 weights lifetime
  (default: 240 FRB2 frames, about 60 s).
* ``write_decimation`` writes only the first X of every Y frames of an output
  (``hdf5FileWrite`` ``write_x_frames`` / ``per_y_frames``); ``write_frames`` then counts all
  frames, written or not.

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

* ``inventVoltage`` (``input_kind: sky``) produces point-source tones
  with per-dish geometric phases, noise, time windows, and one-hot or few-hot inputs. Tones have
  an integer number of cycles per frame, so frames repeat; the generator copies frames from a
  precomputed pool and changes only the time-limited parts. Dish positions are computed from the
  config's ``telescope.dish_inputs``, not from ``Telescope``, so that a positioning bug does not
  cancel between generator and beamformer. Missing dishes receive nothing. Each source can be
  restricted to a list of inputs.
* ``inventPLMask`` produces random packet loss at a given rate (deterministic, reproducible in
  Python) and loss boxes for edge cases, at the 2-sample resolution of the PL mask
  (``testPacketLoss`` in ``lib/testing/testSkySignal.hpp``; jinja variables ``loss_rate``,
  ``loss_seed`` and ``loss_boxes``).
* ``inventBFMask`` produces bad feed schedules: random bad feeds that change every few frames
  (``testBadFeeds``; jinja variables ``bf_change_frames``, ``bf_random_count`` and ``bf_seed``),
  and with ``bf_mode: auto`` marks the Missing dishes of a partly filled array bad.
* All generators take ``first_fpga_seq_num``. Tests start at a large value,
  ``2**33 + 7 * num_times``: frame-aligned, as ``N2Accumulate`` requires, but aligned to no
  lifetime or integration boundary. T9 starts at ``2**33 + 5 * num_times``, three frames before
  an N2 bin, so that the bins begin and end at mask changes.
* ``inventVoltage`` can overwrite the voltages of lost packets with 0+0i, as the receivers do,
  or poison them with the int4 no-data value ``-8-8i`` (jinja variable ``lost_voltage``), and
  replace the voltages of bad feeds and Missing dishes by random bytes (``bad_voltage``).
* For CHIME, ``inventVoltage`` writes one voltage buffer per frequency, with the inputs in ADC
  order (``input_reorder``, as RadioTelescopeFEngine.jl writes them), which the transpose
  undoes. It writes the lost samples as the receivers do: one byte per sample for each group of 4
  frequencies, with no per-input information. ``zeroSamples`` and ``lostSamplesToPLMask`` then
  zero the lost voltages and build the PL mask. ``bad_voltage: random_valid`` gives random bytes
  without -8 (finding below).

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
       channel with the predicted phase for each U. Beams peak in the right place.
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
       scaled run: a distinct factor in [1, 2) per fine channel, ``bb_gain`` doubled and
       ``frb1_input_scale`` halved (``setFRB1Phase`` refuses more than 1.5 times the production
       scale for CHORD).
     - upchan (all U), BB, FRB1
     - Per-fine-channel ``upchan_gain``, ``bb_gain`` and ``frb1_input_scale`` act as configured:
       the unrounded noise σ, inferred from the int4 rms with the rounded-Gaussian model, scales
       with the gain; the FRB1 intensity by 1/4 times the upchannelized power ratio.
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
     - T2's weak tone on a BB and an FRB2 beam at once, ``grid_degrees`` (BB a single column,
       ``num_x`` 1) and ``manual`` (irregular positions, random 64-bit ids); lifetimes of 3
       frames (BB, FRB1) and one FRB2 frame.
     - BB, FRB2, beam positions and ids
     - Positions and ids equal the config, one frame per lifetime at ``seq0 + k L`` (FRB2:
       ``seq0 + T* + k L``); the peak is in the beam configured at the source; every BB beam is
       non-zero; the tone in the source beam is the same in every window across the lifetime
       boundaries.
     - positions, ids, BB peak exact; FRB2 as T2; windows 6σ (σ ≈ 0.1%)
   * - T8 Packet loss
     - T2's weak tone in noise; random loss at 0.01% (lost voltages 0+0i) and at 1% (-8-8i),
       plus loss boxes at frame, FRB frame and ``T*`` edges and single sample pairs; a reference
       run without loss.
     - PL masks (input, expanded, upchan for all U, BB, FRB), upchan (all U), BB, FRB1, FRB2,
       FRB3, N2 counts and visibilities
     - Masks equal the propagated injected loss; valid data equal the reference run bit for
       bit; FRB3 is -8 exactly where invalid; N2 counts the lost samples, and V n_valid equals
       the reference minus the lost samples.
     - masks and valid data exact; FRB3 one step; N2 1e-6 (float32)
   * - T9 Bad feed mask
     - T2's weak tone in noise; 3 to 5 random bad inputs changing every 3 frames, lifetime 3
       frames; full and half-filled array; bad inputs and Missing dishes carry random bytes; a
       reference run without poison.
     - both mask streams, BB phases, FRB1 weights, BB, FRB1, FRB2, FRB3, RFI S012/S012tilde/
       SKtilde/mask, N2
     - Masks equal the schedule; A and W are zero exactly for bad inputs; outputs equal the
       reference except FRB samples whose window holds poison their weights do not mask; S0tilde
       counts the good inputs; N2 flags are the AND over the bin, good products unchanged.
     - exact; FRB3 one step
   * - T9r Bad feed REST
     - As T9 without poison; masks from two ``bufferBadInputs``, all good until a POST to
       ``/updatable_config/bad_inputs``.
     - both mask streams, BB phases, FRB1 weights, S012tilde, N2
     - Each stream changes once to the posted set, at the first common boundary at or after the
       posted FPGA start time; A, W, S0tilde and the N2 flags follow.
     - exact
   * - T10 Long run
     - T2's weak tone on an FRB2 beam in noise; 0.01% loss (0+0i), 3 to 5 poisoned bad inputs
       changing every 9 frames, lifetimes of 3 frames (BB, FRB1) and one FRB2 frame; 8640
       frames (362 s simulated, 10 minutes on cx67).
     - both mask streams, BB and FRB2 beam positions and ids, FRB PL mask, ``n2k_plcounts``, N2
       (16 frequencies), FRB2 (decimated)
     - Every writer gets all frames with contiguous ``fpga_seq_num``; every mask, beam and N2
       switch on time; PL mask and N2 counts equal the injected loss, N2 flags the schedule;
       loss rate as configured; no NaN or Inf; FRB2 peak in the source beam; resident memory
       flat after warm-up.
     - exact; loss rate 3σ; memory 32 MiB

The pathfinder is tested first, then CHORD, then CHIME. All Level 1 tests pass on all three. On
CHIME, BB is tested as on the other telescopes, the HFB upchannelizer (U = 128) in T1, T2 and T8,
and the HFB1 and HFB2 timing in T1; T9 and T10 use a bad feed mask lifetime of at least 2 T* (see
Findings).

Level 2 tests (planned)
=======================

Level 2 measures how much signal survives each stage. Each check measures a quantity with its
statistical error (an efficiency, a response curve, a noise level) and compares it with a model.
None of these tests exists yet.

A check passes when every measured value agrees with its model within its statistical error: 3σ
for a single value, more when many values are compared (5σ for 10^4 values), plus float16
rounding for float16 outputs. The checks report the measured efficiencies. There are no
requirement thresholds, since no digital loss budget is defined yet.

Test inputs
-----------

``testSkySignal`` produces tones and independent white noise per input. Level 2 needs four
extensions:

* **Correlated noise source.** The same noise in every dish, delayed by the geometric delay at
  every frequency inside the coarse channel, so that bandwidth smearing is real. It is a sum of
  tones with random amplitudes and phases at every frequency of its period, each with its own
  delay. Its strength is given as S/SEFD per dish: about 0.3 for
  Cyg A on CHORD, 0.06 for Cyg A per CHIME feed, 0.02 for a 100 Jy FRB.
* **F-engine channel shape.** Noise and sources are shaped by the response of the F-engine PFB,
  including the alias from the neighbouring channel near the channel edges, before they are
  rounded to int4. ``pfb_prototype`` (``lib/utils/pfbPrototype.hpp``) gives the prototype filter.
  With white noise per coarse channel, the upchannelizer sees a flat noise spectrum, which the
  F-engine does not deliver.
* **Drifting source.** The direction follows a fixed RA and Dec as the Earth rotates. ``run.py``
  computes the directions with astropy and passes them to the generator, so that an error in
  ``Telescope`` does not cancel. Within a frame the phase is interpolated linearly; on CHORD's
  longest east-west baseline at 1.5 GHz it changes by less than 1° per frame.
* **Noise that does not repeat.** ``inventVoltage`` copies frames from a pool of
  ``noise_period_frames`` precomputed frames (default 4), so its noise repeats. An N2 integration
  spans 24 frames and a drifting source is not periodic, so these tests compute every frame,
  which is slower.

One run of T12 uses RadioTelescopeFEngine.jl files instead, read with ``hdf5FileReadSingleFile``
(384 to 512 MiB per frame on disk). They come from a full ADC, PFB and int4 F-engine that is
independent of kotekan.

Tests
-----

.. list-table::
   :header-rows: 1
   :widths: 10 25 15 40 10

   * - Test
     - Input
     - Outputs
     - Check
     - Tolerance
   * - T11 Upchannelizer response
     - Weak tones at many offsets within a coarse channel, one offset per group of inputs
       (``active_inputs``), in noise with the F-engine channel shape; a variant with a few
       adjacent coarse channels (jinja variable); T1's windowed tone for the time response.
     - upchan (all U), on CHIME also the HFB upchannelizer (U = 128)
     - The complex response of every fine channel against the tone offset (passband ripple,
       scalloping, leakage into neighbouring fine channels, stopband, phase) equals the window and
       ``upchan_gain`` model (``kit.upchan_window``). Each fine channel peaks at the frequency of
       its ``freq_upchan_index``. Across adjacent coarse channels, the fine channels tile the band
       as modelled. The output edge of a windowed tone follows the cumulative window.
     - pass rule
   * - T12 Loading and quantization efficiency
     - Noise of 2.8 LSB with the F-engine channel shape and a weak correlated source (S/SEFD
       0.02); variants with input rms from 1.5 to 4 LSB, with a bright source (Cyg A), and from
       RadioTelescopeFEngine.jl files.
     - upchan (all U), BB, FRB1, FRB2, FRB3, HFB
     - Per output and fine channel: the rms against its target, its flatness across fine
       channels, the fraction at ±7. The source's SNR in each output, against its SNR before
       quantization, gives the efficiency of each requantization (upchan, BB including CHORD's
       block shifts, FRB3) and the bias of FRB1's float16 accumulation. The bright variant gives
       the source power at which each output compresses.
     - pass rule
   * - T13 Beam sensitivity and FRB2 beam shape
     - A weak correlated source on a BB beam, on an FRB2 beam, between beams and far from
       zenith; for the beam shape, weak tones at a grid of positions around one FRB2 (HFB2)
       beam, one source per fine channel.
     - BB, FRB1, FRB2; on CHIME HFB1 and HFB2
     - The source power over the noise power in BB, FRB1 and FRB2 beams is D·S/SEFD times the T12
       efficiencies. BB bandwidth smearing away from zenith follows
       ``⟨|(1/D) Σ_d exp(2πi δν τ_d)|²⟩`` over the coarse channel, with τ_d the delay of
       dish d between source and beam direction. FRB2 and HFB2 respond as a beam steered at
       their position (below), with the peak at the configured position at every fine
       frequency; the response midway between production FRB2 beams is reported. The FRB1, FRB2
       and HFB intensities of noise have the predicted mean, a variance as χ² with the right
       degrees of freedom, and the predicted correlation between neighbouring beams.
     - pass rule
   * - T14 N2 amplitude and noise
     - Correlated sources with known weak and strong correlations in noise that does not repeat;
       fringe stopping off; a variant with T8's packet loss.
     - N2 visibilities, weights and counts
     - V_ab / √(V_aa V_bb) equals the correlation, with a model of int4 quantization of
       correlated Gaussian voltages. The variance of V_ab between integrations is V_aa V_bb / N,
       with N the valid samples. The variance estimate (1 / weight) agrees with that scatter, for
       ``EvenOddPosDef`` (pathfinder, CHORD) and ``CHIMEv1`` (CHIME).
     - pass rule
   * - T15 Fringe stopping and drift
     - A drifting source near the phase centre and one away from it, in noise that does not
       repeat; several minutes of simulated time; a variant that switches fringe stopping by
       REST (``fringestop_updatable_config``).
     - N2; BB with a tracking beam (``tracking_beams``) on the source and fixed beams; FRB2
     - Each N2 bin equals the model visibilities of its sub-integrations, rotated by
       fringe-stopping phases that the check computes with astropy for the bin centre, and
       averaged; without fringe stopping, the unrotated average. The truncation of fringe-stopped
       visibilities (Findings) stays below the noise of a long average. The tracking BB beam
       keeps the source power; fixed BB and FRB2 beams see the source drift through, an
       east-west cut of their response. The REST switch takes effect at the first bin at or after
       the posted time.
     - pass rule

**The FRB2 beam model.** FRB2 interpolates FRB1 intensities. ``cudaCalcFRB2Weights`` sets
``W2 = U(p, M, θ_M) U(q, N, θ_N)``, the trigonometric interpolation of Kendrick's note with
trapezoid weights. Along M, the FRB1 intensity is a trigonometric polynomial with lags below M,
which its 2M beams determine. An FRB2 beam at n_b therefore responds to a source at n as a beam
steered at n_b: ``|AF(n − n_b, ν)|²``, with AF the array factor of the M × N dish grid with the
FRB1 weights. The model is exact up to float16 rounding and the width of a fine channel. An error in
the interpolation can leave every Level 1 peak check intact. The CHIME frequency ordering in the
second Nyquist zone (Findings) shifts a beam in proportion to its angle, so it should show far
from zenith. HFB2 uses the same weights and ``cudaFRBBeamReformer``.

Expected results:

* A CHORD BB beam centred on Cyg A has √(1 + 0.3 · 512) ≈ 12 times the noise rms that
  ``bb_gain`` = 1/√D sets (about 2.8 LSB), so it saturates. T12's bright variant measures this
  for every output.
* T11 and T12 test the upchannelizer window and per-fine-channel gains against F-engine input;
  they should be rerun when either changes.

Order: the input extensions first, then T12, T11, T13 and T14; T15 last. T15 simulates several
minutes; at T10's rates that takes 1.5 minutes per simulated minute on CHORD and 4 on CHIME, more
if the generator limits the rate.

Left for later levels: the SK false-flag rate on Gaussian noise, and whether a bright noise-like
source is flagged (Level 3); dispersed pulses, FRB3 dynamic range for bright bursts, and HI lines
through the HFB (Level 4).

Missing features (known gaps)
=============================

These features are missing in kotekan. Add the tests once they exist.

* **Changing beam pointings at runtime.** ``setBBBeams`` and ``setFRBBeams`` serve only GET
  ``/<stage>/beams``. Future test: re-point the beams every few frames and check that the peak
  moves exactly at the lifetime boundary.
* **Time-varying gains.** The upchannelizer gain is ``do_once``, and the CHORD configs have no
  per-dish gains. Future test: T5 with gains that change every few frames.

Two features that were missing when this plan was written are now on develop:

* **Bad feed mask in BB and FRB1** (#1778): ``calcBBPhase`` and ``setFRB1Phase`` zero the phases
  and weights of bad feeds. T9 checks that BB and FRB1 to FRB3 do not change when bad feeds are
  poisoned.
* **BB packet-loss mask** (#1766): T8 checks the BB mask (the AND over dishes of the expanded
  mask), and that valid BB samples equal those of a run without loss.

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
* ``setBBBeams`` and ``setFRBBeams`` place the beams as configured in ``grid_degrees`` and
  ``manual`` mode, including a single-beam axis at its midpoint and 64-bit ids, and the BB and
  FRB2 beams follow them (T7). With lifetimes of 3 frames (BB, FRB1) and one FRB2 frame, the
  tone in the source beam changes by at most 0.44% at a boundary, within the noise. With static
  pointings, a stale matrix from the previous lifetime would go unnoticed.
* The ``setFRBBeams`` documentation names its mode key ``fixed_mode`` and the type of ``beams``
  ``FixedBBBeam`` (``lib/stages/setFRBBeams.cpp:78-90``); the key is ``mode`` and the type
  ``FRBBeam``. ``num_beams`` is described as "fixed + tracking" (line 71), but there are no
  tracking FRB beams.
* Every PL mask is the exact propagation of the injected loss (T8), at the 2-sample resolution
  of the mask, including the ``T*`` skip and frame and FRB frame boundaries. Valid
  upchannelized, BB, FRB1 and FRB2 samples are bit-identical with and without loss, FRB3 is -8
  exactly where the FRB mask is invalid, and N2 counts the lost samples and normalizes by the
  valid ones to float32 precision (5e-8).
* **N2 and the RFI statistics need lost samples to be 0+0i** (T8). ``cudaCorrelator`` and the
  S1/S2 kernel read only the voltages (``external/n2k/include/n2k/rfi_kernels.hpp:142``); the
  PL mask enters only the counts and S0. ``TransposeBasebandArray`` writes 0+0i for lost
  blocks (``lib/stages/TransposeBasebandArray.cpp:370``). With -8-8i there instead, RFI
  excision flagged 1.3 M samples of an integration and the visibilities were off by up to 19%.
* **N2 flags included the masks of the frames before an integration** (T9). ``N2Accumulate``
  folded the mask of every frame into the flags, including the frames it skips while waiting for
  the first bin boundary, and after finishing a bin folded the mask of the frame that completed
  it into the next bin. Inputs bad only before a bin were therefore flagged in it: two and five
  extra inputs in T9. Fixed in #1790.
* Every other subsystem excludes bad inputs exactly (T9): the BB phases and FRB1 weights are zero
  exactly for the bad inputs of their lifetime frame, BB beams and all FRB1/FRB2 samples whose
  window holds no unmasked poison are bit-identical with and without poison (the last FRB sample
  of each weight frame straddles the mask change and is exposed), S012tilde sums only the good
  inputs, and good N2 products are unchanged.
* **Two ``bufferBadInputs`` stages switched at different samples** (T9r). Each applied a posted
  update to the frames it produced after the start time by the wall clock, and each runs ahead
  of its consumers by its own buffer depth. In three runs the one-frame stream for
  ``N2Accumulate`` changed 11 to 23 frames after the beamformer stream.
  #1793 applies updates by FPGA time, at a configurable alignment boundary
  (``bf_mask_update_alignment_in_samples``). With the same boundary for both stages, N2 and the
  beamformers now get the same mask at the same sample.
* **First-stage RFI excision flags everything when more than 20% of the inputs are masked**
  (T9). The SK validity threshold ``rfi_feed_averaged_min_good_frac`` (0.8 in ``config/fengine``)
  counts masked inputs as missing samples, so for the half-filled array SKtilde is invalid, the
  RFI mask is all zero and every N2 sample is flagged. ``config/chord/pathfinder.j2`` uses 0.1.
* **N2 fringe stopping depends on the start time** (T9). Without a ``gps_time`` block the
  telescope takes the system time at startup as its time origin
  (``lib/utils/CHORDTelescope.cpp:270``), and about 60 fringe-stopped visibilities and weights
  of an integration differed between two identical runs. The tests now fix the origin.
* **The hdf5FileWrite decimation wrote one frame too many** (T10). With ``write_x_frames`` X
  and ``per_y_frames`` Y it wrote the first X + 1 of every Y frames, not the first X as
  documented. Fixed in #1797.
* Over 8640 frames (362 s simulated) everything stays exact (T10): 2880 BB/FRB1 lifetimes, 1439
  FRB2 lifetimes, 959 bad feed changes and 359 N2 integrations switch on time, every writer gets
  every frame, the FRB PL mask and the N2 counts follow the loss, and resident memory grows by
  1.6 MiB after warm-up. ``hdf5FileWrite`` holds about one extra frame while it compresses it
  (475 MiB for FRB2), and writing an FRB2 frame stalls the pipeline for about 5 s.
* T1 to T10 pass on CHORD. Only array-size dependent parts of the checks changed: T2's N2 tone
  amplitude, T4's BB model, T5's FRB1 scale, T6's dish choice and FRB1 peak threshold, and T7's
  BB column (below).
* A coherent tone of a given SNR is flagged by first-stage RFI excision more readily on larger
  arrays: the feed-averaged SK threshold scales as 1/√N. With T2's N2 tone (A = 2 in 2.8 LSB
  noise), 29% of the pathfinder samples and 95% of the CHORD samples were flagged. T2 now scales A
  with N (1.47 for CHORD; 99.5% valid).
* For D > 128 the BB kernel shifts the partial sum of each block of 2^(σ+4) dishes by σ before
  adding the blocks (four blocks of 128 for CHORD, σ = 3; ``julia/kernels/bb.jl:997``). The
  output then differs from a single shift of the whole sum at rounding boundaries. T4 models the
  blocks and finds BB exact on CHORD; the largest int16 partial sum at full scale is 13435.
* On CHORD, ``setFRB1Phase`` allows at most 1.52 times the production ``frb1_input_scale``. Its
  bound weights all 24 × 24 FRB1 cells, including the 64 empty ones, so it is 27% above the real
  worst case of 22300, which T4 reaches (22272).
* On CHORD's 48 × 48 FRB1 beam grid the fringe value next to the peak is 0.9957, within the
  measurement error; T6 puts the peak threshold in the first gap between predicted levels.
* T7's single BB column had the pathfinder geometry. On CHORD (96 beams 0.078° apart), a column
  that starts at the source has a grating lobe of the source on another BB beam at every
  frequency, so no BB peak is unambiguous; the column is now centred on the source.
* T2's tone (0.75 LSB) saturates the BB source beam for D > 64: 0.75 √512 ≈ 17 LSB for CHORD.
  The BB peak checks tolerate this.
* On CHORD a written FRB2 frame stalls the pipeline for about 10 s. With the FRB2 weights
  recomputed every FRB frame and ``frb2_weights_ring_depth`` 1, T10 still runs at 16 frames/s;
  resident memory is 13.4 GB and grows by 0.6 MiB, GPU memory 20 GB.
* ``setUpchanGain`` checks the number of gains (one per fine channel) only with ``assert``
  (seen while writing T5, not tested), which a Release build drops: extra gains are ignored,
  and too few throw ``std::out_of_range``.

CHIME (``config/fengine``, run on an A40):

* **The production config** ``config/chime_upgrade_frb.j2`` **has bugs that the**
  ``config/fengine`` **CHIME setup (#1804) fixes.** It does not start with the current kotekan:
  ``processFRBFeedGains`` lacks ``frb1_swap_MN: true`` and the FRB1 weights buffer lists its dish
  axes as ``dishN, dishM``, and the SKtilde buffer names its last axis ``S`` instead of ``SK``.
  ``N2Accumulate`` does not read the bad feed mask, so bad inputs are not flagged in N2 (none of
  154 in T9); it needs the one-frame mask stream that the test configs use. The dpdk handlers of
  link group 0 write ``lost_samples_buffer_0`` for ``host_voltage_buffer_3`` to ``_6``, but
  ``zeroSamples`` (``id // 4``) and ``lostSamplesToPLMask`` apply it to buffers 0 to 3 (code
  reading). It keeps ``upchan_gain`` 1, which leaves the U = 16 output at 0.70 LSB rms for 2.8
  LSB input noise, uses the identity transpose, and gives default feed gains 1+1i, for which
  ``processFRBFeedGains`` warns that FRB1 can overflow float16.
* **The CHIME transpose stops on -8** (T9). ``julia/kernels/xpose2048.jl`` traps if any input
  nibble is -8 (no data), and kotekan dies with CUDA error 719. One input sending -8 stops a CHIME
  node. T9 and T10 poison CHIME inputs with random bytes without -8.
* **The FRB1 weights lag the bad feed mask by T*** (T9). ``upchan_output_alignment`` makes T* 3
  frames on CHIME, and the weights made from mask frame k apply from T* after the start of that
  frame, so after a mask change FRB1 uses the previous bad inputs for 3 frames (of 24 in
  production). This holds on every telescope; #1802 removes the lag when the lifetime divides
  T*. T9 uses a mask lifetime of at least 2 T* on CHIME.
* **The HFB gain saturates T2's tone.** With ``upchan_gain`` 0.8 √128 the 0.75 LSB tone reaches
  6.8 LSB at U = 128 and saturates the int4 output; T2 scales the HFB gains by 1/√2.
* The CHIME BB beams are a narrow column (1 × 32, ``grid_degrees``): sources off the column get
  at most 2.2% of the on-source power in any BB beam, so the BB peak checks skip sources whose
  strongest beam has less than 10% of it.
* ``CHIMETelescope`` ignores dish types: it writes grid indices for Missing dishes in the BB beam
  positions (T6), and ``inventBFMask`` mode ``auto`` cannot find Missing dishes (T9 lists them).
  It centres the array, so the written feed positions are offset by (-33.0, -38.862) m (T6).
* ``cudaCalcFRB2Weights`` assigns rising frequencies to the fine channels in every Nyquist zone
  (``lib/cuda/cudaCalcFRB2Weights.cpp:205``), while the fine channels count down in the second
  zone. The FRB2 beam positions are off by at most half a coarse channel, about 0.05%, too
  little for T2 to see.
* The N2 counts from ``lostSamplesToN2Counts`` are exact (T8).
* ``Config::exists`` does not look in parent blocks, while ``get`` and ``get_default`` do
  (``lib/core/Config.cpp:140``): a stage that tests a top-level key such as ``input_reorder``
  with ``exists`` does not find it.
* With BB and HFB, CHIME T10 runs at 5 to 6 frames/s, against 15 frames/s without them; the
  cause is not investigated.
* The CHIME tests assume kotekan's conventions (channel c at 800 MHz - c Δν, the geometric phase
  exp(-2πiν n·r/c)); these are not checked against ICEboard data.

Data volume and run time
========================

* Input frames are 384 MiB (pathfinder, CHORD) or 512 MiB (CHIME) per 41.9 ms; they are not
  written.
* A test writes at most about 10 GiB (8 frames of upchannelized voltages and BB beams, a few N2
  integrations).
* A run simulates 2 to 4 seconds and takes a few minutes; T1 to T9 take about 45 minutes per
  telescope.
* T10 simulates 8640 frames (362 s). Kotekan runs at about 15 frames/s on cx67 (0.64 times real
  time), so this takes 10 minutes, and the check 5 minutes more. It writes 4.2 GB: only the
  small outputs, 16 frequencies of N2, and FRB2 decimated to one frame in 359 (all of them would
  be 14 GB of N2 and 470 GB of FRB2).
* On CHORD, T1 to T9 take about 35 minutes. T10 takes about 11 minutes and writes about 33 GB,
  mostly N2 (16 of 48 frequencies, 5.2 MB per frequency and integration).
* On CHIME (A40), a run takes 16 to 60 s (the T6 ``half`` run about 2 minutes); T1 to T9 take
  about 25 minutes. T10 runs at 5 to 6 frames/s (1600 s) with 17 GB of resident memory and 35 GB
  of GPU memory.

Future ideas
============

* Run the tests with ``poison_buffers: true``.
* Run the tests on a Debug build, or a Release build with asserts enabled, with shorter runs.
* Add the tests listed under missing features.
* Combine the output of several X-engine nodes to test all frequencies.
* Test the FRB back-end.
