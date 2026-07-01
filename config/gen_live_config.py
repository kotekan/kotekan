#!/usr/bin/env python3
"""Generate a LEAN DISTRIBUTED live airspy GNSS pipeline -- the valved CHORD-mirror
chain + GPS-only viewer -- for a chosen signal. Pick the signal via env GNSS_SIGNAL:

  GNSS_SIGNAL=L1CA   python3 config/gen_live_config.py   # -> config/live_l1.yaml   (default)
  GNSS_SIGNAL=L2C_CM python3 config/gen_live_config.py   # -> config/live_l2c.yaml
  GNSS_SIGNAL=L1C_P  python3 config/gen_live_config.py   # -> config/live_l1c.yaml (BOC(1,1) pilot)
  GNSS_SIGNAL=L5_Q   python3 config/gen_live_config.py   # -> config/live_l5.yaml

Ultimately CHORD runs every band/signal in parallel; on the single airspy prototype
each is its own band (one run at a time, retune the front end). This generator is the
build-out: one chain, parameterized by signal, with the per-channel plan derived from
the signal's bandwidth -- so adding L5 etc. is a table row, not a rewrite.

  airspyInput -> fftwEngine (PFB, N=12 ch ~208 kHz) -> Valve -> GnssSubbandSplit
    search arm:  ch_c -> bufferSend -> bufferRecv -> GnssChannelGather -> GnssChannelizedSearch
    track  arm:  ch_c -> GnssChannelizedTracker (per channel) -> GnssCoherentCombiner
                 -> rawFileWrite (recorded |A|)
    viewer:      SpawnProcess -> livebeam_server.py --no-power-stream (GPS Sky panel :8080)

The **valve** after the F-engine (chan_buf -> chan_buf2) drops a whole frame cleanly
(fpga_seq metadata preserved -> a KNOWN gap, not silent USB corruption) if a stage falls
behind, so the F-engine -- hence the airspy USB callback -- can NEVER block. (The old
un-valved live_full.yaml backpressured the airspy -> sample drops -> lock thrashed.)

5 MSPS / 2.5 MHz front end, carrier at f_offset = Fs/4 = 1.25 MHz = bin N/2. The covering
channels (those the signal's main lobe illuminates) are computed per signal from its chip
rate: L1 C/A (1.023 Mcps, ~2 MHz) fills all 12; the narrower L2C CM (511.5 kcps) lights a
subset. Only covering channels get a tracker (no point on dead channels).

*** L2C IS HEAVIER + LESS PROVEN THAN L1: *** the CM code is 20 ms (10230 chips @ 511.5
kcps) vs L1's 1 ms, so each despread is ~20x the work and the 20 ms coherent integration
needs a FINE Doppler grid (doppler_step <= 50 Hz; a coarse grid washes the peak out). CM
also carries only HALF the L2C power (CL is the dataless other half) -> ~3 dB weaker than
L1. Blind channelized L2C acquisition over the full +-6 kHz grid is expensive and UNVALIDATED
in realtime (the offline airspy_gps_l2c_offline.yaml validated L2C via GpsReplicaCorrelator,
not this channelized search) -- so lean on the broker's almanac Doppler assist (run_live.sh
passes --carrier-hz from the config's freq; L2 geometric Doppler ~ 0.779 x L1) to narrow the
search. Not every PRN broadcasts L2C (Block IIR-M/IIF/III do; older IIR don't).
"""
import os

# --- signal table: add a row to support a new band -----------------------------
# chip_hz sets the main-lobe half-bandwidth (covering channels); 20 ms L2C CM needs a
# finer doppler_step + more acquire_windows (weaker, longer coherent window).
# period = primary-code period; cap_coh = cap the search/tracker COHERENT window at 1 code
# period. N = PFB channels (Fs/(2N) wide each).
#
# *** WINDOW <-> CODE-PERIOD INVARIANT (the L2C no-detect bug, fixed 2026-06-28): *** the
# search integrates the |D|^2 surface INCOHERENTLY over contiguous windows of hops_per_record
# hops. That only stacks (the peak is stationary) if each window spans an INTEGER number of
# code periods; otherwise a window's start walks the code phase by (hpr*fft_len mod code_samples)
# per window and the peak smears across bins -> the incoherent sum gives ~1/nwin instead of the
# intended sqrt(nwin), i.e. detection DIES (test_gnss_channelized_acquire:
# l2c_incoherent_window_must_span_integer_code_periods). One code period is an integer number of
# hops iff fft_len=2N divides code_samples=round(period*Fs). L1 @ N=12: 5000 samples, any window
# is fine (cap_coh False -> default 3-period window, 3 ms << the 20 ms nav bit). L2C: 100000
# samples, so 2N MUST divide 100000 -- N=12 (fft_len 24) does NOT (one period = 4166.67 hops),
# which is exactly why channelized L2C saw nothing. N=10 (fft_len 20) divides it: 1 period =
# 5000 hops, stationary AND nav-safe (the CNAV symbol IS one 20 ms period, so a 3-period window
# would also span data-bit flips: ~6 dB + smear). Hence N is per-signal; the assert below is the
# guard against a future row silently reintroducing the smear.
# sbw = airspy device sample_bw (MHz): 2.5 -> 5 MSPS real / 1.25 MHz IF (L1/L2); 10 -> 20 MSPS
# real / 5 MHz IF (L5). Fs = 2*sbw*1e6, f_offset = Fs/4 are DERIVED from it below. L5 is 10.23
# Mcps (~20 MHz main lobe) so it needs the wide front end; at 20 MSPS we capture the central
# ~10 MHz (carrier +-5 MHz) -- about half the main lobe, ~3 dB and coarser code-phase resolution,
# but easily enough to despread. We target the L5 Q5 PILOT (dataless -> coherent integration for
# seconds with no nav bit, ideal for beam mapping / peeling; the 20-chip NH overlay is constant
# within the 1 ms window so it doesn't bite acquisition). L5 is in the DME/TACAN band (RFI here,
# clean at CHORD). cap_coh keeps the window = 1 code period (1 ms) so it stays within one NH chip.
SIGNALS = {
    "L1CA":   dict(name="GPS_L1CA",   freq=1575.42, chip=1.023e6, dstep=100.0, snr=10.0, N=12, sbw=2.5,
                   win=20, period=1e-3,  cap_coh=False, out="live_l1.yaml",  base="/tmp/gpslive"),
    "L2C_CM": dict(name="GPS_L2C_CM", freq=1227.6,  chip=511.5e3, dstep=50.0,  snr=8.0,  N=10, sbw=2.5,
                   win=40, period=20e-3, cap_coh=True,  out="live_l2c.yaml",  base="/tmp/gpsl2c"),
    # L1C-P (modernized civil pilot, Block III): BOC(1,1) -> the replica's channels() applies the
    # sine sub-carrier (committed 1d3c6c51), splitting the band into +-1.023 MHz lobes with a NULL
    # at the carrier; at the 2.5 MHz front end all 10 channels are "covering" (the central bin sits
    # in the null -> ~no energy, harmless). Code period 10 ms (10230 chips @ 1.023 Mcps), so N=10
    # is FORCED by the integer-code-period invariant (fft_len 2N=20 divides CODE_SAMPLES=50000 ->
    # 2500 hops/period; N=12 does NOT). cap_coh=True is REQUIRED beyond nav-safety: L1C-P carries
    # the 1800-symbol L1CO overlay (each symbol = one 10 ms primary period) and that generator is
    # not built yet (descriptor secondary_length=0), so a multi-period coherent window would
    # integrate across UNKNOWN overlay flips and cancel; capping the window to one 10 ms period
    # stays inside a single L1CO symbol. dstep=50 Hz (10 ms coherent -> ~1/2T cells, like L2C). The
    # combiner secondary_overlay wipe (deep |A|) is the follow-on once the L1CO generator lands. L1
    # is out of CHORD's band, so this is an airspy-prototype + BOC pathfinder, not a mapping handle.
    "L1C_P":  dict(name="GPS_L1C_P",  freq=1575.42, chip=1.023e6, dstep=50.0,  snr=8.0,  N=10, sbw=2.5,
                   win=20, period=10e-3, cap_coh=True,  boc_m=1, out="live_l1c.yaml",  base="/tmp/gpsl1c"),
    # win=100: L5 records are only 1 ms, so match the monolithic's 100 ms incoherent depth (it was
    # 20 ms = 5x shallower, which is why the channelized cleared only the strongest sat vs mono's six).
    "L5_Q":   dict(name="GPS_L5_Q",   freq=1176.45, chip=10.23e6, dstep=250.0, snr=10.0, N=10, sbw=10.0,
                   win=100, period=1e-3, cap_coh=True,  out="live_l5.yaml",   base="/tmp/gpsl5",
                   overlay="L5_NH20"), # dataless Q5 pilot: combiner wipes the known NH20 -> deep |A|
}
SIG = os.environ.get("GNSS_SIGNAL", "L1CA")
if SIG not in SIGNALS:
    raise SystemExit("GNSS_SIGNAL must be one of %s (got %r)" % (list(SIGNALS), SIG))
S = SIGNALS[SIG]

# --- combiner integration (env-tunable) ----------------------------------------
# INTEGRATION_MODE=block (default) accumulates INTEGRATION_LENGTH records, emits, resets.
# INTEGRATION_MODE=rolling runs an exponential moving average with time constant
# INTEGRATION_LENGTH records (no nav-bit cap on the incoherent |A|), emitting every
# OUTPUT_EVERY records -- so a weak sat's |A| (slot3) / nav-wiped |A| (slot8) climb out
# continuously over a long window. For minutes of rolling incoherent integration set a big
# INTEGRATION_LENGTH (records are one code period: 1 ms on L1, 20 ms on L2C), e.g.
#   INTEGRATION_MODE=rolling INTEGRATION_LENGTH=3000 GNSS_SIGNAL=L2C_CM python3 ... # ~60 s
INTEG_MODE = os.environ.get("INTEGRATION_MODE", "block")
if INTEG_MODE not in ("block", "rolling"):
    raise SystemExit("INTEGRATION_MODE must be block or rolling (got %r)" % INTEG_MODE)
INTEG_LEN = int(os.environ.get("INTEGRATION_LENGTH", "50"))
# FLL_LOCK_AMP: freeze the tracker's carrier FLL when the despread |A| drops below this (a signal
# dropout -- radar sweep, the broker coasting the seed), so f_track doesn't wander on noise. 0 =
# off (the FLL wander over a ~30 s coast is tolerable anyway); set ~half the locked |A| to enable.
FLL_LOCK_AMP = float(os.environ.get("FLL_LOCK_AMP", "0"))
# FLL_GAIN: the tracker carrier-loop gain (default 0.03). Set 0 for pure FEED-FORWARD (open-loop on
# the broker's predicted Doppler) -- the decisive test for a MARGINAL signal whose closed FLL chases
# its own ~1-sigma noise and randomises the per-record carrier phase (the L1C-P deep-wipe wall: the
# per-record phase decohered record-to-record even though C/A coheres for seconds on the same clock).
# fll_gain=0 leaves only the smooth clock-drift residual, which the offline derotation can remove.
FLL_GAIN = float(os.environ.get("FLL_GAIN", "0.03"))
# HOLD_LOCK_AMP / HOLD_RECORDS: held-reference deep-integration mode (0 = off). Once a tracker
# despreads its channel above HOLD_LOCK_AMP for HOLD_RECORDS straight records it FREEZES its
# absolute-anchored code+carrier reference (no broker re-seed, no per-record max-power pull-in) so
# the per-record carrier phase stays continuous -- the fix for the marginal-BOC deep-wipe wall (the
# noisy search + BOC-sidelobe pull-in were jumping cp ~150 chips/record and randomising the phase;
# see the 2026-06-30 L1C findings). Threshold is the PER-CHANNEL |A| (each tracker owns one
# channel): L1C-P strong channels ran ~0.2-0.5, edge channels ~0.03, so ~0.1 engages the strong set.
HOLD_LOCK_AMP = float(os.environ.get("HOLD_LOCK_AMP", "0"))
HOLD_RECORDS = int(os.environ.get("HOLD_RECORDS", "5"))

# OUT_NAME / BASE_DIR override the output config name + record dir, so a capture variant (e.g.
# INTEGRATION_LENGTH=1 -> record every raw per-record A for offline integration analysis) can be
# generated WITHOUT clobbering the live config: e.g. the L2C deep-integration reference capture is
#   GNSS_SIGNAL=L2C_CM INTEGRATION_LENGTH=1 OUT_NAME=live_intgn_l2c.yaml BASE_DIR=/tmp/gpsintgn_l2c \
#     python3 config/gen_live_config.py   (then run it, then python3 .../gps_intgn_check.py the dir).
OUT_NAME = os.environ.get("OUT_NAME", S["out"])
BASE_DIR = os.environ.get("BASE_DIR", S["base"])
# OVERLAY overrides the combiner secondary_overlay (deep pilot wipe). Defaults to the signal row's
# overlay (L5 always wipes its NH20), "" for signals without one. Set it to turn a basic config
# into a deep-wipe variant WITHOUT baking the wipe into the basic config -- e.g. the L1C-P deep
# wipe (its 1800-symbol L1CO overlay needs a LONG rolling window to make the phase search well-posed):
#   GNSS_SIGNAL=L1C_P INTEGRATION_MODE=rolling INTEGRATION_LENGTH=3600 OVERLAY=L1CO \
#     OUT_NAME=live_l1c_wipe.yaml BASE_DIR=/tmp/gpsl1cwipe python3 config/gen_live_config.py
OVERLAY = os.environ.get("OVERLAY", S.get("overlay", "") or "")

# --- front-end / channel plan --------------------------------------------------
# SBW overrides the airspy front-end bandwidth (MHz): 2.5 -> 5 MSPS / 1.25 MHz IF, 10 -> 20 MSPS /
# 5 MHz IF. Default = the signal row's sbw. There is no reason to lock a BOC signal (L1C, later
# Galileo E1/E6) to 2.5 MHz: its power sits in TWO +-boc_m*1.023 MHz lobes, and at 2.5 MHz the
# upper lobe (~2.27 MHz) is clipped at the band edge. SBW=10 captures BOTH full lobes -> ~+2-3 dB
# and a sharper BOC correlation peak. FS / F_OFFSET / BIN_W / the covering plan all derive from it.
SBW = float(os.environ.get("SBW", S["sbw"]))  # airspy device sample_bw (MHz)
N = S["N"]                      # PFB channels: Fs/(2N) wide each (L1 N=12 ~208 kHz; L2C/L5 N=10)
FS = 2.0 * SBW * 1e6            # real rate = 2x the airspy device sample_bw (2.5->5e6, 10->20e6)
F_OFFSET = FS / 4.0             # = Fs/4; carrier lands on bin N/2 (even N -> bin centre)
# Emit Fs/f_offset in "<x>e6" form with a guaranteed decimal point: YAML 1.1 needs the dot to
# read scientific notation as a FLOAT (bare "5e+06" parses as a STRING -> config get<double> fails).
FS_STR = "%se6" % (FS / 1e6)
FOFF_STR = "%se6" % (F_OFFSET / 1e6)
DOPPLER_MARGIN = 250000.0
BIN_W = FS / (2 * N)
CODE_SAMPLES = round(S["period"] * FS)
if S["cap_coh"] and CODE_SAMPLES % (2 * N) != 0:
    raise SystemExit(
        "fft_len=2N=%d must divide code_samples=%d for an integer-code-period coherent window "
        "(else the incoherent search smears); pick N so 2N | %d (e.g. N=10 -> 5000 hops/period)"
        % (2 * N, CODE_SAMPLES, CODE_SAMPLES))
# Covering channels: r2c bin j centred at j*BIN_W over [0, Fs/2); keep those whose passband
# overlaps the signal's occupied band carrier +- _half. For BPSK _half = chip_rate (mirrors
# gnss::covering_channels). For BOC the power is in lobes out at +-boc_m*1.023 MHz, so the
# occupied band runs to ~boc_m*1.023 MHz + chip_rate -- WITHOUT this the wide (10 MHz) plan would
# keep only the central (null) bins and MISS the BOC lobes. At 2.5 MHz it is a no-op (all bins
# already covered). boc_m defaults to 0 (BPSK).
_half = S.get("boc_m", 0) * 1.023e6 + S["chip"] + DOPPLER_MARGIN
_lo, _hi = F_OFFSET - _half, F_OFFSET + _half
COV = [j for j in range(N)
       if (j * BIN_W - 0.5 * BIN_W) < _hi and (j * BIN_W + 0.5 * BIN_W) > _lo]
NCOV = len(COV)
# Coherent-window cap: hops_per_record = EXACTLY 1 code period (CODE_SAMPLES // fft_len, an
# integer because the assert above guarantees fft_len | CODE_SAMPLES). One period is both
# nav-safe (never integrates coherently across an L2C CNAV symbol boundary) AND a legal
# incoherent window (integer code periods -> stationary peak; see the invariant note above).
# Omitted for L1 (the default 3-period ~3 ms window is well inside the 20 ms nav bit).
HOPS = (", hops_per_record: %d" % (CODE_SAMPLES // (2 * N))) if S["cap_coh"] else ""
PRNS = "[%s]" % ", ".join(str(p) for p in [28, 32, 31, 25, 3, 10, 26, 1, 12, 13, 2])
NPRN = 11
PORT0 = 15001

L = []
L.append("""---
# AUTO-GENERATED by gen_live_config.py (GNSS_SIGNAL=%s) -- lean DISTRIBUTED live airspy
# pipeline: valved CHORD-mirror (per-channel track + bufferSend/Recv node mimic + central
# combine) + GPS-only viewer. See the generator header + docs/gnss_pipeline_reference.md.
# Run (run_live.sh CFG=this):
#   LAT=43.968697 LON=-79.252106 ALT=260 CFG=config/%s ./config/run_live.sh
type: config
log_level: info
instrument_name: airspy_gps
telescope: { name: ICETelescope, num_polarizations: 1, num_dishes: 1 }

# Let the browser GPS panel (served on :8080) reach kotekan's REST API cross-origin.
rest_server:
    cors_allow_origins: ["http://localhost:8080", "http://127.0.0.1:8080"]

samples_per_data_set: 49920
bytes_per_sample: 2
buffer_depth: 16
sizeof_float: 4
cpu_affinity: [0, 1, 2, 3, 4, 5, 6, 7]

spectrum_length: %d
num_taps: 4
sample_rate: %s
f_offset: %s
signal: %s
code_doppler_sign: 1.0
n_prn: %d
record_floats: 11
doppler_margin_hz: 250000.0
acquire_snr: %g
acquire_windows: %d
doppler_min: -6000.0
doppler_max: 6000.0
doppler_step: %g
# run_live.sh auto-patches prns:/n_prn: to what's overhead now (gps_visible_prns.py).
prns: %s

gnss_pool: { kotekan_metadata_pool: GnssChanMetadata, num_metadata_objects: 32 * buffer_depth }

input_buf: { kotekan_buffer: standard, metadata_pool: none,      num_frames: buffer_depth, frame_size: samples_per_data_set * bytes_per_sample }
chan_buf:  { kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: buffer_depth, frame_size: samples_per_data_set * sizeof_float }
chan_buf2: { kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: buffer_depth, frame_size: samples_per_data_set * sizeof_float }
out_buf:   { kotekan_buffer: standard, metadata_pool: none, num_frames: buffer_depth, frame_size: n_prn * record_floats * sizeof_float }
""" % (SIG, OUT_NAME, N, FS_STR, FOFF_STR, S["name"], NPRN, S["snr"], S["win"], S["dstep"], PRNS))

# Per-channel buffers, COVERING channels only (no buffer/stage on dead channels).
for c in COV:
    L.append("ch_%02d: { kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: buffer_depth, frame_size: samples_per_data_set * sizeof_float / %d }" % (c, N))
for c in COV:
    L.append("rc_%02d:  { kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: buffer_depth, frame_size: samples_per_data_set * sizeof_float / %d }" % (c, N))
    L.append("rec_%02d: { kotekan_buffer: standard, metadata_pool: none, num_frames: buffer_depth, frame_size: n_prn * record_floats * sizeof_float }" % c)
L.append("gather_buf: { kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: buffer_depth, frame_size: samples_per_data_set * sizeof_float / %d * %d }" % (N, NCOV))
# Beam-cube output: per PRN, OUT_HEADER(4) + NCOV per-channel |A| + UTC(float64=2) floats.
BEAM_RECFLOATS = 4 + NCOV + 2
L.append("beam_buf: { kotekan_buffer: standard, metadata_pool: none, num_frames: buffer_depth, frame_size: n_prn * %d * sizeof_float }" % BEAM_RECFLOATS)
L.append("")

L.append("""airspy_in:
    kotekan_stage: airspyInput
    out_buf: input_buf
    freq: %g
    sample_bw: %g
    gain_lna: 14
    gain_mix: 15
    gain_if: 15
    biast_power: true
fengine: { kotekan_stage: fftwEngine, in_buf: input_buf, out_buf: chan_buf, input_type: real, pfb_window: hamming }
# VALVE: drops a whole frame (cleanly -- fpga_seq preserved) if chan_buf2 is full, so the
# F-engine / airspy USB callback can never block. This is the fix live_full lacked.
valve: { kotekan_stage: Valve, in_buf: chan_buf, out_buf: chan_buf2 }""" % (S["freq"], SBW))

out_bufs = "[" + ", ".join("ch_%02d" % c for c in COV) + "]"
lo = "[" + ", ".join(str(c) for c in COV) + "]"
hi = "[" + ", ".join(str(c + 1) for c in COV) + "]"
L.append("split: { kotekan_stage: GnssSubbandSplit, in_buf: chan_buf2, out_bufs: %s, subband_lo: %s, subband_hi: %s }" % (out_bufs, lo, hi))
L.append("")

# Search arm: ship each covering channel over the network (inter-node mimic; drop_frames
# so a slow async search never forces track-arm drops) -> gather -> search.
for i, c in enumerate(COV):
    L.append("send_%02d: { kotekan_stage: bufferSend, buf: ch_%02d, server_ip: 127.0.0.1, server_port: %d, drop_frames: true }" % (c, c, PORT0 + i))
    L.append("recv_%02d: { kotekan_stage: bufferRecv, buf: rc_%02d, listen_port: %d, num_threads: 1 }" % (c, c, PORT0 + i))
gather_in = "[" + ", ".join("rc_%02d" % c for c in COV) + "]"
L.append("gather: { kotekan_stage: GnssChannelGather, in_bufs: %s, out_buf: gather_buf }" % gather_in)
L.append("search: { kotekan_stage: GnssChannelizedSearch, in_buf: gather_buf, channel_offset: %d, n_channels: %d, acquire_snr: %g, acquire_windows: %d%s }" % (COV[0], NCOV, S["snr"], S["win"], HOPS))
L.append("")

# Track arm: one tracker PER covering channel (n_channels:1) = the CHORD per-node
# correlation; the combiner coherently reassembles. Broker seeds every tracker.
for c in COV:
    L.append("track_%02d: { kotekan_stage: GnssChannelizedTracker, in_buf: ch_%02d, out_buf: rec_%02d, channel_offset: %d, n_channels: 1, pullin_chips: 1.0, pullin_step: 0.5, capture_utc0: 1.0, fll_gain: %g, fll_lock_amp: %g, hold_lock_amp: %g, hold_lock_records: %d%s }" % (c, c, c, c, FLL_GAIN, FLL_LOCK_AMP, HOLD_LOCK_AMP, HOLD_RECORDS, HOPS))
comb_in = "[" + ", ".join("rec_%02d" % c for c in COV) + "]"
_roll = (", integration_mode: rolling" if INTEG_MODE == "rolling" else "")
# Dataless-pilot deep integration: the combiner wipes a KNOWN secondary overlay (the L5 Q5
# NH20) at its searched alignment -> deep coherent |A| past the 1 ms primary period (slot 8 /
# deep_amplitude + deep_snr), no nav-bit estimate. Empty for signals without an overlay.
_overlay = (", secondary_overlay: %s" % OVERLAY) if OVERLAY else ""
L.append("combiner: { kotekan_stage: GnssCoherentCombiner, in_bufs: %s, out_buf: out_buf, integration_length: %d%s%s }"
         % (comb_in, INTEG_LEN, _roll, _overlay))
L.append('record: { kotekan_stage: rawFileWrite, in_buf: out_buf, base_dir: "%s", file_name: "level", file_ext: "raw", prefix_hostname: false, num_frames_per_file: 1000000, exit_after_n_files: 1000000 }' % BASE_DIR)
# Beam cube: the SAME per-channel tracker records, integrated PER CHANNEL (no cross-channel sum)
# -> beam(t, frequency). Frequency-resolves the antenna beam (matters across L5's ~10 MHz). Second
# consumer of rec_* alongside the combiner; it's light, and the valve absorbs any back-pressure.
L.append("beamcube: { kotekan_stage: GnssBeamCube, in_bufs: %s, out_buf: beam_buf, channel0: %d, integration_length: %d%s }"
         % (comb_in, COV[0], INTEG_LEN, _roll))
L.append('beamrec: { kotekan_stage: rawFileWrite, in_buf: beam_buf, base_dir: "%s", file_name: "beam", file_ext: "raw", prefix_hostname: false, num_frames_per_file: 1000000, exit_after_n_files: 1000000 }' % BASE_DIR)
L.append("")

# GPS-only browser viewer (no power stream / waterfall): serves the GPS Sky panel on :8080,
# polling REST + /gps_sky. SpawnProcess just drains out_buf (pure drain) so it can't backpressure.
L.append("spawn_pyviewer: { kotekan_stage: SpawnProcess, in_buf: out_buf, exec: '${VIEWER_PYTHON:-python3} python/scripts/js_viewer/livebeam_server.py --no-power-stream' }")

open("config/" + OUT_NAME, "w").write("\n".join(L) + "\n")
print("wrote config/%s: %s @ %g MHz, %d covering channels %s -> %d trackers; valve + GPS viewer"
      % (OUT_NAME, S["name"], S["freq"], NCOV, COV, NCOV))
print("  broker carrier: --carrier-hz %g  | TRK='track_{%02d..%02d}' (non-contiguous if subset)"
      % (S["freq"] * 1e6, COV[0], COV[-1]))
