#!/usr/bin/env python3
"""Validate the CHANNELIZED (subband) L2C CM despread against the MONOLITHIC one.

The L1->subband move had subtleties; L2C adds one the monolithic correlator handles
explicitly but the channelized replica does NOT: L2C CM is TIME-MULTIPLEXED with CL
(chip-by-chip) at 2x the CM chip rate. GnssCorrelatorCore builds the CM replica at the
combined 1.023 Mcps rate with CM chips at the tdm parity and ZEROS at the CL positions
(comb_mult=2). ChannelizedReplicaBank::code_chip just indexes the 511.5 kcps CM code
directly -- no comb_mult/tdm. This script asks, numerically: does that break detection?

Synthetic L2C: random CM + CL codes (the interleaving is code-agnostic), interleaved at
1.023 Mcps, real passband at Fs/4 IF + Doppler, optional noise. We despread three ways at
the true (cp,dop) and at several WRONG cp (the floor), and report peak / off-peak margin:

  * MONOLITHIC     -- CM-only at combined rate (zeros at CL): the reference.
  * SUBBAND-current -- channelized replica as the code does it NOW (511.5 kcps CM, held).
  * SUBBAND-tdmfix  -- channelized replica at combined rate w/ zeros at CL (the candidate fix).

If current ~ monolithic, the subband path is fine and the live no-detect is SNR/antenna; if
current collapses and tdmfix recovers, the missing comb_mult/tdm is the bug.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gps_hoprate_validate import pfb_prototype, exact_pfb

Fs = 5.0e6
F_OFF = 1.25e6            # Fs/4 IF
FC = 1227.6e6            # L2 carrier
CHIP_CM = 511.5e3        # CM chip rate
LCM = 10230             # CM chips / 20 ms period
LCOMB = 2 * LCM          # combined (CM+CL) chips
TCODE = 20e-3
NS = int(round(Fs * TCODE))   # 100000 samples / code period
FFT_LEN, NUM_TAPS = 24, 4     # 12 channels @ ~208 kHz (matches live_l2c.yaml)
PROTO = pfb_prototype(FFT_LEN, NUM_TAPS, "hamming")
N_BINS = FFT_LEN // 2 + 1

# True signal params.
CP_CM = 123.4            # true CM code phase (chips)
DOP = 1500.0            # true Doppler (Hz)


def make_codes(seed=0):
    rng = np.random.default_rng(seed)
    cm = (rng.integers(0, 2, LCM) * 2 - 1).astype(np.float64)
    cl = (rng.integers(0, 2, LCM) * 2 - 1).astype(np.float64)
    comb = np.empty(LCOMB)          # transmitted: CM at even chips, CL at odd
    comb[0::2], comb[1::2] = cm, cl
    cm_only = np.zeros(LCOMB)        # CM-only at combined rate (zeros where CL is)
    cm_only[0::2] = cm
    return cm, cl, comb, cm_only


def make_signal(comb, noise=0.0, seed=1):
    """Real passband ADC samples of the interleaved L2C signal at (CP_CM, DOP)."""
    n = np.arange(NS)
    cps_comb = (2 * CHIP_CM) / Fs * (1.0 + DOP / FC)   # combined chips/sample
    chip = np.floor(2 * CP_CM + n * cps_comb).astype(np.int64) % LCOMB
    phase = 2 * np.pi * (F_OFF + DOP) / Fs * n
    x = comb[chip] * np.cos(phase)
    if noise > 0.0:
        x = x + noise * np.random.default_rng(seed).standard_normal(NS)
    return x


def pfb_data(x):
    """F-engine: fold + rfft the raw samples per hop -> (N_BINS, n_hops)."""
    warm = NUM_TAPS - 1
    n_hops = NS // FFT_LEN - warm
    hist = np.zeros(FFT_LEN * NUM_TAPS)
    chan = np.zeros((N_BINS, n_hops), dtype=complex)
    for h in range(n_hops + warm):
        block = x[h * FFT_LEN:(h + 1) * FFT_LEN]
        hist[FFT_LEN:] = hist[:-FFT_LEN]
        hist[:FFT_LEN] = block[::-1]
        fold = sum(PROTO[q * FFT_LEN:(q + 1) * FFT_LEN] * hist[q * FFT_LEN:(q + 1) * FFT_LEN]
                   for q in range(NUM_TAPS))
        if h >= warm:
            chan[:, h - warm] = np.fft.rfft(fold)
    return chan


def mono_despread(x, cm_only, cp_cm):
    """Monolithic CM despread at a trial cp: CM-only replica (zeros at CL), combined rate."""
    n = np.arange(NS)
    cps_comb = (2 * CHIP_CM) / Fs * (1.0 + DOP / FC)
    chip = np.floor(2 * cp_cm + n * cps_comb).astype(np.int64) % LCOMB
    repl = cm_only[chip] * np.exp(-1j * 2 * np.pi * (F_OFF + DOP) / Fs * n)
    return abs(np.sum(x * repl))


def subband_despread(data_chan, code, chip_rate, cp_units, bins):
    """Channelized despread at a trial cp: build the channelized replica (exact_pfb),
    correlate against the F-engine channels over the covering bins. exact_pfb's first
    OUTPUT hop sits at window_start (warm-up consumes the warm hops BEFORE it), whereas
    pfb_data's hop 0 is at sample warm*fft_len -- so align with window_start=warm*fft_len."""
    n_hops = data_chan.shape[1]
    ws = (NUM_TAPS - 1) * FFT_LEN
    repl = exact_pfb(code, FFT_LEN, NUM_TAPS, PROTO, Fs, F_OFF, chip_rate, FC,
                     cp_units, DOP, ws, n_hops)
    return abs(np.sum(np.conj(repl[bins, :]) * data_chan[bins, :]))


def margin(peak_fn):
    """peak (true cp) vs off-peak floor (wrong cps), in dB."""
    peak = peak_fn(CP_CM)
    offs = [CP_CM + d for d in (3.0, 17.0, 101.0, 503.0, 2003.0, 5003.0)]
    floor = np.array([peak_fn(c) for c in offs])
    return peak, floor.mean(), 20 * np.log10(peak / floor.mean())


def main():
    cm, cl, comb, cm_only = make_codes()
    # Covering bins: the signal occupies ~Fs/4 +- ~1 MHz; use the bins the carrier+code light.
    carrier_bin = F_OFF / (Fs / FFT_LEN)            # = 6
    bins = [b for b in range(N_BINS) if abs(b - carrier_bin) <= 5]   # bins 1..11
    print("Fs=%.1f MHz  IF=%.2f MHz  carrier_bin=%.0f  covering bins=%s" %
          (Fs / 1e6, F_OFF / 1e6, carrier_bin, bins))
    print("CM: %d chips @ %.1f kcps (20 ms); combined CM+CL @ %.3f Mcps\n" %
          (LCM, CHIP_CM / 1e3, 2 * CHIP_CM / 1e6))

    for noise, tag in [(0.0, "clean"), (30.0, "noisy (per-sample SNR ~ -30 dB, GPS-like)")]:
        x = make_signal(comb, noise=noise)
        data_chan = pfb_data(x)
        print("=== %s ===" % tag)

        pk, fl, mg = margin(lambda c: mono_despread(x, cm_only, c))
        print("  MONOLITHIC       peak=%.3e floor=%.3e  margin=%6.1f dB" % (pk, fl, mg))

        pk, fl, mono = margin(lambda c: mono_despread(x, cm_only, c))  # recompute for the delta
        _ = pk
        pk, fl, mg_old = margin(lambda c: subband_despread(data_chan, cm, CHIP_CM, c, bins))
        print("  SUBBAND-old      peak=%.3e floor=%.3e  margin=%6.1f dB   (bare 511.5 kcps CM, held -- the BUG)" % (pk, fl, mg_old))

        pk, fl, mg_fix = margin(lambda c: subband_despread(data_chan, cm_only, 2 * CHIP_CM, 2 * c, bins))
        print("  SUBBAND-fixed    peak=%.3e floor=%.3e  margin=%6.1f dB   (combined rate + tdm zeros = ChannelizedReplicaBank now)" % (pk, fl, mg_fix))
        print("    -> fix recovers %.1f dB over the old subband replica (within %.1f dB of monolithic)\n"
              % (mg_fix - mg_old, abs(mono - mg_fix)))


if __name__ == "__main__":
    main()
