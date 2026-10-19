#!/usr/bin/env python3
"""Validate hop-rate channelized-replica generation against the exact full-PFB.

The per-sample polyphase filter that makes R_c collapses to a per-chip filter, because the
spreading code is constant over a chip (~Fs/f_chip samples). This harness replicates the
exact C++ PFB (gnssChannelizedReplica.cpp / pfbPrototype) and prints how closely the
chip-collapse reproduces it when a nav-bit edge falls on a code-period boundary (the case
tests/boost/test_gnss_channelized_replica.cpp leaves to this file). It exits 1 unless the
bit applied per chip matches the reference to EXACT_DB and the per-hop post-multiply is off
by more than 1% on exactly the hops whose filter span straddles the edge.

  exact_pfb       -- the reference (code x carrier -> push/fold -> rfft), bit-for-bit
                     the same algorithm as ChannelizedReplicaBank::channels().
  hoprate_prefix  -- the prefix-sum hop-rate generator, with the bit applied per chip
                     (exact) and per hop (the cheap post-multiply, which smears the hops
                     straddling the edge).
"""
import sys

import numpy as np

# Rounding-level agreement: the per-chip path measures about -200 dB.
EXACT_DB = -100.0


# ---- exact prototype (matches lib/stages/pfbPrototype.cpp) -------------------
def pfb_prototype(num_chan, num_taps, window="hamming"):
    L = num_chan * num_taps
    r = np.arange(L)
    x = (r - (L - 1) / 2.0) / num_chan
    sinc = np.where(x == 0.0, 1.0, np.sin(np.pi * x) / (np.pi * x))
    t = r / (L - 1)
    if window == "hamming":
        w = 0.54 - 0.46 * np.cos(2 * np.pi * t)
    elif window == "hann":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * t)
    elif window == "rect":
        w = np.ones(L)
    else:
        raise ValueError(window)
    h = sinc * w
    return h * (num_chan / h.sum())


# ---- exact reference: ChannelizedReplicaBank::channels() in numpy -----------
def exact_pfb(
    code,
    fft_len,
    num_taps,
    proto,
    Fs,
    f_offset,
    chip_rate,
    f_carrier,
    cp,
    dop,
    window_start,
    n_hops,
    cd_sign=1.0,
    navbit=None,
):
    L = len(code)
    warm = num_taps - 1
    start = window_start - warm * fft_len
    cps = chip_rate / Fs * (1.0 + cd_sign * dop / f_carrier)
    wc = 2 * np.pi * (f_offset + dop) / Fs
    hist = np.zeros(fft_len * num_taps)
    chan = np.zeros((fft_len // 2 + 1, n_hops), dtype=complex)
    for h in range(n_hops + warm):
        n = start + h * fft_len + np.arange(fft_len)
        chip_abs = np.floor(cp + n * cps).astype(
            np.int64
        )  # absolute chip (for nav bit)
        chip = chip_abs % L
        bit = navbit(chip_abs) if navbit is not None else 1.0  # exact: data on the code
        block = np.where(n >= 0, bit * code[chip] * np.cos(wc * n), 0.0)
        hist[fft_len:] = hist[:-fft_len]  # slide old back
        hist[:fft_len] = block[::-1]  # newest sample -> hist[0]
        fold = sum(
            proto[q * fft_len : (q + 1) * fft_len]
            * hist[q * fft_len : (q + 1) * fft_len]
            for q in range(num_taps)
        )
        spec = np.fft.rfft(fold)
        if h >= warm:
            chan[:, h - warm] = spec
    return chan


def hoprate_prefix(
    code,
    fft_len,
    num_taps,
    proto,
    Fs,
    f_offset,
    chip_rate,
    f_carrier,
    cp,
    dop,
    window_start,
    n_hops,
    j,
    n_chips=40,
    cd_sign=1.0,
    navbit=None,
    navbit_per_hop=False,
):
    """Exact hop-rate generator via the prefix sum: each chip's filter contribution is
    Phi[k_hi+1] - Phi[k_lo] (Phi[k] = sum of filt[0:k]) over its integer tap range -- no
    interpolation, because the chip boundaries land between integer taps. O(n_chips) per hop,
    memory = the prefix sum. Chip edges inside the filter span are exact. navbit applies the
    +-1 data: per chip (navbit_per_hop=False) is exact (the bit edge is chip-aligned), per hop
    is the cheap post-multiply approximation that smears the ~num_taps hops straddling a bit
    edge."""
    Lc = len(code)
    Lf = fft_len * num_taps
    cps = chip_rate / Fs * (1.0 + cd_sign * dop / f_carrier)
    wc = 2 * np.pi * (f_offset + dop) / Fs
    k = np.arange(Lf)
    filtA = proto * np.exp(-1j * (2 * np.pi * j / fft_len + wc) * k)
    filtB = proto * np.exp(-1j * (2 * np.pi * j / fft_len - wc) * k)
    PhiA = np.concatenate([[0], np.cumsum(filtA)])  # PhiA[k] = Σ filtA[0:k]
    PhiB = np.concatenate([[0], np.cumsum(filtB)])
    out = np.zeros(n_hops, dtype=complex)
    for m in range(n_hops):
        n_m = window_start + (m + 1) * fft_len - 1
        C = cp + n_m * cps
        chip0 = int(np.floor(C))
        phi = C - chip0
        hop_bit = navbit(chip0) if (navbit is not None and navbit_per_hop) else 1.0
        sA = sB = 0.0 + 0j
        for d in range(n_chips):
            klo = max(int(np.floor((phi + d - 1) / cps)) + 1, 0)
            khi = min(int(np.floor((phi + d) / cps)), Lf - 1)
            if khi < klo:
                continue
            cidx = chip0 - d
            cv = code[cidx % Lc]
            if navbit is not None and not navbit_per_hop:
                cv *= navbit(cidx)  # exact: bit per chip
            sA += cv * (PhiA[khi + 1] - PhiA[klo])
            sB += cv * (PhiB[khi + 1] - PhiB[klo])
        out[m] = (
            0.5 * hop_bit * (np.exp(1j * wc * n_m) * sA + np.exp(-1j * wc * n_m) * sB)
        )
    return out


def db(x):
    return 20 * np.log10(x + 1e-300)


def main():
    # airspy-scale params, as in test_gnss_channelized_replica.cpp's 5 MSPS case: N=20
    # channels, fft_len = 2N.
    Fs, fft_len, num_taps = 5.0e6, 40, 4
    chip_rate, f_offset, f_carrier = 1.023e6, 1.25e6, 1575.42e6
    proto = pfb_prototype(fft_len, num_taps, "hamming")
    rng = np.random.RandomState(1)
    code = rng.choice([-1.0, 1.0], size=1023)
    window_start, n_hops = 1_000_000, 125
    print(
        "samples/chip = Fs/chip_rate = %.2f ; filter spans %.1f chips"
        % (Fs / chip_rate, num_taps * fft_len / (Fs / chip_rate))
    )

    def rel(a, r):
        return db(
            np.sqrt(np.mean(np.abs(a - r) ** 2)) / np.sqrt(np.mean(np.abs(r) ** 2))
        )

    dop = 2000.0
    print("\nNAV-BIT EDGE: a 20 ms bit edge falls on a code-period (=chip) boundary,")
    print(
        "   so applying the +-1 bit PER CHIP is exact; the per-hop post-multiply smears the "
        "hops straddling the edge."
    )
    Lc = len(code)
    center = 300.0 + (window_start + n_hops // 2 * fft_len) * (chip_rate / Fs)
    edge_chip = round(center / Lc) * Lc  # a code-period boundary in the window

    def navbit(c):
        return np.where(np.asarray(c) < edge_chip, 1.0, -1.0)

    # The hops whose filter span (num_taps * fft_len samples ending at the hop's newest
    # sample) holds chips on both sides of the edge.
    cps = chip_rate / Fs * (1.0 + dop / f_carrier)
    n_end = window_start + (np.arange(n_hops) + 1) * fft_len - 1
    first = np.floor(300.0 + (n_end - num_taps * fft_len + 1) * cps)
    last = np.floor(300.0 + n_end * cps)
    straddle = np.flatnonzero((first < edge_chip) & (last >= edge_chip))
    fails = [] if len(straddle) else ["no hop straddles the edge: nothing is tested"]

    ref = exact_pfb(
        code,
        fft_len,
        num_taps,
        proto,
        Fs,
        f_offset,
        chip_rate,
        f_carrier,
        300.0,
        dop,
        window_start,
        n_hops,
        navbit=navbit,
    )  # the exact data-wiped replica
    for j in (8, 10, 12):
        exact_w = hoprate_prefix(
            code,
            fft_len,
            num_taps,
            proto,
            Fs,
            f_offset,
            chip_rate,
            f_carrier,
            300.0,
            dop,
            window_start,
            n_hops,
            j,
            navbit=navbit,
            navbit_per_hop=False,
        )
        approx_w = hoprate_prefix(
            code,
            fft_len,
            num_taps,
            proto,
            Fs,
            f_offset,
            chip_rate,
            f_carrier,
            300.0,
            dop,
            window_start,
            n_hops,
            j,
            navbit=navbit,
            navbit_per_hop=True,
        )
        worst = db(
            np.max(np.abs(approx_w - ref[j])) / np.sqrt(np.mean(np.abs(ref[j]) ** 2))
        )
        bad = np.flatnonzero(
            np.abs(approx_w - ref[j]) > 0.01 * np.sqrt(np.mean(np.abs(ref[j]) ** 2))
        )
        nbad = len(bad)
        e_exact = rel(exact_w, ref[j])
        print(
            "   j=%2d | per-chip(exact) %6.1f dB | per-hop(approx) rms %6.1f dB, worst-hop %5.1f "
            "dB, %d straddling hops" % (j, e_exact, rel(approx_w, ref[j]), worst, nbad)
        )
        if not e_exact < EXACT_DB:
            fails.append(
                "j=%d per-chip %.1f dB, limit %.0f dB" % (j, e_exact, EXACT_DB)
            )
        if not np.array_equal(bad, straddle):
            fails.append(
                "j=%d per-hop off on hops %s, the edge straddles hops %s"
                % (j, bad.tolist(), straddle.tolist())
            )
    for f in fails:
        print("FAIL", f)
    print("FAIL" if fails else "OK")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
