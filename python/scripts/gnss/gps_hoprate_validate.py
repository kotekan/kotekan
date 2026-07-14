#!/usr/bin/env python3
"""Validate hop-rate channelized-replica generation against the exact full-PFB.

Pathway #5: the per-sample polyphase filter that makes R_c collapses to a per-CHIP
filter, because the spreading code is constant over a chip (~Fs/f_chip samples).
This harness replicates the exact C++ PFB (gnssChannelizedReplica.cpp / pfbPrototype)
and checks the chip-collapse reproduces it.

Three levels:
  exact_pfb       -- the reference (code x carrier -> push/fold -> rfft), bit-for-bit
                     the same algorithm as ChannelizedReplicaBank::channels().
  hoprate_exact   -- carrier-factored + chip-collapsed, filter summed over each chip
                     EXACTLY per hop. Should match the reference to ~machine precision
                     (it is the same sum regrouped) -> validates the math/bookkeeping.
  hoprate_slow    -- drop the fast (2*w_carrier) image -> the leakage we'd trade for
                     halving the work. Reports the error floor.
"""
import numpy as np


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
def exact_pfb(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate, f_carrier,
              cp, dop, window_start, n_hops, cd_sign=1.0, navbit=None):
    L = len(code)
    warm = num_taps - 1
    start = window_start - warm * fft_len
    cps = chip_rate / Fs * (1.0 + cd_sign * dop / f_carrier)
    wc = 2 * np.pi * (f_offset + dop) / Fs
    hist = np.zeros(fft_len * num_taps)
    chan = np.zeros((fft_len // 2 + 1, n_hops), dtype=complex)
    for h in range(n_hops + warm):
        n = start + h * fft_len + np.arange(fft_len)
        chip_abs = np.floor(cp + n * cps).astype(np.int64)        # absolute chip (for nav bit)
        chip = chip_abs % L
        bit = navbit(chip_abs) if navbit is not None else 1.0     # exact: data on the code
        block = np.where(n >= 0, bit * code[chip] * np.cos(wc * n), 0.0)
        hist[fft_len:] = hist[:-fft_len]            # slide old back
        hist[:fft_len] = block[::-1]                # newest sample -> hist[0]
        fold = sum(proto[q * fft_len:(q + 1) * fft_len] * hist[q * fft_len:(q + 1) * fft_len]
                   for q in range(num_taps))
        spec = np.fft.rfft(fold)
        if h >= warm:
            chan[:, h - warm] = spec
    return chan


# ---- hop-rate generator: carrier-factored + chip-collapsed ------------------
def hoprate(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate, f_carrier,
            cp, dop, window_start, n_hops, j, cd_sign=1.0, slow_only=False):
    Lc = len(code)
    Lf = fft_len * num_taps
    cps = chip_rate / Fs * (1.0 + cd_sign * dop / f_carrier)
    wc = 2 * np.pi * (f_offset + dop) / Fs
    k = np.arange(Lf)
    # R_j[m] = Σ_k proto[k] e^{-i2πjk/N'} code[n_m-k] cos(wc(n_m-k)),  N'=fft_len
    #        = 1/2 [ e^{+i wc n_m} Σ filterA·code + e^{-i wc n_m} Σ filterB·code ]
    # filterB (slow, channel-carrier offset 2πj/N' - wc) dominates a covering channel;
    # filterA (fast, +2wc) is the image we can drop.
    filtA = proto * np.exp(-1j * (2 * np.pi * j / fft_len + wc) * k)
    filtB = proto * np.exp(-1j * (2 * np.pi * j / fft_len - wc) * k)
    out = np.zeros(n_hops, dtype=complex)
    for m in range(n_hops):
        n_m = window_start + (m + 1) * fft_len - 1
        chip = np.floor(cp + (n_m - k) * cps).astype(np.int64)   # chip per tap
        c0 = chip.min()
        ci = chip - c0
        nb = ci.max() + 1
        WB = (np.bincount(ci, filtB.real, nb) + 1j * np.bincount(ci, filtB.imag, nb))
        codev = code[(c0 + np.arange(nb)) % Lc]
        sB = np.sum(codev * WB)
        val = np.exp(-1j * wc * n_m) * sB
        if not slow_only:
            WA = (np.bincount(ci, filtA.real, nb) + 1j * np.bincount(ci, filtA.imag, nb))
            val += np.exp(1j * wc * n_m) * np.sum(codev * WA)
        out[m] = 0.5 * val
    return out


def precompute_Wbank(filt, fft_len, num_taps, cps, n_phi, n_chips):
    """Per-sub-chip-phase filter sums: Wbank[g, d] = Σ_{k: chip0-chip(k)=d} filt[k] at
    sub-chip phase φ_g. chip(k) = floor(C - k·cps), C = chip0 + φ, so d = -floor(φ - k·cps).
    Depends only on (filt, cps) -> rebuild slowly as the Doppler drifts."""
    Lf = fft_len * num_taps
    k = np.arange(Lf)
    bank = np.zeros((n_phi, n_chips), dtype=complex)
    for g in range(n_phi):
        phi = (g + 0.5) / n_phi
        d = (-np.floor(phi - k * cps)).astype(np.int64)
        m = d < n_chips
        bank[g] = (np.bincount(d[m], filt.real[m], n_chips)
                   + 1j * np.bincount(d[m], filt.imag[m], n_chips))
    return bank


def hoprate_phibank(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate, f_carrier,
                    cp, dop, window_start, n_hops, j, n_phi=64, n_chips=40, cd_sign=1.0):
    """The actual fast generator: precompute W(φ) once, then each hop is n_chips MACs
    (+ a per-hop carrier phasor + a φ interpolation). Both images kept."""
    Lc = len(code)
    cps = chip_rate / Fs * (1.0 + cd_sign * dop / f_carrier)
    wc = 2 * np.pi * (f_offset + dop) / Fs
    k = np.arange(fft_len * num_taps)
    filtA = proto * np.exp(-1j * (2 * np.pi * j / fft_len + wc) * k)
    filtB = proto * np.exp(-1j * (2 * np.pi * j / fft_len - wc) * k)
    WA = precompute_Wbank(filtA, fft_len, num_taps, cps, n_phi, n_chips)
    WB = precompute_Wbank(filtB, fft_len, num_taps, cps, n_phi, n_chips)
    out = np.zeros(n_hops, dtype=complex)
    dd = np.arange(n_chips)
    for m in range(n_hops):
        n_m = window_start + (m + 1) * fft_len - 1
        C = cp + n_m * cps
        chip0 = int(np.floor(C))
        phi = C - chip0
        g = phi * n_phi - 0.5                       # periodic linear interp on the φ grid
        g0 = int(np.floor(g)) % n_phi
        f = g - np.floor(g)
        wa = (1 - f) * WA[g0] + f * WA[(g0 + 1) % n_phi]
        wb = (1 - f) * WB[g0] + f * WB[(g0 + 1) % n_phi]
        codev = code[(chip0 - dd) % Lc]
        out[m] = 0.5 * (np.exp(1j * wc * n_m) * np.sum(codev * wa)
                        + np.exp(-1j * wc * n_m) * np.sum(codev * wb))
    return out


def hoprate_prefix(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate, f_carrier,
                   cp, dop, window_start, n_hops, j, n_chips=40, cd_sign=1.0,
                   navbit=None, navbit_per_hop=False):
    """EXACT hop-rate generator via the prefix-sum: each chip's filter contribution is
    Phi[k_hi] - Phi[k_lo-1] over its INTEGER tap range -- no interpolation, because the
    chip boundaries land between integer taps. O(n_chips)/hop, memory = the prefix sum.
    Edge 1 (chip edge in the filter) is exact here. navbit applies the +-1 data; per-chip
    (navbit_per_hop=False) is exact (the bit edge is chip-aligned), per-hop is the cheap
    post-multiply approximation that smears the ~P hops straddling a bit edge."""
    Lc = len(code)
    Lf = fft_len * num_taps
    cps = chip_rate / Fs * (1.0 + cd_sign * dop / f_carrier)
    wc = 2 * np.pi * (f_offset + dop) / Fs
    k = np.arange(Lf)
    filtA = proto * np.exp(-1j * (2 * np.pi * j / fft_len + wc) * k)
    filtB = proto * np.exp(-1j * (2 * np.pi * j / fft_len - wc) * k)
    PhiA = np.concatenate([[0], np.cumsum(filtA)])               # PhiA[k] = Σ filtA[0:k]
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
                cv *= navbit(cidx)                              # exact: bit per chip
            sA += cv * (PhiA[khi + 1] - PhiA[klo])
            sB += cv * (PhiB[khi + 1] - PhiB[klo])
        out[m] = 0.5 * hop_bit * (np.exp(1j * wc * n_m) * sA + np.exp(-1j * wc * n_m) * sB)
    return out


def db(x):
    return 20 * np.log10(x + 1e-300)


def main():
    # airspy-scale params (== ChannelizedReplicaBank usage): Fs 5 MSPS, N=20.
    Fs, fft_len, num_taps = 5.0e6, 40, 4
    chip_rate, f_offset, f_carrier = 1.023e6, 1.25e6, 1575.42e6
    proto = pfb_prototype(fft_len, num_taps, "hamming")
    rng = np.random.RandomState(1)
    code = rng.choice([-1.0, 1.0], size=1023)
    window_start, n_hops = 1_000_000, 125
    print("samples/chip = Fs/chip_rate = %.2f ; filter spans %.1f chips"
          % (Fs / chip_rate, num_taps * fft_len / (Fs / chip_rate)))

    print("\nA) exact chip-collapse (per-hop, O(filter)) vs reference:")
    print("   dop     j  | both-images(dB)  slow-only(dB)")
    for dop in (0.0, 2000.0, -3500.0):
        ref = exact_pfb(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate,
                        f_carrier, 300.0, dop, window_start, n_hops)
        for j in (8, 10, 12):
            r = ref[j]
            rms = np.sqrt(np.mean(np.abs(r) ** 2))
            he = hoprate(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate,
                         f_carrier, 300.0, dop, window_start, n_hops, j)
            hs = hoprate(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate,
                         f_carrier, 300.0, dop, window_start, n_hops, j, slow_only=True)
            print("   %+5.0f  %3d |   %7.1f       %7.1f"
                  % (dop, j, db(np.sqrt(np.mean(np.abs(he - r) ** 2)) / rms),
                     db(np.sqrt(np.mean(np.abs(hs - r) ** 2)) / rms)))

    print("\nB) FAST φ-bank (per-hop O(n_chips)=%d) vs reference, by φ-grid size:" % 40)
    dop = 2000.0
    ref = exact_pfb(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate,
                    f_carrier, 300.0, dop, window_start, n_hops)
    print("   n_phi |  j=8     j=10    j=12   (err dB)")
    for n_phi in (64, 256, 1024, 4096):
        errs = []
        for j in (8, 10, 12):
            r = ref[j]
            rms = np.sqrt(np.mean(np.abs(r) ** 2))
            hp = hoprate_phibank(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate,
                                 f_carrier, 300.0, dop, window_start, n_hops, j,
                                 n_phi=n_phi, n_chips=40)
            errs.append(db(np.sqrt(np.mean(np.abs(hp - r) ** 2)) / rms))
        print("   %4d  | %6.1f  %6.1f  %6.1f" % (n_phi, errs[0], errs[1], errs[2]))

    def rel(a, r):
        return db(np.sqrt(np.mean(np.abs(a - r) ** 2)) / np.sqrt(np.mean(np.abs(r) ** 2)))

    print("\nC) EXACT prefix-sum generator (Edge 1: chip edge inside the filter):")
    print("   the chip's filter sum = Phi[k_hi]-Phi[k_lo-1] over integer taps -> exact, no interp")
    dop = 2000.0
    ref = exact_pfb(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate, f_carrier,
                    300.0, dop, window_start, n_hops)
    for j in (8, 10, 12):
        hp = hoprate_prefix(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate,
                            f_carrier, 300.0, dop, window_start, n_hops, j, n_chips=40)
        print("   j=%2d  err %6.1f dB" % (j, rel(hp, ref[j])))

    print("\nD) NAV-BIT EDGE (Edge 2): a 20 ms bit edge falls on a code-period (=chip) boundary,")
    print("   so applying the +-1 bit PER CHIP is exact; the per-hop post-multiply smears ~P hops.")
    Lc = len(code)
    center = 300.0 + (window_start + n_hops // 2 * fft_len) * (chip_rate / Fs)
    edge_chip = round(center / Lc) * Lc                      # a code-period boundary in the window

    def navbit(c):
        return np.where(np.asarray(c) < edge_chip, 1.0, -1.0)

    ref = exact_pfb(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate, f_carrier,
                    300.0, dop, window_start, n_hops, navbit=navbit)  # the TRUE wiped replica
    for j in (8, 10, 12):
        exact_w = hoprate_prefix(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate,
                                 f_carrier, 300.0, dop, window_start, n_hops, j,
                                 navbit=navbit, navbit_per_hop=False)
        approx_w = hoprate_prefix(code, fft_len, num_taps, proto, Fs, f_offset, chip_rate,
                                  f_carrier, 300.0, dop, window_start, n_hops, j,
                                  navbit=navbit, navbit_per_hop=True)
        worst = db(np.max(np.abs(approx_w - ref[j])) / np.sqrt(np.mean(np.abs(ref[j]) ** 2)))
        nbad = int(np.sum(np.abs(approx_w - ref[j]) > 0.01 * np.sqrt(np.mean(np.abs(ref[j]) ** 2))))
        print("   j=%2d | per-chip(exact) %6.1f dB | per-hop(approx) rms %6.1f dB, worst-hop %5.1f "
              "dB, %d straddling hops" % (j, rel(exact_w, ref[j]), rel(approx_w, ref[j]), worst, nbad))


if __name__ == "__main__":
    main()
