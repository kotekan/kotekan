#!/usr/bin/env python3
"""Synthetic GuoWang-like signals, and the validation of guowang_search.py on them (#218).

synth() makes complex baseband: a +-1 code (rectangular chips, no transmit filter) with B3I-style
NH20 + 50 bps data, LEO carrier and code Doppler (Doppler f + rate r t, code from the same clock),
in white complex Gaussian noise of unit power per sample. snr_db is signal / noise per sample over
the full sample rate, so C/N0 = snr_db + 10 log10(fs). Optional 4-bit quantisation (noise at 1.8
LSB rms per component, the quiet level of the 09-23-gain-cut channels).

    python guowang_synth.py validate [--quick] [--out PREFIX]     # the case table
    python guowang_synth.py make out.npz --snr 1.6 --prn 23 --dur 2  # one capture for guowang_search
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import guowang_codes as gcodes  # noqa: E402
import guowang_search as gs  # noqa: E402

NH20 = np.array([1 - 2 * int(b) for b in "00000100110101001110"], np.int8)


def synth(
    fs,
    dur,
    code,
    chip_rate,
    f_rf,
    snr_db,
    dopp=12e3,
    rate=-80.0,
    f_off=0.0,
    chip0=1234.5,
    phase=0.3,
    data_bps=50.0,
    nh=True,
    q4=False,
    seed=0,
    chip_offset=0.0,
):
    """Complex64 samples; f_off is the carrier's offset from the sample centre frequency;
    chip_offset (Hz) puts the chip clock off the carrier's (a code Doppler that is not the carrier's)."""
    rng = np.random.default_rng(seed)
    N = len(code)
    T = N / chip_rate
    n = int(dur * fs)
    A = 10 ** (snr_db / 20.0) if snr_db is not None else 0.0
    per_bit = max(1, int(round(1.0 / (data_bps * T)))) if data_bps else 0
    bits = (
        rng.choice(np.array([-1, 1], np.int8), int(dur / T) // max(per_bit, 1) + 4)
        if data_bps
        else None
    )
    x = np.empty(n, np.complex64)
    step = 1 << 21
    for a in range(0, n, step):
        t = np.arange(a, min(a + step, n)) / fs
        chi = (
            chip0
            + (chip_rate + chip_offset) * t
            + chip_rate * (dopp * t + 0.5 * rate * t * t) / f_rf
        )
        k = np.floor(chi).astype(np.int64)
        p = k // N
        s = code[k % N].astype(np.float32)
        if nh:
            s *= NH20[p % 20]
        if data_bps:
            s *= bits[p // per_bit]
        ph = 2 * np.pi * ((f_off + dopp) * t + 0.5 * rate * t * t) + phase
        w = (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t))) * np.sqrt(
            0.5
        )
        x[a : a + len(t)] = A * s * np.exp(1j * ph) + w
    if q4:
        g = 1.8 * np.sqrt(2.0)
        x = (
            np.clip(np.round(x.real * g), -8, 7)
            + 1j * np.clip(np.round(x.imag * g), -8, 7)
        ).astype(np.complex64)
    return x


def band_ratio(snr_db, fs, chip_rate, band=14e6):
    """The rail recorder's burst statistic for this SNR: band power 1262-1276 MHz over the quiet
    level, for a rectangular-chip BPSK centred in the band and noise white over fs."""
    f = np.linspace(-band / 2, band / 2, 4001)
    frac = np.trapezoid(np.sinc(f / chip_rate) ** 2, f) / chip_rate
    return 1.0 + 10 ** (snr_db / 10.0) * frac / (band / fs)


def channelise(x, n_chan, taps=4):
    """Critically sampled polyphase filter bank, taps x n_chan sinc-Hann prototype (the CHIME / CHORD
    F-engine's form): [n_chan, n_t] channels at fs / n_chan, channel k centred at
    (k - n_chan/2) fs / n_chan, as the stitcher expects; conjugated like the CHORD F-engine output.
    Output sample m covers input [m n_chan, (m + taps) n_chan)."""
    L = taps * n_chan
    n = np.arange(L)
    h = (np.sinc((n - L / 2.0 + 0.5) / n_chan) * np.hanning(L)).astype(np.float32)
    nb = (len(x) - L) // n_chan + 1
    out = np.empty((nb, n_chan), np.complex64)
    for a in range(0, nb, 4096):
        m = np.arange(a, min(a + 4096, nb))
        seg = x[(m[:, None] * n_chan + n[None, :])] * h  # [blocks, L]
        out[m] = np.fft.fft(seg.reshape(len(m), taps, n_chan).sum(axis=1), axis=1)
    return np.conj(np.fft.fftshift(out, axes=1).T)


# --------------------------------------------------------------------------- validation


def _truth_chips(code, soft):
    """Chip errors of recovered soft chips against the true code, at the best circular alignment and
    polarity."""
    N = len(code)
    r = np.fft.ifft(
        np.fft.fft(np.sign(soft)) * np.conj(np.fft.fft(code.astype(float)))
    ).real
    k = int(np.argmax(np.abs(r)))
    sgn = np.sign(r[k])
    err = int(np.sum(np.sign(soft) * sgn != np.roll(code, k)))
    return err, N


def run_case(name, x, fs, fc, lobe, families, truth, run_kw, logf):
    t = time.time()
    old = gs.log
    lines = []
    gs.log = lambda *a: lines.append(" ".join(str(v) for v in a))
    try:
        rep = gs.run(x, fs, fc, lobe, families, **run_kw)
    finally:
        gs.log = old
    wall = time.time() - t
    with open(logf, "a") as f:
        f.write(
            "=== %s  truth %s\n"
            % (name, json.dumps({k: v for k, v in truth.items() if k != "code"}))
        )
        f.write("\n".join(lines) + "\n\n")
    f_rf, rc = gs.LOBES[lobe]
    row = dict(case=name, wall=wall)
    cr = rep["chip_rate"]
    t_mid = rep["carrier"]["lines"][0]["t_c"]
    fd_mid = truth.get("dopp", 0) + truth.get("rate", 0) * t_mid
    row["chiprate"] = (
        "%s %.0f" % ("Y" if cr["detected"] else "n", cr["code_doppler"] - fd_mid)
        if cr["detected"]
        else "n"
    )
    pk = rep["period"]["peaks"]
    T_true = truth["N"] / truth["rc"] / (1.0 + fd_mid / f_rf) if "N" in truth else None
    if pk and T_true:
        fp = gs.fundamental(pk)
        row["period"] = (
            "Y %+.1fns" % ((fp - T_true) * 1e9) if fp is not None else "n (no standout)"
        )
    else:
        row["period"] = "Y" if pk else "n"
    car = rep["carrier"]["lines"][0]
    if car["detected"]:
        row["carrier"] = "Y %+.1fHz %+.2fHz/s" % (
            car["f_c"] - fd_mid,
            car["rate"] - truth.get("rate", 0),
        )
    else:
        row["carrier"] = "n"
    row["fold"] = "-"
    if "fold" in rep:
        idn = rep["identify"]
        best = max(
            ((v["top"][0][0], v["top"][0][1], k) for k, v in idn.items() if "top" in v),
            default=(0, "-", "-"),
        )
        row["fold"] = "SNR %.0f e1 %.0f %s rho %.2f" % (
            rep["fold"]["chip_snr"],
            rep["fold"]["e1"],
            best[1],
            best[0],
        )
        if truth.get("code") is not None and len(truth["code"]) == len(rep["_chips"]):
            err, n = _truth_chips(truth["code"], rep["_chips"])
            row["fold"] += " chip err %d/%d" % (err, n)
        elif truth.get("code") is not None:
            row["fold"] += " %d chips, code has %d" % (
                len(rep["_chips"]),
                len(truth["code"]),
            )
        row["bm_L"] = ",".join(str(r["L"]) for r in rep["lfsr"])
    pc = rep.get("pcps", {})
    want = truth.get("label")
    row["pcps"] = []
    for fam, v in pc.items():
        top = v["top"]
        hit = [
            b for b in top if want and b["label"].split("/")[0] == want.split("/")[0]
        ]
        if hit:
            b = hit[0]
            wrong = [w for w in top if w is not b]
            row["pcps"].append(
                "%s: %s %+.1fdB (next %+.1f) %s chip0 err %.2f C/N0 %.1f %.0fs"
                % (
                    fam,
                    b["label"],
                    b["margin_db"],
                    wrong[0]["margin_db"] if wrong else float("nan"),
                    "ID" if v["identified"] else "no-ID",
                    ((b["chip0"] - truth["chip0"] + truth["N"] / 2) % truth["N"])
                    - truth["N"] / 2
                    if "chip0" in truth
                    else float("nan"),
                    b["cn0_dbhz"],
                    v["seconds"],
                )
            )
        else:
            row["pcps"].append(
                "%s: best %s %+.1fdB %s, %.1f dB over the next: %s %.0fs"
                % (
                    fam,
                    top[0]["label"],
                    top[0]["margin_db"],
                    "DET" if top[0]["detected"] else "-",
                    v["gap_db"],
                    "IDENTIFIED" if v["identified"] else "not identified",
                    v["seconds"],
                )
            )
    if "verify" in rep:
        v = rep["verify"]
        row["verify"] = "rc_rx %+.2fHz codeDop %+.1fHz (carrier %+.1f) desp %+.2fHz" % (
            v["chip_rate_rx"] - truth["rc"] * (1 + fd_mid / f_rf),
            v["code_doppler"] - fd_mid,
            v["carrier_doppler"] - fd_mid,
            v["despread_doppler"] - fd_mid,
        )
    return row, rep


def cmd_validate(a):
    fs, f_rf = 25e6, gs.LOBES["b3"][0]
    rc, N = gcodes.B3I_RC, gcodes.B3I_N
    prn = 23
    prn_code = gcodes.b3i_prn([prn])[0]
    g1, m2, _ = gcodes.b3i_structure()
    gold_code = gcodes.gold_codes(g1, m2, [4321], N)[0]
    rnd_code = np.random.default_rng(7).choice(np.array([-1, 1], np.int8), N)
    dur = 2.0
    base = dict(dopp=12e3, rate=-80.0, chip0=1234.5, rc=rc, N=N)
    kw = dict(t_blind=0.5, n_blocks=200)
    snrs = [20.0, 8.1, 1.6, -6.0, -15.0, -20.0, -25.0] if not a.quick else [8.1, -25.0]
    cases = []
    for s in snrs:
        cases.append(
            (
                "PRN%d %+.1fdB" % (prn, s),
                prn_code,
                s,
                "b3i",
                dict(label="PRN%d" % prn),
                {},
            )
        )
    if not a.quick:
        cases.append(
            (
                "gold g2ph4321 +1.6dB",
                gold_code,
                1.6,
                "b3i,b3i_all",
                dict(label="g2ph4321"),
                {},
            )
        )
        cases.append(
            ("random code +1.6dB", rnd_code, 1.6, "b3i", dict(label="none"), {})
        )
        cases.append(
            (
                "PRN%d +8.1dB 4-bit" % prn,
                prn_code,
                8.1,
                "b3i",
                dict(label="PRN%d" % prn),
                dict(q4=True),
            )
        )
        cases.append(
            (
                "PRN%d +1.6dB chip clock +3Hz" % prn,
                prn_code,
                1.6,
                "b3i",
                dict(label="PRN%d" % prn),
                dict(chip_offset=3.0),
            )
        )
        cases.append(("noise only", prn_code, None, "b3i", dict(label="none"), {}))
    logf = (a.out or "validate") + ".log"
    open(logf, "w").close()
    rows = []

    def want(name):
        return not a.only or a.only in name

    for name, code, snr, fam, tr, sk in cases:
        if not want(name):
            continue
        t = time.time()
        x = synth(
            fs, dur, code, rc, f_rf, snr, seed=zlib.crc32(name.encode()) % 1000, **sk
        )
        tg = time.time() - t
        truth = dict(base, code=code, **tr)
        if snr is None:
            truth = dict(label="none", rc=rc, N=N)
        # no carrier on noise: the full Doppler grid, kept to 100 periods
        kwc = dict(kw, n_blocks=100) if snr is None else kw
        row, rep = run_case(name, x, fs, f_rf, "b3", fam, truth, kwc, logf)
        row["snr"] = snr
        row["X"] = band_ratio(snr, fs, rc) if snr is not None else 1.0
        row["cn0"] = snr + 10 * np.log10(fs) if snr is not None else float("nan")
        row["gen"] = tg
        rows.append(row)
        print_row(row)
        del x
    # weak signal, no carrier: the full Doppler / rate grid over the 63 PRNs
    if not a.quick:
        for s in (-30.0, -40.0):
            name = "PRN%d %+.1fdB full grid" % (prn, s)
            if not want(name):
                continue
            x = synth(fs, dur, prn_code, rc, f_rf, s, seed=11)
            truth = dict(base, label="PRN%d" % prn)
            # the blind carrier would find it at these SNRs; force the grid by hiding the carrier
            row, rep = run_case_grid(name, x, fs, f_rf, truth, logf)
            row["snr"], row["X"], row["cn0"], row["gen"] = (
                s,
                band_ratio(s, fs, rc),
                s + 10 * np.log10(fs),
                0.0,
            )
            rows.append(row)
            print_row(row)
            del x
        # the 1202 lobe: a B1I-family code at 2.046 Mcps
        fs_l3 = 8e6
        b1 = gcodes.b1i_prn([7])[0]
        for s in (1.6, -20.0):
            name = "1202 B1I PRN7 %+.1fdB" % s
            if not want(name):
                continue
            x = synth(
                fs_l3,
                dur,
                b1,
                gcodes.B1I_RC,
                gs.LOBES["l3"][0],
                s,
                dopp=-5e3,
                rate=-120.0,
                chip0=321.25,
                seed=5,
            )
            truth = dict(
                dopp=-5e3,
                rate=-120.0,
                chip0=321.25,
                rc=gcodes.B1I_RC,
                N=gcodes.B1I_N,
                label="PRN7",
                code=b1,
            )
            row, rep = run_case(
                name, x, fs_l3, gs.LOBES["l3"][0], "l3", "b1i,b1i_all", truth, kw, logf
            )
            row["snr"], row["X"], row["cn0"], row["gen"] = (
                s,
                float("nan"),
                s + 10 * np.log10(fs_l3),
                0.0,
            )
            rows.append(row)
            print_row(row)
    # channelised input: 128 x 195.3125 kHz conjugated channels, stitched back
    if not a.quick and want("PRN%d +1.6dB channelised" % prn):
        x = synth(fs, dur, prn_code, rc, f_rf, 1.6, seed=3)
        ch = channelise(x, 128)
        del x
        f_ch = f_rf + (np.arange(128) - 64) * fs / 128
        xs, fss, fcs = gs.channels_to_baseband(ch, f_ch, fs / 128, conj=True)
        name = "PRN%d +1.6dB channelised" % prn
        row, rep = run_case(
            name,
            xs,
            fss,
            fcs,
            "b3",
            "b3i",
            # the 4-tap PFB delays by 2 n_chan - 0.5 input samples
            dict(
                base, label="PRN%d" % prn, code=prn_code, chip0=1234.5 + 255.5 / fs * rc
            ),
            kw,
            logf,
        )
        row["snr"], row["X"], row["cn0"], row["gen"] = (
            1.6,
            band_ratio(1.6, fs, rc),
            1.6 + 10 * np.log10(fs),
            0.0,
        )
        rows.append(row)
        print_row(row)
    if a.out:
        with open(a.out + ".json", "w") as f:
            json.dump(rows, f, indent=1, default=gs._jsonable)


def run_case_grid(name, x, fs, fc, truth, logf):
    """The acquisition alone over the full blind grid (+-30 kHz, the rate grid), 63 PRNs."""
    f_rf, rc = gs.LOBES["b3"]
    t = time.time()
    y, fsy = gs.extract_lobe(x, fs, fc, f_rf, rc, 2)
    codes, labels, rcf, N = gcodes.family("b3i")
    T = N / rcf
    nb = 100
    span = nb * T
    step = gs.blind_doppler_step(T, span, fsy, f_rf)
    dop = gs.doppler_grid(-30e3, 30e3, T, step)
    rates = gs.rate_grid(-150.0, 150.0, span, fsy, f_rf)
    r = gs.pcps(y, fsy, codes, rcf, f_rf, dop, rates, nb, labels=labels)
    wall = time.time() - t
    top = r["results"][:3]
    b = [v for v in r["results"] if v["label"] == truth["label"]][0]
    wrong = [v for v in r["results"] if v["label"] != truth["label"]][0]
    with open(logf, "a") as f:
        f.write(
            "=== %s grid: %d dopplers (step %.0f Hz) x %d rates x %d periods, %.0f s\n"
            % (name, len(dop), step, len(rates), nb, wall)
        )
        for v in top:
            f.write(
                "   %s pmr %.3f margin %+.1f dop %+.0f rate %+.0f chip0 %.2f\n"
                % (
                    v["label"],
                    v["pmr"],
                    v["margin_db"],
                    v["doppler"],
                    v["rate"],
                    v["chip0"],
                )
            )
    row = dict(
        case=name, wall=wall, chiprate="-", period="-", carrier="(grid)", fold="-"
    )
    row["pcps"] = [
        "b3i grid %dx%dx%d: %s %+.1fdB (next %s %+.1f) dop %+.0f rate %+.0f chip0 err %.2f %s %.0fs"
        % (
            len(dop),
            len(rates),
            nb,
            b["label"],
            b["margin_db"],
            wrong["label"],
            wrong["margin_db"],
            b["doppler"] - truth["dopp"],
            b["rate"] - truth["rate"],
            ((b["chip0"] - truth["chip0"] + N / 2) % N) - N / 2,
            "DET" if b["detected"] else "-",
            wall,
        )
    ]
    return row, r


def print_row(r):
    print(
        "%-28s SNR %6s dB  X %6.2f  C/N0 %5.1f | chiprate %s | period %s | carrier %s | fold %s | BM %s"
        % (
            r["case"],
            "%+.1f" % r["snr"] if r["snr"] is not None else "-",
            r["X"],
            r["cn0"],
            r["chiprate"],
            r["period"],
            r["carrier"],
            r["fold"],
            r.get("bm_L", "-"),
        )
    )
    for p in r["pcps"]:
        print("      pcps %s" % p)
    if "verify" in r:
        print("      verify %s" % r["verify"])
    print(
        "      wall %.0f s (generate %.0f s)" % (r["wall"], r.get("gen", 0)), flush=True
    )


def cmd_make(a):
    f_rf, rc = gs.LOBES[a.lobe]
    if a.lobe == "b3":
        code = gcodes.b3i_prn([a.prn])[0]
    else:
        code = gcodes.b1i_prn([a.prn])[0]
    x = synth(
        a.fs,
        a.dur,
        code,
        rc,
        f_rf,
        a.snr,
        dopp=a.dopp,
        rate=a.rate,
        q4=a.q4,
        seed=a.seed,
    )
    np.savez(a.output, x=x, fs=a.fs, fc=f_rf)
    print(
        "wrote %s: %d samples, PRN %d, SNR %+.1f dB, Doppler %+.0f Hz, rate %+.1f Hz/s"
        % (a.output, len(x), a.prn, a.snr, a.dopp, a.rate)
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="mode", required=True)
    v = sp.add_parser("validate")
    v.add_argument("--quick", action="store_true")
    v.add_argument("--only", help="run the cases whose name contains this")
    v.add_argument("--out", default="validate")
    m = sp.add_parser("make")
    m.add_argument("output")
    m.add_argument("--lobe", choices=sorted(gs.LOBES), default="b3")
    m.add_argument("--prn", type=int, default=23)
    m.add_argument("--snr", type=float, default=1.6)
    m.add_argument("--fs", type=float, default=25e6)
    m.add_argument("--dur", type=float, default=2.0)
    m.add_argument("--dopp", type=float, default=12e3)
    m.add_argument("--rate", type=float, default=-80.0)
    m.add_argument("--q4", action="store_true")
    m.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    {"validate": cmd_validate, "make": cmd_make}[a.mode](a)


if __name__ == "__main__":
    main()
