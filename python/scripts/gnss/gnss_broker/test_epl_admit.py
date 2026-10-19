"""Displaced-row admission (--presence-admit-displaced) and the probe anchor, in apply_presence.

    python3 -m gnss_broker.test_epl_admit

The cases are row shapes, not cycles (the collapse row, the bright-flat row, the
beyond-range row), so they are asserted directly against fleet.apply_presence, the one
policy both DLL arms judge with, rather than through a replay.

The degeneracy being broken: q ~ 1 means "off-peak" or "weak", and the q bar reads both as
weak. The (E/P, L/P) decomposition separates them: a displaced row's power went to a
neighbouring tap (low pedestal), a weak row's went to noise (high pedestal). The mistake in
each direction:
  * refuse the displaced-but-strong row -> a latch: the offset suppresses q, so the fault
    disables its own cure;
  * admit the bright-flat row -> trimming on noise, which the q bar exists to prevent.
Both directions are pinned below.

@author Keith Vanderlinde
"""

import sys

from gnss_broker.fleet import apply_presence, epl_decompose


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def row(q, disc, p_pow):
    return {"q": q, "disc": disc, "p_pow": p_pow}


PROBES = {90, 91, 92}
ADMIT = {"pedestal_max": 0.3, "off_max_chips": 0.6, "deep_margin": 3.0}


def fleet(**extra):
    """Probes at the floor + two healthy satellites, then the case under test."""
    out = {
        90: row(0.95, +0.02, 1.0),
        91: row(1.00, -0.03, 1.1),
        92: row(1.05, +0.01, 0.9),
        7: row(3.40, -0.05, 50.0),
        12: row(3.10, +0.08, 30.0),
    }
    out.update({int(k): v for k, v in extra.items()})
    return out


def verdict(out, admit):
    apply_presence(
        out, k_sigma=3.0, q_fallback=2.2, probe_prns=PROBES, admit_displaced=admit
    )
    return out


def coh(deep_snr, deep_floor):
    return {"deep_snr": deep_snr, "deep_floor": deep_floor}


def test_blind_evidence():
    """The admission's evidence must not be an on-peak statistic.

    The gate exists to rescue a satellite whose taps are off the peak, and prompt power is
    suppressed by exactly that offset, so a prompt-power bar refuses the satellites it is
    meant to rescue; every tap ratio has the same bias. The evidence is the combiner's
    deep fold (deep_snr against deep_margin x deep_floor), which re-searches code phase and so
    sees the satellite wherever the tap sits; prompt power above the probe bar is accepted too.
    """
    print("admit-displaced: the evidence must be offset-blind")

    # Displaced and dim: prompt 2.4x the probe median, under the 3x probe bar, but the deep
    # fold sees it at deep_snr 55 against a floor of 2.67.
    dim_but_detected = row(0.90, +0.65, 2.4)
    dim_but_detected["coh_row"] = coh(55.0, 2.67)
    out = verdict(fleet(**{"20": dim_but_detected}), ADMIT)
    v = out[20]
    check(
        v["present"] is True,
        "a dim but strongly detected displaced row is admitted (prompt 2.4x would refuse "
        "it; deep_snr 55 vs floor 2.67 does not)",
    )
    check(
        v["present_gate"] == "q+deep:probes+disp",
        "and the admission is ATTRIBUTABLE to the blind detector, not to prompt power",
    )

    # The same row with no deep evidence must still be refused; otherwise the gate has only
    # been widened, not changed in what it asks.
    no_deep = row(0.90, +0.65, 2.4)
    out2 = verdict(fleet(**{"20": no_deep}), ADMIT)
    check(
        out2[20]["present"] is False,
        "the same dim row with no deep row is still refused -- the bar moved to a different "
        "quantity, it did not go away",
    )

    # A centred noise realisation stays out, detected or not: the pedestal test guards that,
    # whatever the evidence.
    noise = row(1.02, +0.00, 2.4)  # q ~ 1, disc ~ 0 -> huge pedestal
    noise["coh_row"] = coh(55.0, 2.67)
    out3 = verdict(fleet(**{"21": noise}), ADMIT)
    check(
        out3[21]["present"] is False,
        "a centred noise row is refused even with a loud deep detection -- the pedestal "
        "test still does that work (the trimming-on-noise case)",
    )

    # Beyond the DLL's pull-in range: refused; that one needs the search, not the loop.
    far = row(0.40, +0.45, 20.0)
    far["coh_row"] = coh(55.0, 2.67)
    out4 = verdict(fleet(**{"22": far}), ADMIT)
    check(
        out4[22]["present"] is False and abs(out4[22]["off_chips"]) > 0.6,
        "a row beyond the pull-in range is refused even though it is bright and loudly "
        "detected -- a detection is not a reason to arm a loop with no gradient to follow",
    )

    # A bright displaced row admits on prompt and is labelled q+p, so the gate label tells
    # whether the deep fold or prompt power carried each admission.
    bright = row(0.90, -0.65, 20.0)
    out5 = verdict(fleet(**{"23": bright}), ADMIT)
    check(
        out5[23]["present"] is True and out5[23]["present_gate"] == "q+p:probes+disp",
        "a BRIGHT displaced row still admits on prompt, labelled 'q+p:probes+disp'",
    )

    # A weak deep detection is not evidence.
    weak = row(0.90, +0.65, 2.4)
    weak["coh_row"] = coh(4.0, 2.67)  # 1.5x floor, under the 3x margin
    out6 = verdict(fleet(**{"24": weak}), ADMIT)
    check(
        out6[24]["present"] is False,
        "a deep detection UNDER the margin (1.5x floor) is not evidence",
    )


def main():
    # ---- 0. epl_decompose on the collapse row and on E = P = L --------------------------
    off, ped = epl_decompose(1.16, -0.70)
    check(
        abs(off - 0.337) < 0.01 and abs(ped - 0.119) < 0.01,
        "the collapse row (q 1.16, disc -0.70) decodes to +0.34 chips late, pedestal 0.12",
    )
    off, ped = epl_decompose(1.0, 0.0)
    check(
        off == 0.0 and ped == float("inf"),
        "E = P = L decodes to pedestal INF -- the no-information answer, never a small offset",
    )

    # ---- 1. admission off: the collapse row is refused (the latch) ----------------------
    out = verdict(fleet(), None)  # baseline sanity first
    check(
        out[7]["present"] and out[12]["present"],
        "healthy satellites are present under the probe-anchored gate",
    )
    check(not any(out[p]["present"] for p in PROBES), "probes are not present")

    e3 = row(1.16, -0.70, 20.0)  # the collapse row: displaced 0.34 chips, bright
    out = verdict(fleet() | {3: dict(e3)}, None)
    check(
        not out[3]["present"], "flag OFF: the collapse row fails presence (the latch)",
    )

    # ---- 2. admission on: the collapse row is admitted, and labelled --------------------
    out = verdict(fleet() | {3: dict(e3)}, ADMIT)
    check(
        out[3]["present"] and out[3]["present_gate"] == "q+p:probes+disp",
        "flag ON: the collapse row is admitted, gate = 'q+p:probes+disp'",
    )
    check(
        abs(out[3]["off_chips"] - 0.337) < 0.01 and out[3]["pedestal"] < 0.3,
        "... and the row carries the decomposition it was admitted on",
    )
    check(
        out[7]["present_gate"] == "q+p:probes",
        "... while an on-peak row's verdict and gate label are untouched",
    )

    # ---- 3. the other direction: bright-flat is still refused --------------------------
    # E = P = L at 20x the floor is a smeared or noise-shaped response with no peak under any
    # tap; admitting it would trim on noise. The pedestal (inf) is what refuses it.
    out = verdict(fleet() | {5: row(1.00, 0.00, 20.0)}, ADMIT)
    check(
        not out[5]["present"],
        "bright but FLAT (E=P=L at 20x): refused -- pedestal inf, no peak to pull in",
    )

    # ---- 4. displaced, dim, no deep evidence: refused by the p bar ---------------------
    out = verdict(fleet() | {6: row(1.16, -0.70, 1.5)}, ADMIT)
    check(
        not out[6]["present"],
        "displaced shape at 1.5x the floor: refused (the p bar is not waived)",
    )

    # ---- 5. beyond pull-in range: refused ----------------------------------------------
    # 0.8 chips late: E has left the correlation triangle, disc is sign-only. From the
    # forward model: q and disc for offset 0.8 at pedestal 0.05.
    def qd(off, ped):
        acf = lambda x: max(0.0, 1.0 - abs(x)) ** 2
        e, pp, l = acf(off + 0.5) + ped, acf(off) + ped, acf(off - 0.5) + ped
        return 2.0 * pp / (e + l), (e - l) / (e + l)

    q8, d8 = qd(0.80, 0.05)
    out = verdict(fleet() | {8: row(q8, d8, 20.0)}, ADMIT)
    check(
        not out[8]["present"],
        "0.8 chips out (q %.2f disc %+.2f): refused -- beyond presence_disp_off_max"
        % (q8, d8),
    )

    # ---- 6. without probes the admission cannot fire -----------------------------------
    # With no probes every row is refused UNANCHORED before the displaced branch is reached.
    out = {7: row(3.4, -0.05, 50.0), 12: row(3.1, 0.08, 30.0), 3: dict(e3)}
    apply_presence(
        out, k_sigma=3.0, q_fallback=2.2, probe_prns=None, admit_displaced=ADMIT
    )
    check(
        out[3].get("present_gate") != "q+p:probes+disp",
        "no probes: the admission branch is unreachable (gate stays %r)"
        % out[3].get("present_gate"),
    )

    # ---- 7. the admission changes only the displaced row's verdict ---------------------
    base = fleet() | {3: dict(e3)}
    a = verdict({k: dict(v) for k, v in base.items()}, None)
    b = verdict({k: dict(v) for k, v in base.items()}, None)
    check(a == b, "determinism sanity")
    c = verdict({k: dict(v) for k, v in base.items()}, ADMIT)
    same = all(a[k]["present"] == c[k]["present"] for k in a if k != 3)
    check(same, "the flag changes PRN 3's verdict and nobody else's")

    # ---- 8. no probe anchor -> refuse ---------------------------------------------------
    # With fewer than 3 probes the noise floor is unmeasured and every row is refused
    # UNANCHORED, unconditionally. There is no bar built from the tracked population to fall
    # back to: a median-based bar over the tracked satellites passes about half of them by
    # construction, whatever the sky.
    thin = {
        90: row(0.95, 0.02, 1.0),
        91: row(1.00, -0.03, 1.1),  # only two probes
        7: row(3.40, -0.05, 50.0),
        12: row(3.10, 0.08, 30.0),
        3: row(2.90, 0.02, 25.0),
        5: row(1.20, -0.30, 8.0),
    }
    b = {k: dict(v) for k, v in thin.items()}
    apply_presence(b, k_sigma=3.0, q_fallback=2.2, probe_prns={90, 91})
    check(
        all(v["present_gate"] == "UNANCHORED" for v in b.values()),
        "too few probes: every row reads UNANCHORED, with no flag to turn it off",
    )
    check(
        not any(v["present"] for v in b.values()),
        "... and nobody is admitted (no presence -> no arming -> no trimming)",
    )
    check(b[7]["n_probe_q"] == 2, "... and the row records how many probes there were")
    # A refusal must still emit a well-formed row: consumers format q_floor with %.2f, so with
    # the noise q unmeasured it carries q_fallback, never None.
    thin4 = {
        90: row(0.9, 0.0, 1.0),
        91: row(1.0, 0.0, 1.1),
        7: row(3.4, 0.0, 50.0),
        12: row(3.1, 0.0, 30.0),
    }  # two probes: the noise q is unmeasured
    d = {k: dict(v) for k, v in thin4.items()}
    apply_presence(d, k_sigma=3.0, q_fallback=2.2, probe_prns={90, 91})
    check(
        all(isinstance(v["q_floor"], float) for v in d.values()),
        "an UNANCHORED row still carries a numeric q_floor (%r), never None"
        % d[7]["q_floor"],
    )
    try:
        "%.2f %.2f" % (d[7]["q_floor"], d[7]["p_floor"] or 0.0)
        fmt_ok = True
    except TypeError:
        fmt_ok = False
    check(fmt_ok, "... and formats with %.2f exactly as the shipped log line does")
    # and it must not fire when there are enough probes
    c = fleet() | {3: row(2.90, 0.02, 25.0)}
    apply_presence(c, k_sigma=3.0, q_fallback=2.2, probe_prns=PROBES)
    check(
        all(v["present_gate"] != "UNANCHORED" for v in c.values()),
        "with 3 probes the refusal does not fire and the normal gate runs",
    )

    # The source itself must not build a bar out of the tracked population: a flag can be
    # re-set and a default flipped back, so this checks the code, not the call path.
    # Comments are stripped first, so a comment naming the expression cannot fail the check;
    # apply_presence's docstring is not stripped.
    import inspect
    from gnss_broker import fleet as _fleet

    src = "\n".join(
        ln.split("#")[0] for ln in inspect.getsource(_fleet.apply_presence).splitlines()
    )
    for banned in (
        '_floor([v["q"] for v in out.values()]',
        '_floor([v["p_pow"] for v in out.values()]',
    ):
        check(
            banned not in src,
            "apply_presence contains no peer-population floor: %s" % banned[:34],
        )
    check(
        "peers:" not in src, "... and no row can be labelled p_floor_src='peers:N'",
    )

    test_blind_evidence()
    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
