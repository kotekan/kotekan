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
    """The admission's evidence must not be prompt power alone.

    The gate exists to rescue a satellite whose taps are off the peak, and prompt power is
    suppressed by exactly that offset, so a prompt-power bar refuses the satellites it is
    meant to rescue; every tap ratio has the same bias. The evidence is the combiner's
    deep fold (deep_snr against deep_margin x deep_floor), a coherent fold of the same prompt
    whose ~10x integration gain keeps a satellite a fraction of a chip off-peak detectable;
    prompt power above the probe bar is accepted too.
    """
    print("admit-displaced: the evidence must not be prompt power alone")

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
        "and the admission is ATTRIBUTABLE to the deep fold, not to prompt power",
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


def taps(off, ped, spacing, p_pow=20.0):
    """(E, P, L) of a triangular correlation with the taps `spacing` either side of Prompt,
    the peak `off` chips late and a noise pedestal `ped`, scaled to prompt power p_pow."""
    acf = lambda x: max(0.0, 1.0 - abs(x)) ** 2
    e, p, l = acf(off + spacing) + ped, acf(off) + ped, acf(off - spacing) + ped
    return e * p_pow / p, p_pow, l * p_pow / p


def taps_row(e, p, l):
    return row(2.0 * p / (e + l), (e - l) / (e + l), p)


def test_spacing():
    """The displaced admission fits at the chain's tap spacing (--dll-spacing).

    gps_l2c and glo_l2oc run 0.4-chip taps. Read at 0.5, a row 0.30 chips late on a 0.25
    pedestal fits pedestal 0.31 (refused as weak) and one 0.65 chips late fits 0.56 chips
    (admitted inside the pull-in bar). Both arms (fleet_dll, fleet_dll_comb) and their callers
    in the broker carry the spacing to apply_presence.
    """
    print("admit-displaced: the fit runs at the chain's dll_spacing")
    import ast
    import os
    from gnss_broker import combdll
    from gnss_broker import fleet as _fleet

    near = taps(0.30, 0.25, 0.4)  # admitted at 0.4: inside both bars
    far = taps(0.65, 0.05, 0.4)  # refused at 0.4: past presence_disp_off_max

    def judged(v20, v21, what):
        check(
            v20["present"] is True
            and v20["present_gate"] == "q+p:probes+disp"
            and abs(v20["off_chips"] - 0.30) < 0.01
            and abs(v20["pedestal"] - 0.25) < 0.01,
            "%s: 0.30 chips late on a 0.25 pedestal fits as such and is admitted "
            "(off %+.3f ped %.3f)"
            % (what, v20.get("off_chips", 0), v20.get("pedestal", 0)),
        )
        check(
            v21["present"] is False and abs(v21.get("off_chips", 0) - 0.65) < 0.01,
            "%s: 0.65 chips late fits as such and is refused (off %+.3f)"
            % (what, v21.get("off_chips", 0)),
        )

    out = fleet(**{"20": taps_row(*near), "21": taps_row(*far)})
    try:
        apply_presence(
            out,
            k_sigma=3.0,
            q_fallback=2.2,
            probe_prns=PROBES,
            admit_displaced=ADMIT,
            dll_spacing=0.4,
        )
        judged(out[20], out[21], "apply_presence at 0.4")
    except TypeError as e:
        check(False, "apply_presence takes dll_spacing: %s" % e)

    # The comb arm, fed its taps directly: each row's (q, disc, p_pow) back to (E, P, L).
    per_prn = {}
    for p, v in fleet(**{"20": taps_row(*near), "21": taps_row(*far)}).items():
        el = 2.0 * v["p_pow"] / v["q"]
        per_prn[p] = {
            "e": 0.5 * (1.0 + v["disc"]) * el,
            "p": v["p_pow"],
            "l": 0.5 * (1.0 - v["disc"]) * el,
            "n_rec": 1,
            "hop": 1000,
            "n_inst": 4,
            "n_chan": 28,
        }
    try:
        out = combdll.fleet_dll_comb(
            None,
            "gps_l2c",
            probe_prns=PROBES,
            taps_src=lambda chain, prns: per_prn,
            admit_displaced=ADMIT,
            dll_spacing=0.4,
        )
        judged(out[20], out[21], "fleet_dll_comb at 0.4")
    except TypeError as e:
        check(False, "fleet_dll_comb takes dll_spacing: %s" % e)

    # The polled arm, through its REST poll.
    polled = [
        {
            "prn": p,
            "pow_hop": 1000,
            "e_pow": t["e"],
            "p_pow": t["p"],
            "l_pow": t["l"],
            "n_chan": 7.0,
        }
        for p, t in per_prn.items()
    ]
    real_get = _fleet._get
    _fleet._get = lambda url, timeout=5.0: polled
    try:
        out = _fleet.fleet_dll(
            ["http://cx19:12048/gnss0"],
            hop_window=4096,
            min_instances=1,
            k_sigma=3.0,
            q_fallback=2.2,
            probe_prns=PROBES,
            admit_displaced=ADMIT,
            hop_hist={},
            now=0.0,
            dll_spacing=0.4,
        )
        judged(out[20], out[21], "fleet_dll at 0.4")
    except TypeError as e:
        check(False, "fleet_dll takes dll_spacing: %s" % e)
    finally:
        _fleet._get = real_get

    # Read at the 0.5 default, the same rows invert: the spacing is what decides them.
    out = verdict(fleet(**{"20": taps_row(*near), "21": taps_row(*far)}), ADMIT)
    check(
        out[20]["present"] is False and out[21]["present"] is True,
        "at the 0.5 default the near row is refused (ped %.3f) and the far one admitted "
        "(off %+.3f)" % (out[20]["pedestal"], out[21]["off_chips"]),
    )

    # The broker's own calls of both arms pass the chain's --dll-spacing.
    here = os.path.dirname(os.path.abspath(_fleet.__file__))
    calls = []
    for name in sorted(os.listdir(here)):
        if not name.endswith(".py") or name.startswith("test_"):
            continue
        with open(os.path.join(here, name)) as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname not in ("fleet_dll", "fleet_dll_comb"):
                continue
            kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
            calls.append((name, fname, kw.get("dll_spacing")))
    check(
        len(calls) >= 2 and all(sp == "ctx.args.dll_spacing" for _, _, sp in calls),
        "every broker call of fleet_dll / fleet_dll_comb passes dll_spacing="
        "ctx.args.dll_spacing: %s" % calls,
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
    test_spacing()
    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
