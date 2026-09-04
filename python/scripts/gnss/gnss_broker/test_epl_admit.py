"""The E3 fix, presence half: the displaced-row admission, on the numbers that motivated it.

    python3 -m gnss_broker.test_epl_admit

WHY THIS EXISTS RATHER THAN A FIXTURE. The digest gate replays recorded POST streams, and no
recorded transcript carries --presence-admit-displaced -- so the admission branch would never
execute in replay and never be compared to anything (the C_LIGHT blind spot). The decisive
cases are also SHAPES, not cycles: the E3 collapse row, the bright-flat row, the beyond-range
row. They are asserted here directly, against apply_presence itself -- the one shared policy
both DLL arms judge with -- not against a paraphrase of it.

THE DEGENERACY BEING BROKEN. q ~ 1 means "off-peak" OR "weak", and the q bar reads both as
weak. The (E/P, L/P) decomposition separates them exactly: a displaced row's power went to a
NEIGHBOURING TAP (pedestal low), a weak row's went to NOISE (pedestal high). The dangerous
mistake in each direction:
  * refuse the displaced-but-strong row -> the E3 latch (an offset suppresses q, so the
    fault disables its own cure: 12 minutes of outage from a 60 s fade);
  * admit the bright-flat row -> trimming on noise, the 2026-08-14 bds_b2a PRN 32/35 failure
    that the q bar exists to prevent.
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
    out = {90: row(0.95, +0.02, 1.0), 91: row(1.00, -0.03, 1.1), 92: row(1.05, +0.01, 0.9),
           7: row(3.40, -0.05, 50.0), 12: row(3.10, +0.08, 30.0)}
    out.update({int(k): v for k, v in extra.items()})
    return out


def verdict(out, admit):
    apply_presence(out, k_sigma=3.0, q_fallback=2.2, probe_prns=PROBES,
                   admit_displaced=admit)
    return out


def coh(deep_snr, deep_floor):
    return {"deep_snr": deep_snr, "deep_floor": deep_floor}


def test_blind_evidence():
    """The admission's EVIDENCE must not be an on-peak statistic.

    ⚠️ THE BUG THIS PINS (KV, 2026-08-27). This gate exists to rescue the satellite whose
    taps are off the peak, and it required prompt power above the probe floor to believe the
    satellite was there at all -- but PROMPT POWER IS SUPPRESSED BY EXACTLY THAT OFFSET. It
    is q's disease one step along, and it was the third time the same kind of statistic was
    swapped for another of its kind (q -> prompt -> prompt), against a comment in this very
    file saying "do not fix this a third time with another tap ratio".

    Measured on sky the day it was found: 7 satellites seen by an OFFSET-BLIND detector at
    deep_snr 18-56 and C/N0 up to 26.8 dB-Hz, prompt power 0.5-2.7x noise. The prompt bar
    admitted TWO. The other five failed on p precisely because they were displaced.
    """
    print("admit-displaced: the evidence must be OFFSET-BLIND")

    # bds_b2a PRN 20, from the 2026-08-27 12:4x poll: displaced, prompt DIM (2.65x, under the
    # 3x probe bar), but the deep fold -- which re-searches code phase -- sees it at 55 sigma
    # against a floor of 2.67.
    dim_but_detected = row(0.90, +0.65, 2.4)
    dim_but_detected["coh_row"] = coh(55.0, 2.67)
    out = verdict(fleet(**{"20": dim_but_detected}), ADMIT)
    v = out[20]
    check(v["present"] is True,
          "a DIM but strongly-DETECTED displaced row is admitted (prompt 2.65x would refuse "
          "it; deep_snr 55 vs floor 2.67 does not)")
    check(v["present_gate"] == "q+deep:probes+disp",
          "and the admission is ATTRIBUTABLE to the blind detector, not to prompt power")

    # The same row with NO deep evidence at all must still be refused -- otherwise this has
    # just widened the gate rather than changed what it asks.
    no_deep = row(0.90, +0.65, 2.4)
    out2 = verdict(fleet(**{"20": no_deep}), ADMIT)
    check(out2[20]["present"] is False,
          "the SAME dim row with no deep row is still refused -- the bar moved to a different "
          "quantity, it did not go away")

    # A centred NOISE realisation, detected or not, must stay out: the pedestal test is what
    # guards that, and loosening the evidence must not have loosened it.
    noise = row(1.02, +0.00, 2.4)      # q ~ 1, disc ~ 0 -> huge pedestal
    noise["coh_row"] = coh(55.0, 2.67)
    out3 = verdict(fleet(**{"21": noise}), ADMIT)
    check(out3[21]["present"] is False,
          "a CENTRED NOISE row is refused even with a loud deep detection -- the pedestal "
          "test still does that work (the 2026-08-14 trimming-on-noise case)")

    # Beyond the DLL's pull-in range: refused, correctly -- that one needs the search.
    far = row(0.40, +0.45, 20.0)
    far["coh_row"] = coh(55.0, 2.67)
    out4 = verdict(fleet(**{"22": far}), ADMIT)
    check(out4[22]["present"] is False and abs(out4[22]["off_chips"]) > 0.6,
          "a row BEYOND the pull-in range is refused even though it is bright AND loudly "
          "detected -- a detection is not a reason to arm a loop with no gradient to follow")

    # A bright row keeps the old path and the old label, so the A/B can separate "the deep
    # path did the work" from "it would have passed on prompt anyway".
    bright = row(0.90, -0.65, 20.0)
    out5 = verdict(fleet(**{"23": bright}), ADMIT)
    check(out5[23]["present"] is True and out5[23]["present_gate"] == "q+p:probes+disp",
          "a BRIGHT displaced row still admits on prompt, labelled 'q+p:probes+disp'")

    # A weak deep detection is not evidence.
    weak = row(0.90, +0.65, 2.4)
    weak["coh_row"] = coh(4.0, 2.67)   # 1.5x floor, under the 3x margin
    out6 = verdict(fleet(**{"24": weak}), ADMIT)
    check(out6[24]["present"] is False,
          "a deep detection UNDER the margin (1.5x floor) is not evidence")


def main():
    # ---- 0. the algebra itself, on the banked E3 numbers -------------------------------
    off, ped = epl_decompose(1.16, -0.70)
    check(abs(off - 0.337) < 0.01 and abs(ped - 0.119) < 0.01,
          "E3's collapse row (q 1.16, disc -0.70) decodes to +0.34 chips late, pedestal 0.12")
    off, ped = epl_decompose(1.0, 0.0)
    check(off == 0.0 and ped == float("inf"),
          "E = P = L decodes to pedestal INF -- the no-information answer, never a small offset")

    # ---- 1. flag OFF: the E3 row is refused, exactly as it was on the day --------------
    out = verdict(fleet(), None)  # baseline sanity first
    check(out[7]["present"] and out[12]["present"],
          "healthy satellites are present under the probe-anchored gate")
    check(not any(out[p]["present"] for p in PROBES), "probes are not present")

    e3 = row(1.16, -0.70, 20.0)          # the collapse row: displaced 0.34 chips, BRIGHT
    out = verdict(fleet() | {3: dict(e3)}, None)
    check(not out[3]["present"],
          "flag OFF: the E3 row fails presence (this is the latch, reproduced)")

    # ---- 2. flag ON: the E3 row is admitted, and labelled ------------------------------
    out = verdict(fleet() | {3: dict(e3)}, ADMIT)
    check(out[3]["present"] and out[3]["present_gate"] == "q+p:probes+disp",
          "flag ON: the E3 row is admitted, gate = 'q+p:probes+disp'")
    check(abs(out[3]["off_chips"] - 0.337) < 0.01 and out[3]["pedestal"] < 0.3,
          "... and the row carries the decomposition it was admitted on")
    check(out[7]["present_gate"] == "q+p:probes",
          "... while an on-peak row's verdict and gate label are untouched")

    # ---- 3. the OTHER direction: bright-flat is still refused --------------------------
    # E = P = L at 20x the floor is a smeared or noise-shaped response with no peak under
    # ANY tap. Admitting it would re-create the 2026-08-14 trimming-on-noise failure with
    # extra steps -- the pedestal (inf) is what refuses it.
    out = verdict(fleet() | {5: row(1.00, 0.00, 20.0)}, ADMIT)
    check(not out[5]["present"],
          "bright but FLAT (E=P=L at 20x): refused -- pedestal inf, no peak to pull in")

    # ---- 4. displaced but DIM: refused by the same p bar as everyone -------------------
    out = verdict(fleet() | {6: row(1.16, -0.70, 1.5)}, ADMIT)
    check(not out[6]["present"],
          "displaced shape at 1.5x the floor: refused (the p bar is not waived)")

    # ---- 5. beyond pull-in range: refused ----------------------------------------------
    # 0.8 chips late: E has left the correlation triangle, disc is sign-only. From the
    # forward model: q and disc for offset 0.8 at pedestal 0.05.
    def qd(off, ped):
        acf = lambda x: max(0.0, 1.0 - abs(x)) ** 2
        e, pp, l = acf(off + 0.5) + ped, acf(off) + ped, acf(off - 0.5) + ped
        return 2.0 * pp / (e + l), (e - l) / (e + l)
    q8, d8 = qd(0.80, 0.05)
    out = verdict(fleet() | {8: row(q8, d8, 20.0)}, ADMIT)
    check(not out[8]["present"],
          "0.8 chips out (q %.2f disc %+.2f): refused -- beyond --presence-disp-off-max" % (q8, d8))

    # ---- 6. WITHOUT probes the admission cannot fire at all ----------------------------
    # The p bar without probes is a peer competition; admission on top of one would admit
    # peers' noise. The branch is inside the probe-anchored gate by construction.
    out = {7: row(3.4, -0.05, 50.0), 12: row(3.1, 0.08, 30.0), 3: dict(e3)}
    apply_presence(out, k_sigma=3.0, q_fallback=2.2, probe_prns=None, admit_displaced=ADMIT)
    check(out[3].get("present_gate") != "q+p:probes+disp",
          "no probes: the admission branch is unreachable (gate stays %r)"
          % out[3].get("present_gate"))

    # ---- 7. flag OFF leaves every verdict AND every key identical ----------------------
    base = fleet() | {3: dict(e3)}
    a = verdict({k: dict(v) for k, v in base.items()}, None)
    b = verdict({k: dict(v) for k, v in base.items()}, None)
    check(a == b, "determinism sanity")
    c = verdict({k: dict(v) for k, v in base.items()}, ADMIT)
    same = all(a[k]["present"] == c[k]["present"] for k in a if k != 3)
    check(same, "the flag changes PRN 3's verdict and nobody else's")

    # ---- 8. NO PROBE ANCHOR -> REFUSE. The peer branch is DELETED, not defaulted off ----
    # (KV, 2026-08-27: "peer comparisons can never tell us about a given signal".) The
    # fallback passed about half the population by construction; with too few probes the
    # honest answer is "I cannot tell", and it must be SAID. There is no longer a flag: the
    # refusal is the only path, because there is nothing left to fall back to.
    thin = {90: row(0.95, 0.02, 1.0), 91: row(1.00, -0.03, 1.1),      # only TWO probes
            7: row(3.40, -0.05, 50.0), 12: row(3.10, 0.08, 30.0),
            3: row(2.90, 0.02, 25.0), 5: row(1.20, -0.30, 8.0)}
    b = {k: dict(v) for k, v in thin.items()}
    apply_presence(b, k_sigma=3.0, q_fallback=2.2, probe_prns={90, 91})
    check(all(v["present_gate"] == "UNANCHORED" for v in b.values()),
          "too few probes: every row reads UNANCHORED, with no flag to turn it off")
    check(not any(v["present"] for v in b.values()),
          "... and NOBODY is admitted (no presence -> no arming -> no trimming)")
    check(b[7]["n_probe_q"] == 2, "... and the row records how many probes there were")
    # ⚠️ A REFUSAL MUST STILL EMIT A WELL-FORMED ROW. With too few rows to characterise a
    # population _floor returns (None, None, None); publishing that raw q_floor took bds_b2a
    # down with TypeError('must be real number, not NoneType') the moment a downstream log
    # line formatted it with %.2f. "I cannot judge" is a verdict, not a licence to emit None
    # into fields every consumer already treats as numbers.
    thin4 = {90: row(0.9, 0.0, 1.0), 91: row(1.0, 0.0, 1.1),
             7: row(3.4, 0.0, 50.0), 12: row(3.1, 0.0, 30.0)}     # 4 rows: _floor gives None
    d = {k: dict(v) for k, v in thin4.items()}
    apply_presence(d, k_sigma=3.0, q_fallback=2.2, probe_prns={90, 91})
    check(all(isinstance(v["q_floor"], float) for v in d.values()),
          "an UNANCHORED row still carries a NUMERIC q_floor (%r) -- the None that killed a "
          "chain" % d[7]["q_floor"])
    try:
        "%.2f %.2f" % (d[7]["q_floor"], d[7]["p_floor"] or 0.0)
        fmt_ok = True
    except TypeError:
        fmt_ok = False
    check(fmt_ok, "... and formats with %.2f exactly as the shipped log line does")
    # and it must NOT fire when the probes ARE sufficient
    c = fleet() | {3: row(2.90, 0.02, 25.0)}
    apply_presence(c, k_sigma=3.0, q_fallback=2.2, probe_prns=PROBES)
    check(all(v["present_gate"] != "UNANCHORED" for v in c.values()),
          "with 3 probes the refusal does not fire and the normal gate runs")

    # ⚠️ THE PEER BRANCH IS GONE FROM THE SOURCE, NOT JUST FROM THE CALL PATH. A flag can be
    # re-set and a default can be flipped back; this asserts the code itself no longer builds
    # a bar out of the tracked population. If someone reintroduces it, this goes red before
    # anything reaches the sky.
    # ⚠️ STRIP COMMENTS FIRST. The deletion is explained in a comment that necessarily
    # QUOTES the deleted expression, so a naive substring search finds the documentation and
    # reports the bug it is documenting. Ban the executable line, not the prose about it.
    import inspect
    from gnss_broker import fleet as _fleet
    src = "\n".join(ln.split("#")[0] for ln in
                    inspect.getsource(_fleet.apply_presence).splitlines())
    for banned in ('_floor([v["q"] for v in out.values()]',
                   '_floor([v["p_pow"] for v in out.values()]'):
        check(banned not in src,
              "apply_presence contains no peer-population floor: %s" % banned[:34])
    check("peers:" not in src,
          "... and no row can be labelled p_floor_src='peers:N' any more")

    test_blind_evidence()
    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
