"""The displaced admission refuses a chain event.

    python3 -m gnss_broker.test_disp_refuse

No recorded transcript carries --presence-admit-displaced, so the digest gate
(scripts/gnss/broker_equiv.py) never executes this branch; the shapes are asserted here against
apply_presence and displaced_chain_event, the rule both DLL arms (fleet_dll and fleet_dll_comb)
judge with.

The fixture rows are gal_e6 PRESENCE +displaced lines. While one satellite crossed the dish's
first null, every PRN was admitted at once, all at +0.41..+0.47 chips: N satellites do not move
by one offset in one cycle, the chain did, and admitting them re-seeds each off its peak, where
the leak-bounded DLL then holds it. A single admission (E4 alone at +0.589) is the gate doing
its job and must still pass.

@author Keith Vanderlinde
"""

import sys

from gnss_broker import fleet as _fleet
from gnss_broker.fleet import apply_presence, displaced_chain_event


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def row(q, disc, p_pow):
    return {"q": q, "disc": disc, "p_pow": p_pow}


PROBES = {90, 91, 92}


def admit(browned_out=False):
    return {
        "pedestal_max": 0.3,
        "off_max_chips": 0.6,
        "deep_margin": 3.0,
        "browned_out": browned_out,
    }


def fleet(**extra):
    """Probes at the floor + two healthy satellites, then the rows under test."""
    out = {
        90: row(0.95, +0.02, 1.0),
        91: row(1.00, -0.03, 1.1),
        92: row(1.05, +0.01, 0.9),
        7: row(3.40, -0.05, 50.0),
        12: row(3.10, +0.08, 30.0),
    }
    out.update({int(k): v for k, v in extra.items()})
    return out


def verdict(out, adm):
    """apply_presence, with the rate-limited log captured."""
    lines = []
    keep = _fleet._log_rl
    _fleet._log_rl = lambda key, msg, every_s=10.0: lines.append((key, msg))
    try:
        apply_presence(
            out, k_sigma=3.0, q_fallback=2.2, probe_prns=PROBES, admit_displaced=adm
        )
    finally:
        _fleet._log_rl = keep
    return out, lines


# ---- The rule, on the logged numbers -----------------------------------------------------

# gal_e6, the null crossing: (prn, off, ped, p/floor, via-p) from the PRESENCE +displaced line.
NULL_CROSSING = [
    (31, +0.466, 0.052, 20.6, True),
    (4, +0.410, 0.055, 20.6, True),
    (23, +0.437, 0.035, 17.6, True),
    (5, +0.419, 0.044, 10.9, True),
    (16, +0.452, 0.043, 6.6, True),
]
# gal_e6: E4 admitted alone at +0.589.
E4_ALONE = [(4, +0.589, 0.059, 8.7, True)]
# gal_e6: two at one offset (E6 -0.251, E4 -0.273), under the chain bar of three.
TWO_SAME = [(6, -0.251, 0.066, 28.1, True), (4, -0.273, 0.038, 14.6, True)]
# bds_b2a: dim satellites the offset-blind fold admits on their peak, as in most cycles. Not
# displaced, so there is no chain event in them; refusing these would empty the chain.
ON_PEAK = [
    (22, -0.002, 0.218, 0.7, False),
    (39, -0.012, 0.061, 0.3, False),
    (34, +0.009, 0.101, 0.3, False),
    (36, -0.010, 0.082, 0.3, False),
]


def test_rule():
    print(
        "displaced_chain_event: the chain bar is three PRNs within 0.1 chips, or a brownout"
    )
    refused, why = displaced_chain_event(NULL_CROSSING, False)
    check(
        sorted(a[0] for a in refused) == [4, 5, 16, 23, 31] and why == "same offset",
        "22:34:52Z: all five PRNs at +0.41..+0.47 are refused as one offset",
    )
    refused, why = displaced_chain_event(E4_ALONE, False)
    check(not refused, "23:24:35Z: E4 alone at +0.589 is admitted")
    refused, why = displaced_chain_event(TWO_SAME, False)
    check(
        not refused, "22:41:02Z: two PRNs at one offset are under the bar and admitted"
    )
    refused, why = displaced_chain_event(E4_ALONE, True)
    check(
        refused == E4_ALONE and why == "BROWNOUT open",
        "a single admission during a brownout is refused",
    )
    refused, why = displaced_chain_event(ON_PEAK, False)
    check(not refused, "four dim satellites on their peaks are not a chain event")
    refused, why = displaced_chain_event(ON_PEAK, True)
    check(
        not refused, "nor are they refused during a brownout: nothing to re-seed wrong"
    )
    refused, why = displaced_chain_event(ON_PEAK + NULL_CROSSING, False)
    check(
        sorted(a[0] for a in refused) == [4, 5, 16, 23, 31],
        "the on-peak rows beside the cluster neither join it nor shield it",
    )
    # The cluster is judged per PRN: an outlier beside a cluster stays admitted.
    mixed = NULL_CROSSING + [(9, -0.300, 0.040, 5.0, True)]
    refused, why = displaced_chain_event(mixed, False)
    check(
        sorted(a[0] for a in refused) == [4, 5, 16, 23, 31],
        "a PRN at its own offset beside the cluster is not refused with it",
    )


# ---- The gate, end to end ----------------------------------------------------------------

# Bright, displaced, inside the pull-in range: the shape the displaced gate exists to admit.
# Three identical rows decompose to one offset.
DISPLACED = (0.90, -0.65, 20.0)


def test_gate():
    print("apply_presence: the refusal reverts the row and logs once")
    out, lines = verdict(
        fleet(**{"4": row(*DISPLACED), "5": row(*DISPLACED), "16": row(*DISPLACED)}),
        admit(),
    )
    check(
        all(out[p]["present"] is False for p in (4, 5, 16)),
        "three bright rows at one offset: none is present",
    )
    check(
        all(out[p]["present_gate"] == "q+p:probes" for p in (4, 5, 16)),
        "and each row carries the q/prompt verdict, not a displaced label",
    )
    check(
        all(abs(out[p]["off_chips"] - out[4]["off_chips"]) <= 0.1 for p in (5, 16)),
        "the rows did decompose to one offset (%+.3f chips)" % out[4]["off_chips"],
    )
    keys = [k for k, _ in lines]
    check(
        keys == ["presence-disp-refused-same offset"],
        "one refusal line, no admission line: %s" % keys,
    )
    check(
        lines and "3 PRN(s)" in lines[0][1] and "same offset" in lines[0][1],
        "the line carries the count and the reason: %s"
        % (lines[0][1][:80] if lines else ""),
    )
    check(
        out[7]["present"] is True and out[12]["present"] is True,
        "the on-peak satellites are untouched",
    )

    out, lines = verdict(fleet(**{"4": row(*DISPLACED)}), admit())
    check(
        out[4]["present"] is True and out[4]["present_gate"] == "q+p:probes+disp",
        "one displaced row alone is admitted as before",
    )
    check([k for k, _ in lines] == ["presence-disp"], "and logged as an admission")

    out, lines = verdict(
        fleet(**{"4": row(*DISPLACED), "5": row(0.90, +0.65, 20.0)}), admit()
    )
    check(
        out[4]["present"] is True and out[5]["present"] is True,
        "two rows at two offsets are both admitted",
    )

    out, lines = verdict(fleet(**{"4": row(*DISPLACED)}), admit(browned_out=True))
    check(
        out[4]["present"] is False and out[4]["present_gate"] == "q+p:probes",
        "the same row while the chain is browned out is refused",
    )
    check(
        lines and "BROWNOUT open" in lines[0][1], "and the line names the brownout",
    )

    out, lines = verdict(fleet(**{"4": row(*DISPLACED)}), None)
    check(
        out[4]["present"] is False and not lines,
        "with the displaced gate off nothing is admitted or logged",
    )


def main():
    test_rule()
    test_gate()
    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
