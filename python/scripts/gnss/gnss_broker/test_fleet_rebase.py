"""After an F-engine re-base, a pre-outage instance must not set the fleet hop.

    python3 -m gnss_broker.test_fleet_rebase

fleet_dll takes the fleet hop as the newest instance's, which is right for a laggard (only ever
behind). A re-base restarts the F-engine counter at 0, and a node not yet relaunched keeps
serving its pre-outage row, far ahead of the live ones. Not excluded, it wins the
max, the live instances fall outside hop_window, and the published fleet hop is the old
session's; the obs writers then stamp rows with the new frame0 and the old hop, dating them
that far in the future. The numbers below are from a real re-base.

@author Keith Vanderlinde
"""

import sys

import gnss_broker.fleet as fleet
from gnss_broker.fleet import fleet_dll, live_instances

RATE = 195312.5
FRAME0_NEW = 1790626549.0  # the new session's frame-0 UTC, s
NOW = 1790626598.0  # 49 s after the re-base
H_OLD = 53825781250  # the old session's last hop, ~3.2 days of counting
H_NEW = int((NOW - 2.0 - FRAME0_NEW) * RATE)

_fails = []


def check(name, ok, detail=""):
    print(
        "  [%s] %s%s"
        % ("PASS" if ok else "FAIL", name, (" -- " + detail) if detail else "")
    )
    if not ok:
        _fails.append(name)


def test_live_instances():
    hist = {}
    ex = live_instances({"a": 100, "b": 200}, hist, NOW)
    check("first call: no history, nothing excluded", ex == {}, str(ex))
    ex = live_instances({"a": 300, "b": 400}, hist, NOW + 2)
    check("steady state: everyone advances, nothing excluded", ex == {}, str(ex))

    # the outage: every instance frozen at the old session's hops, for an hour
    hist = {u: (H_OLD - i, NOW - 3600.0) for i, u in enumerate("abcd")}
    ex = live_instances({u: h for u, (h, _) in hist.items()}, hist, NOW)
    check("outage: nobody advances, nobody accused", ex == {}, str(ex))
    # the re-base: a and b relaunch (a backwards jump is a change), c and d still serve old rows
    ex = live_instances(
        {"a": H_NEW, "b": H_NEW - 4096, "c": H_OLD - 2, "d": H_OLD - 3}, hist, NOW
    )
    check(
        "re-base, broker still on the old anchor: the pre-outage instances are frozen",
        ex == {"c": "frozen", "d": "frozen"},
        str(ex),
    )

    # a broker restarted on the new anchor has no history: the anchor alone must do it
    ex = live_instances(
        {"a": H_NEW, "c": H_OLD - 2}, {}, NOW, anchor_utc=FRAME0_NEW, hops_per_sec=RATE
    )
    check(
        "re-base, new anchor, no history: the old row is AHEAD",
        ex == {"c": "ahead"},
        str(ex),
    )
    ex = live_instances(
        {"a": H_OLD, "c": H_OLD - 2}, {}, NOW, anchor_utc=FRAME0_NEW, hops_per_sec=RATE
    )
    check(
        "everyone ahead: the anchor is wrong, so nobody is excluded", ex == {}, str(ex)
    )
    ex = live_instances(
        {"a": H_NEW + int(30 * RATE)}, {}, NOW, anchor_utc=FRAME0_NEW, hops_per_sec=RATE
    )
    check("30 s ahead is inside the 60 s margin", ex == {}, str(ex))
    check(
        "no hop yet (-1) is ignored, not judged",
        live_instances({"a": -1, "b": 5}, {}, NOW) == {},
    )


def row(prn, hop, deep):
    return {
        "prn": prn,
        "pow_hop": hop,
        "e_pow": 1.0,
        "p_pow": 4.0,
        "l_pow": 1.0,
        "n_chan": 7.0,
        "deep_snr": deep,
        "amp_snr": 10.0,
        "coherence_s": 1.0,
        "utc": 0.0,
    }


def test_fleet_dll_end_to_end():
    polls = {
        "u0": [row(3, H_NEW, 50.0), row(5, H_NEW, 50.0)],
        "u1": [row(3, H_NEW - 4096, 40.0), row(5, H_NEW - 4096, 40.0)],
        "u2": [row(3, H_OLD, 99.0), row(5, H_OLD, 99.0)],
    }  # stale, and the "best" deep
    real_get = fleet._get
    fleet._get = lambda url, timeout=5.0: polls[url.rsplit("/", 1)[0]]
    try:
        src = {}
        out = fleet_dll(
            list(polls),
            hop_window=int(2 * RATE),
            min_instances=2,
            k_sigma=3.0,
            q_fallback=2.2,
            src_hops=src,
            hop_hist={},
            anchor_utc=FRAME0_NEW,
            hops_per_sec=RATE,
            now=NOW,
        )
        check(
            "the fleet hop is the live instances'",
            out.get(3, {}).get("hop") == H_NEW,
            str(out.get(3, {}).get("hop")),
        )
        check(
            "the coherent row comes from a live instance",
            (out.get(3, {}).get("coh_row") or {}).get("pow_hop") == H_NEW,
        )
        check(
            "the excluded instance still reports its hop to the instruments",
            src.get("u2") == H_OLD,
            str(src),
        )
    finally:
        fleet._get = real_get


if __name__ == "__main__":
    test_live_instances()
    test_fleet_dll_end_to_end()
    if _fails:
        print("FAILED: %d" % len(_fails))
        sys.exit(1)
    print("OK")
