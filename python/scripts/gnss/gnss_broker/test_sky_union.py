"""The served sky: one constellation, several chains, one union.

    python3 -m gnss_broker.test_sky_union

Several chains predict one constellation (GPS l5/l2c, BeiDou b2a/b2b/b3i), each filtered to
the satellites that transmit its signal. get_sky used to serve whichever chain wrote last, so
the viewer's rows came and went every cycle. What is pinned:
  * a constellation is the union of its chains' satellites, whatever the write order;
  * a satellite two chains predict takes the newest prediction and its age;
  * a satellite only a frozen chain holds stays, with that chain's growing age;
  * a single-chain broker (chain=None) and the per-chain view both still serve;
  * GET /get_sky answers through the real dispatch, which holds the publisher's lock (a
    re-entrant take deadlocked the broker 10-26 14:43Z).

@author Keith Vanderlinde
"""

import json
import sys
import urllib.request

from gnss_broker import publish

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def _pub():
    # The store and sky_rows only; no port is opened until start().
    return publish.FleetPublisher(0, lambda *a, **k: None)


def _prns(rows, sysc):
    return sorted(r["prn"] for r in rows if r["const"] == sysc)


def main():
    print("union across chains, either write order")
    l2c = {p: (40.0 + p, 100.0 + p) for p in (1, 3, 5, 7, 15, 17)}
    l5 = {p: (40.0 + p, 100.0 + p) for p in (1, 3)}
    for order in (("l2c", "l5"), ("l5", "l2c")):
        pub = _pub()
        for ch in order:
            pub.set_sky("G", l2c if ch == "l2c" else l5, 100.0, chain="gps_" + ch)
        got = _prns(pub.sky_rows(101.0), "G")
        check(got == [1, 3, 5, 7, 15, 17], "write order %s -> %s" % ("/".join(order), got))

    print("overlap takes the newest")
    pub = _pub()
    pub.set_sky("C", {20: (30.0, 10.0)}, 100.0, chain="bds_b2a")
    pub.set_sky("C", {20: (31.0, 11.0), 21: (50.0, 12.0)}, 130.0, chain="bds_b3i")
    rows = {r["prn"]: r for r in pub.sky_rows(140.0) if r["const"] == "C"}
    c20 = rows[20]
    check(c20["el"] == 31.0 and c20["age_s"] == 10.0, "C20 from b3i, age 10: %s" % c20)
    check(rows[21]["age_s"] == 10.0, "C21 present: %s" % rows[21])

    print("a frozen chain ages, it does not vanish")
    pub.set_sky("C", {21: (50.5, 12.0)}, 200.0, chain="bds_b2b")
    rows = {r["prn"]: r for r in pub.sky_rows(205.0) if r["const"] == "C"}
    check(rows[20]["age_s"] == 75.0, "C20 only on b3i, age 75: %s" % rows[20])
    c21 = rows[21]
    check(c21["age_s"] == 5.0 and c21["el"] == 50.5, "C21 newest from b2b: %s" % c21)

    print("constellations stay apart; the single-chain and per-chain paths")
    pub = _pub()
    pub.set_sky("E", {4: (60.0, 200.0)}, 10.0)
    view = publish._ChainView(pub, "gps_l5")
    view.set_sky("G", {9: (20.0, 300.0)}, 12.0)
    rows = pub.sky_rows(12.0)
    check(
        [(r["const"], r["prn"]) for r in rows] == [("E", 4), ("G", 9)],
        "E4 + G9, sorted: %s" % [(r["const"], r["prn"]) for r in rows],
    )
    check(set(pub._sky["G"]) == {"gps_l5"}, "view tags its chain: %s" % list(pub._sky["G"]))

    print("GET /get_sky through the dispatch")
    pub.set_sky("C", {13: (85.0, 170.0)}, 12.0, chain="bds_b3i")
    url = "http://127.0.0.1:%d/get_sky" % pub._srv.server_address[1]
    try:
        with urllib.request.urlopen(url, timeout=5.0) as r:
            got = [(x["const"], x["prn"]) for x in json.loads(r.read())]
    except Exception as e:
        got = "no answer: %r" % e
    check(got == [("C", 13), ("E", 4), ("G", 9)], "served %s" % (got,))
    pub._srv.shutdown()

    if _fails:
        print("FAIL: %d" % len(_fails))
        sys.exit(1)
    print("all pass")


if __name__ == "__main__":
    main()
