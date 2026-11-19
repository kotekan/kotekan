"""node_api: a chain instance's rows and the one name of a request in either form.

    python3 -m gnss_broker.test_node_api
"""

import sys

from gnss_broker import node_api


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


N = "http://cx52:12048"
INJ = N + "/gnss1_gal_e5a_inject"
ASM = N + "/gnss0_gps_l5_n2assemble"
CMB = N + "/gnss0_bds_b2a_n2combine"


def test_row():
    check(node_api.row(INJ) == (N, 1, "gal_e5a", "inject"), "a tagged inject row")
    check(node_api.row(CMB) == (N, 0, "bds_b2a", "n2combine"), "a tagged combiner row")
    for p in (
        N + "/gnss0_inject",
        N + "/gnss0_n2combine",
        N + "/gnss0_e5a_inject",
        N + "/gnss0_gps_l9_inject",
        N + "/gnss0_gal_e5a_n2assemble_tiles",
        N + "/gnss0_srch_tap",
        "http://localhost:12777/track",
        "http://localhost:12050/gps_search",
    ):
        check(node_api.row(p) is None, "not a row named by its broker chain: %s" % p)


def test_canonical():
    c = node_api.canonical
    key = "node:%s/gnss1/inject/gal_e5a/set_seeds" % N
    body = [{"prn": 3, "cp": 1.5}]
    check(c(INJ + "/set_seeds", body) == (key, body), "a row's POST")
    check(
        c(N + "/gnss1_inject/set_seeds", {"chain": "gal_e5a", "body": body})
        == (key, body),
        "the same POST to its host by chain",
    )
    g = "node:%s/gnss0/n2assemble/gps_l5/get_spectrum?window=3" % N
    check(
        c(ASM + "/get_spectrum?window=3")[0] == g
        and c(N + "/gnss0_records/get_spectrum?chain=gps_l5&window=3")[0] == g
        and c(N + "/gnss0_records/get_spectrum?window=3&chain=gps_l5")[0] == g,
        "a GET with a query, in either form and either order",
    )
    check(
        c(N + "/gnss0_records/get_spectrum?chain=gal_e5a&window=3")[0] != g
        and c(N + "/gnss1_inject/set_seeds", {"chain": "bds_b2a", "body": body})[0]
        != key
        and c(N + "/gnss1_bds_b2a_inject/set_seeds", body)[0] != key,
        "another chain is another request",
    )
    for u, p in (
        (N + "/gnss0_combine/get_status", None),
        (N + "/gnss1_inject/set_seeds", body),
        (N + "/gnss1_inject/set_seeds", {"chain": "gal_e5a", "body": body, "x": 1}),
        (N + "/gnss1_inject/set_trim", {"chains": {"gal_e5a": []}}),
        (N + "/gnss0_inject/set_seeds", body),
        ("http://localhost:12050/gps_search/get_detections", None),
    ):
        check(c(u, p) == (u, p), "not one row's request, named as it is: %s" % u)
    check(
        c(N + "/gnss0_e5a_inject/set_seeds", body)
        == ("node:%s/gnss0/inject/gal_e5a/set_seeds" % N, body)
        and c(N + "/gnss1_n2combine/get_status")[0]
        == "node:%s/gnss1/n2combine/gps_l5/get_status" % N,
        "names from before the tags, through stage_chain's allowlists",
    )
    k = c(N + "/gnss1_inject/set_seeds", {"chain": "gal_e5a", "body": body})[0]
    check(c(k) == (k, None), "a canonical name is its own")


def main():
    test_row()
    test_canonical()
    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: node_api rows and canonical names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
