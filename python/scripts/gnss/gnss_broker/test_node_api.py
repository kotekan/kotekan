"""node_api: a chain instance's requests at its row's own path or by chain, and their one name.

    python3 -m gnss_broker.test_node_api
"""

import sys
import time

from gnss_broker import node_api, signals


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


N = "http://node-a:12048"
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
    signals._CHAINS["gps_l1_ca"] = signals._CHAINS["gps_l5"]
    try:
        check(
            node_api.row(N + "/gnss0_gps_l1_ca_inject")
            == (N, 0, "gps_l1_ca", "inject"),
            "a chain key of three tokens",
        )
    finally:
        del signals._CHAINS["gps_l1_ca"]


def test_url_and_post_request():
    node_api.set_hosts([])
    check(
        node_api.url(ASM, "get_spectrum", "window=3") == ASM + "/get_spectrum?window=3"
        and node_api.post_request(INJ, "set_seeds", [1]) == (INJ + "/set_seeds", [1]),
        "no hosts: the row's own path and body",
    )
    node_api.set_hosts([N + "/gnss1_inject", N + "/gnss0_records ", ""])
    check(
        node_api.post_request(INJ, "set_seeds", [1])
        == (N + "/gnss1_inject/set_seeds", {"chain": "gal_e5a", "body": [1]}),
        "a POST to a hosted inject: its host's path, the chain in the body",
    )
    check(
        node_api.url(INJ, "get_prns") == N + "/gnss1_inject/get_prns?chain=gal_e5a",
        "a GET from a hosted inject: the chain in the query",
    )
    check(
        node_api.url(ASM, "get_spectrum", "window=3")
        == N + "/gnss0_records/get_spectrum?chain=gps_l5&window=3",
        "a hosted record assembler: the chain, then the query",
    )
    check(
        node_api.url(CMB, "get_status") == CMB + "/get_status"
        and node_api.post_request(CMB, "x", {}) == (CMB + "/x", {}),
        "a stage not hosted on that half keeps the row's own path",
    )
    for p in (
        N + "/gnss0_gal_e5a_inject",
        "http://node-b:12048/gnss1_gal_e5a_inject",
        N + "/gnss1_e5a_inject",
        "http://localhost:12050/gps_search",
    ):
        check(
            node_api.url(p, "get_prns") == p + "/get_prns"
            and node_api.post_request(p, "set_prns", [2]) == (p + "/set_prns", [2]),
            "not a hosted row, used as given: %s" % p,
        )
    try:
        node_api.set_hosts([N + "/gnss0_telem"])
        check(False, "a base that is not a half's host is refused")
    except ValueError as e:
        check("gnss0_telem" in str(e), "a base that is not a half's host is refused")
    node_api.set_hosts([])
    mine = [N + "/gnss1_inject"]
    check(
        node_api.url(INJ, "get_prns", hosts=mine)
        == N + "/gnss1_inject/get_prns?chain=gal_e5a"
        and node_api.url(INJ, "get_prns") == INJ + "/get_prns"
        and node_api.post_request(INJ, "set_trim", [1], hosts=mine)
        == (N + "/gnss1_inject/set_trim", {"chain": "gal_e5a", "body": [1]}),
        "a caller's own hosts, without set_hosts",
    )
    node_api.set_hosts([N + "/gnss1_inject", N + "/gnss0_records"])
    for u, p in (
        (node_api.url(ASM, "get_spectrum", "window=3"), None),
        node_api.post_request(INJ, "set_seeds", [{"prn": 3}]),
    ):
        k = node_api.canonical(u, p)
        check(k[0].startswith("node:"), "a request in either form has a row's name")
    node_api.set_hosts([])
    check(
        node_api.canonical(*node_api.post_request(INJ, "set_seeds", [1]))
        == node_api.canonical(
            *node_api.post_request(INJ, "set_seeds", [1], hosts=mine)
        ),
        "the two forms of one POST have one canonical name and body",
    )


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


def test_by_row():
    """A 404 by chain puts the request at its row's own path and the host's rows there for
    BY_ROW_S; anything else is not a host's request by chain."""
    node_api.set_hosts([N + "/gnss1_inject", N + "/gnss0_records"])
    try:
        u, p = node_api.post_request(INJ, "set_seeds", [1])
        check(
            node_api.by_row(u, p) == (INJ + "/set_seeds", [1], N + "/gnss1_inject"),
            "a POST by chain goes to its row with the row's own body",
        )
        check(
            node_api.url(INJ, "get_prns") == INJ + "/get_prns"
            and node_api.post_request(INJ, "set_trim", [2]) == (INJ + "/set_trim", [2])
            and node_api.hosted(INJ) is None,
            "then the host's rows are asked at their own paths",
        )
        check(
            node_api.url(ASM, "get_spectrum")
            == N + "/gnss0_records/get_spectrum?chain=gps_l5",
            "and another host is still asked by chain",
        )
        check(
            node_api.by_row(N + "/gnss0_records/get_spectrum?chain=gps_l5&window=3")
            == (ASM + "/get_spectrum?window=3", None, N + "/gnss0_records"),
            "a GET by chain keeps the rest of its query",
        )
        node_api._by_row[N + "/gnss1_inject"] = time.monotonic() - 1.0
        check(
            node_api.url(INJ, "get_prns") == N + "/gnss1_inject/get_prns?chain=gal_e5a",
            "after BY_ROW_S the host is asked by chain again",
        )
        for u, p in (
            (INJ + "/set_seeds", [1]),
            (N + "/gnss0_records/get_spectrum", None),
            (N + "/gnss1_inject/set_trim", {"chains": {"gal_e5a": []}}),
            ("http://localhost:12050/gps_search/get_detections", None),
        ):
            check(node_api.by_row(u, p) is None, "not a request by chain: %s" % u)
    finally:
        node_api.set_hosts([])
        node_api._by_row.clear()


def test_names():
    """A recorded request's canonical name must name the url it was sent to."""
    body = {"chain": "gal_e5a", "body": [1]}
    for u, p in (
        (INJ + "/set_seeds", [1]),
        (N + "/gnss1_inject/set_seeds", body),
        (N + "/gnss0_records/get_spectrum?chain=gps_l5&window=3", None),
        (N + "/gnss1_n2combine/get_status", None),
    ):
        check(node_api.names(u, node_api.canonical(u, p)[0]), "its own name: %s" % u)
    key = node_api.canonical(INJ + "/set_seeds")[0]
    other = key.replace("gal_e5a", "bds_b2a")
    check(
        not node_api.names(N + "/gnss1_bds_b2a_inject/set_seeds", key)
        and not node_api.names(INJ + "/set_seeds", other)
        and not node_api.names(N + "/gnss0_inject/set_seeds", key)
        and not node_api.names(N + "/gnss1_records/set_seeds", key)
        and not node_api.names("http://localhost:12050/gps_search/x", key),
        "a name that does not fit the url: another chain, half or host",
    )


def test_canonical_requests():
    """A batched set_trim is each chain's own set_trim; any other request is canonical()'s."""
    trims = {"gps_l5": [{"prn": 3, "trim_chips": 0.1}], "gal_e5a": []}
    got = node_api.canonical_requests(N + "/gnss1_inject/set_trim", {"chains": trims})
    want = [
        node_api.canonical(N + "/gnss1_gal_e5a_inject/set_trim", []),
        node_api.canonical(N + "/gnss1_gps_l5_inject/set_trim", trims["gps_l5"]),
    ]
    check(got == want, "a batch is its chains' own set_trims, in chain order")
    body = {"chain": "gal_e5a", "body": []}
    check(
        node_api.canonical_requests(N + "/gnss1_inject/set_trim", body)
        == [node_api.canonical(N + "/gnss1_inject/set_trim", body)]
        and node_api.canonical_requests(N + "/gnss1_records/set_trim", {"chains": {}})
        == [(N + "/gnss1_records/set_trim", {"chains": {}})],
        "anything else as canonical() names it",
    )


def main():
    test_row()
    test_url_and_post_request()
    test_canonical()
    test_by_row()
    test_names()
    test_canonical_requests()
    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: node_api rows, forms and canonical names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
