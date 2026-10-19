#!/usr/bin/env python3
"""Does the viewer call a dead band 'locked'? Run the shipped JS against real rows.

    /home/kvand/gnss/venv-ft/bin/python test_lock_metrics.py [publisher_host:port]

With nothing on the wire a chain can still serve sig in the thousands, far past the 3-sigma
bar, and the incoherent C/N0 has a pole at u -> a. A lock rule keyed on sig, or an unguarded
C/N0, then draws a full constellation of locked satellites at implausible C/N0 on a dead
band. This checks that the lock rule judges on q against the published floor (sig decides
only where a chain publishes no q), that a noise probe never reads locked, that a dark probe
on one signal does not grey a satellite another signal holds (#206), and that the C/N0
refuses rather than diverges.

It parses the real file. The lock rule and signal_metrics() are extracted from
app/panels/gps_feed.js by brace matching and executed with dukpy, so this cannot drift from
what the browser runs the way a re-implementation would. A syntax error in the extracted code
fails this too; check_js.sh syntax-checks every viewer file.

It checks both directions. A test that only checks "noise reads unlocked" passes just as well
on a viewer that never reports a lock at all, so the healthy synthetic rows below are what
make the negative result mean something.

The live section reads the chain list from the publisher's /get_chains and each chain's rows
from /<chain>/get_status over HTTP from the host this runs on, so publisher_host:port (default
127.0.0.1:12060, the broker's publisher on the gnss VM) must be reachable from here. It prints
counts and asserts nothing. The fetch path itself is checked against a canned publisher on a
loopback port.

@author Keith Vanderlinde
"""
import http.server
import json
import re
import sys
import threading
import urllib.request

import dukpy

HERE = __file__.rsplit("/", 1)[0]
FEED = HERE + "/app/panels/gps_feed.js"

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def brace_block(src, start_pat):
    """The source of a function, from its declaration to the matching close brace."""
    m = re.search(start_pat, src)
    if not m:
        raise SystemExit(
            "cannot find %r in %s -- the panel was restructured" % (start_pat, FEED)
        )
    i = src.index("{", m.start())
    depth, j = 0, i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[m.start() : j + 1]
        j += 1
    raise SystemExit("unbalanced braces after %r" % start_pat)


def build_js():
    src = open(FEED).read()
    consts = "\n".join(
        l
        for l in src.split("\n")
        if re.match(
            r"const (AMP_LOCK|SNR_LOCK|Q_LOCK|Q_FLOOR_MARGIN|" r"CN0_INC_MAX_RATIO)\b",
            l,
        )
    )
    for name in (
        "AMP_LOCK",
        "SNR_LOCK",
        "Q_LOCK",
        "Q_FLOOR_MARGIN",
        "CN0_INC_MAX_RATIO",
    ):
        if name not in consts:
            raise SystemExit("constant %s missing from %s" % (name, FEED))
    metrics = brace_block(src, r"export function signal_metrics")
    metrics = re.sub(r"^export\s+", "", metrics)
    # The lock rule lives inside _merge(); take it verbatim from `const have_sig` to the
    # statement before `return list;`.
    m = re.search(r"( *const have_sig = .*?)\n\s*return list;", src, re.S)
    if not m:
        raise SystemExit("cannot find the lock block in %s" % FEED)
    lock = m.group(1)
    # _merge() itself, so the rules are also checked on the rows the panel renders: the row
    # summary must carry what the lock rule reads (q, q_floor, noise_probe).
    merge = "function " + brace_block(src, r"_merge\(\) \{")
    return """
%s
%s
%s
function run_merged(units) {
    var panel = {_last: {units: units},
                 signals: [{key: "a", sys: "G", t_rec: 0.001}, {key: "b", sys: "G", t_rec: 0.001}]};
    var list = _merge.call(panel);
    var out = [];
    for (var k = 0; k < list.length; k++)
        out.push({prn: list[k].prn, active: !!list[k].active,
                  q: list[k].q === undefined ? null : list[k].q});
    return out;
}
function run(rows, t_rec) {
    var list = [];
    for (var i = 0; i < rows.length; i++) {
        var m = signal_metrics(rows[i], t_rec);
        m.prn = rows[i].prn;
        list.push(m);
    }
%s
    var out = [];
    for (var k = 0; k < list.length; k++)
        out.push({prn: list[k].prn, active: !!list[k].active,
                  cn0: (list[k].cn0 === null || list[k].cn0 === undefined) ? null : list[k].cn0,
                  q: list[k].q === undefined ? null : list[k].q,
                  sig: list[k].sig});
    return out;
}
""" % (
        consts,
        metrics,
        merge,
        lock,
    )


def run_js(js, rows, t_rec=0.001):
    return dukpy.evaljs(
        js + "\nrun(dukpy['rows'], dukpy['t_rec']);", rows=rows, t_rec=t_rec
    )


def run_merged(js, units):
    return dukpy.evaljs(js + "\nrun_merged(dukpy['units']);", units=units)


def get_json(host, path):
    """GET http://<host>/<path> as JSON; None when the publisher cannot be read."""
    try:
        with urllib.request.urlopen("http://%s/%s" % (host, path), timeout=5) as r:
            return json.load(r)
    except Exception as e:
        print("   (%s unreachable: %s)" % (path, e))
        return None


def chains(host):
    """The chain ids the publisher is running, from its /get_chains discovery list."""
    d = get_json(host, "get_chains")
    return [c["chain"] for c in d if c.get("chain")] if isinstance(d, list) else []


def fetch(host, chain):
    d = get_json(host, "%s/get_status" % chain)
    return d if isinstance(d, list) else []


def canned_publisher(routes):
    """A throwaway HTTP server on a free loopback port answering GET <path> with routes[path]
    as JSON (404 for any other path). The caller shuts it down."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in routes:
                self.send_error(404)
                return
            body = json.dumps(routes[self.path]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:12060"
    js = build_js()
    print("extracted lock rule + signal_metrics from gps_feed.js (it parses)\n")

    # ---- positive control: a healthy satellite must read locked -------------------------
    print(
        "healthy satellites (synthetic) -- the direction that makes a null mean something"
    )
    healthy = [
        {
            "prn": 1,
            "fleet_q": 3.4,
            "fleet_q_floor": 1.1,
            "sig": 5000.0,
            "amplitude": 1e-6,
            "unbiased_amplitude": 8e-7,
            "cn0_prompt_db": 42.0,
            "cn0_prompt_duty": 0.98,
        },
        {
            "prn": 2,
            "fleet_q": 2.3,
            "fleet_q_floor": 1.1,
            "sig": 40.0,
            "amplitude": 1e-6,
            "unbiased_amplitude": 5e-7,
            "cn0_prompt_db": 37.5,
            "cn0_prompt_duty": 0.8,
        },
    ]
    r = {x["prn"]: x for x in run_js(js, healthy)}
    check(
        r[1]["active"] and r[2]["active"], "a satellite with q 3.4 / 2.3 reads LOCKED"
    )
    check(
        abs(r[1]["cn0"] - 42.0) < 1e-9,
        "and its published C/N0 is passed through (42.0)",
    )

    # ---- the marginal and at-noise cases ------------------------------------------------
    print("\nthe bars")
    edge = [
        {"prn": 3, "fleet_q": 1.9, "fleet_q_floor": 1.1, "sig": 9e3},  # marginal
        {"prn": 4, "fleet_q": 1.05, "fleet_q_floor": 1.1, "sig": 9e3},  # at the floor
        {
            "prn": 5,
            "fleet_q": 2.4,
            "fleet_q_floor": 2.3,
            "sig": 9e3,
        },  # above the bar, at the floor
    ]
    r = {x["prn"]: x for x in run_js(js, edge)}
    check(
        not r[3]["active"],
        "q 1.9 (under Q_LOCK) is not locked -- half-locked is not locked",
    )
    check(not r[4]["active"], "q 1.05 against a 1.10 floor is not locked")
    check(
        not r[5]["active"],
        "q 2.4 clears the fixed bar but sits on a raised floor (2.3) -- still not locked, "
        "because the published floor beats a hardcoded number",
    )

    # ---- sig cannot override q; it decides only where no q is published -----------------
    print("\nsig cannot manufacture a lock")
    r = {
        x["prn"]: x
        for x in run_js(
            js, [{"prn": 6, "fleet_q": 1.0, "fleet_q_floor": 1.1, "sig": 10000.0}]
        )
    }
    check(
        not r[6]["active"],
        "sig 10,000 with q 1.0 reads UNLOCKED (a dead band can serve sig this high)",
    )
    r = {x["prn"]: x for x in run_js(js, [{"prn": 7, "sig": 10000.0}])}
    check(
        r[7]["active"], "but a chain that publishes no q still falls back to sig",
    )

    # ---- the C/N0 pole -------------------------------------------------------------------
    print("\nthe fallback incoherent C/N0 and its pole at u -> a")
    pole = [
        # A broker that declined: duty present, no cn0_prompt_db -> a gap, never a fallback.
        {
            "prn": 8,
            "fleet_q": 1.0,
            "fleet_q_floor": 1.1,
            "sig": 9e3,
            "cn0_prompt_duty": 0.01,
            "amplitude": 2.8e-7,
            "unbiased_amplitude": 2.79999e-7,
        },
        # A broker that publishes neither key, at the pole (u/a ~ 1).
        {
            "prn": 9,
            "fleet_q": 1.0,
            "fleet_q_floor": 1.1,
            "sig": 9e3,
            "amplitude": 2.8e-7,
            "unbiased_amplitude": 2.79999e-7,
        },
        # Neither key, on a real signal: still renders.
        {
            "prn": 10,
            "fleet_q": 3.0,
            "fleet_q_floor": 1.1,
            "sig": 9e3,
            "amplitude": 1e-6,
            "unbiased_amplitude": 5e-7,
        },
    ]
    r = {x["prn"]: x for x in run_js(js, pole)}
    check(
        r[8]["cn0"] is None,
        "a declined C/N0 (duty, no cn0_prompt_db) renders a gap, not the fallback estimate",
    )
    check(
        r[9]["cn0"] is None,
        "a row with neither C/N0 key at u/a = 0.999996 is refused (the pole)",
    )
    check(
        r[10]["cn0"] is not None and 20.0 < r[10]["cn0"] < 45.0,
        "a row with neither C/N0 key, on real signal, renders a sane value (%.1f dB-Hz)"
        % (r[10]["cn0"] or 0),
    )

    # ---- the noise probes ----------------------------------------------------------------
    print("\nnoise probes are never satellites")
    r = {
        x["prn"]: x
        for x in run_js(
            js,
            [
                {
                    "prn": 11,
                    "fleet_q": 3.5,
                    "fleet_q_floor": 1.1,
                    "sig": 9e3,
                    "noise_probe": True,
                    "cn0_prompt_db": 31.0,
                    "cn0_prompt_duty": 0.9,
                },
                {
                    "prn": 12,
                    "fleet_q": 3.5,
                    "fleet_q_floor": 1.1,
                    "sig": 9e3,
                    "cn0_prompt_db": 31.0,
                    "cn0_prompt_duty": 0.9,
                },
            ],
        )
    }
    check(
        not r[11]["active"],
        "a noise_probe row cannot read locked even at q 3.5 -- it is the floor by "
        "construction, and publish.py flags it so no consumer plots it as a satellite",
    )
    check(r[12]["active"], "the identical non-probe row does read locked (the control)")

    # ---- the same rules through _merge(), the rows the panel renders ---------------------
    print("\nthrough _merge() -- the rows the panel actually renders")
    dead = dict(prn=6, fleet_q=1.0, fleet_q_floor=1.1, sig=1e4)
    probe = dict(prn=11, fleet_q=3.5, fleet_q_floor=1.1, sig=9e3, noise_probe=True)
    two_a = dict(prn=7, fleet_q=1.0, fleet_q_floor=1.1, sig=1e4)
    two_b = dict(prn=7, fleet_q=3.0, fleet_q_floor=1.1, sig=50.0)
    no_q = dict(prn=8, sig=5000.0)
    r = {
        x["prn"]: x
        for x in run_merged(
            js,
            {
                "a": {"status": [healthy[0], dead, probe, two_a, no_q]},
                "b": {"status": [two_b]},
            },
        )
    }
    check(
        r[1]["active"], "a healthy satellite reads LOCKED through _merge (the control)"
    )
    check(not r[6]["active"], "sig 10,000 with q 1.0 reads UNLOCKED through _merge")
    check(not r[11]["active"], "a noise probe reads UNLOCKED through _merge")
    check(
        r[7]["active"] and r[7]["q"] == 3.0,
        "a satellite dead on one signal and locked on another reads LOCKED, on the "
        "locked signal's q",
    )
    check(r[8]["active"], "a row with no q on any signal still falls back to sig")

    # ---- dark probes (glo_l2oc --probe-dark, 10-18): an overhead satellite seeded as the
    # noise reference on a signal it cannot broadcast, while another chain tracks it (#206) --
    print("\ndark probes: a probe on one signal does not grey a satellite another signal holds")
    dark_a = dict(prn=4, fleet_q=0.9, fleet_q_floor=1.1, sig=-0.2, noise_probe=True)
    dark_b = dict(
        prn=4, fleet_q=5.9, fleet_q_floor=1.1, sig=3000.0, cn0_prompt_db=36.0,
        cn0_prompt_duty=1.0,
    )
    both_a = dict(prn=5, fleet_q=0.9, fleet_q_floor=1.1, sig=-0.2, noise_probe=True)
    both_b = dict(prn=5, fleet_q=1.0, fleet_q_floor=1.1, sig=0.5, noise_probe=True)
    r = {
        x["prn"]: x
        for x in run_merged(
            js,
            {"a": {"status": [dark_a, both_a, probe]}, "b": {"status": [dark_b, both_b]}},
        )
    }
    check(
        r[4]["active"] and r[4]["q"] == 5.9,
        "a dark probe on one signal, locked at q 5.9 on another, reads LOCKED on that q "
        "(R4: L2OC probe, L3OC lock)",
    )
    check(
        not r[5]["active"],
        "a satellite that is a probe on every signal reporting it stays UNLOCKED",
    )
    check(
        not r[11]["active"],
        "a probe on its only signal stays UNLOCKED (the below-horizon kind)",
    )

    # ---- the fetch path, against a canned publisher --------------------------------------
    print("\nfetch: the chains from /get_chains, the rows from /<chain>/get_status")
    srv = canned_publisher(
        {
            "/get_chains": [{"chain": "gps_l5"}, {"chain": "glo_l2oc"}],
            "/gps_l5/get_status": healthy,
            "/glo_l2oc/get_status": [probe],
        }
    )
    try:
        canned = "127.0.0.1:%d" % srv.server_address[1]
        check(
            chains(canned) == ["gps_l5", "glo_l2oc"],
            "the live section reads every chain the publisher lists",
        )
        check(
            fetch(canned, "gps_l5") == healthy and fetch(canned, "glo_l2oc") == [probe],
            "a chain's rows are its /<chain>/get_status, read from the host given",
        )
    finally:
        srv.shutdown()
        srv.server_close()

    # ---- the live rows -------------------------------------------------------------------
    print("\nlive rows from the publisher at %s" % host)
    tot = locked = withc = 0
    worst = None
    for ch in chains(host):
        rows = fetch(host, ch)
        if not rows:
            continue
        out = run_js(js, rows)
        for o in out:
            tot += 1
            locked += 1 if o["active"] else 0
            withc += 1 if o["cn0"] is not None else 0
            if worst is None or (o["sig"] or 0) > worst[0]:
                worst = ((o["sig"] or 0), ch, o["prn"], o["q"], o["active"])
    if not tot:
        print("   no live rows -- skipping (this section needs a running publisher)")
    else:
        print(
            "   %d satellite row(s): %d would render LOCKED, %d would show a C/N0"
            % (tot, locked, withc)
        )
        if worst:
            print(
                "   loudest sig: %s PRN %s  sig %.0f  q %s  active=%s"
                % (worst[1], worst[2], worst[0], worst[3], worst[4])
            )
        # No assertion on the count: this file must stay useful with signal, where a nonzero
        # locked count is the correct answer. The number is printed so a human can read it
        # against what the sky is actually doing.

    print("\nFAILED (%d)" % len(_fails) if _fails else "\nOK")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
