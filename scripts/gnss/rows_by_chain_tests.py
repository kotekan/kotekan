#!/usr/bin/env python3
"""The broker's fallbacks for rows asked by chain, offline. Run on demand, not by CI or
run_tests.py (its name is not test_*.py):

    python3 scripts/gnss/rows_by_chain_tests.py     (from the checkout's root, the broker's venv)

1. A GPU half's host asked by chain that answers 404 (a node older than --node-hosts says) is
   asked again at the row's own path, a GET with its query and a POST with the row's own body,
   and that host's rows are then asked at their own paths for node_api.BY_ROW_S. A local HTTP
   server stands in for the node.
2. trimarm.trim_targets follows that window: the half's tracker is its own set_trim within it.
3. The start-up gather check (gps_distributed_broker.main, --fleet-trim-half-targets): a gather
   that batches gives the halves' targets; one whose interface lacks rest/fleet_trim_batch, or
   that serves no interface, refuses the chain; one that cannot be read starts the chain on the
   trackers' own set_trim with an ERROR, and is read again after trimarm.GATHER_RECHECK_S.
4. --node-hosts is per process: under broker_multi a chain naming other hosts than a running
   chain refuses to start, and the running chain keeps its hosts.

main() is run only to the end of its start-up checks: its first call after them,
resolve_prefix, is replaced by one that stops it.
"""
import contextlib
import http.server
import io
import json
import os
import re
import sys
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import broker_multi  # noqa: E402

from gnss_broker import iface, node_api, transport, trimarm  # noqa: E402

broker = broker_multi.broker

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class _Later(object):
    """The time module with monotonic() dt seconds on, for a module's clock."""

    def __init__(self, dt):
        self.dt = dt

    def monotonic(self):
        return time.monotonic() + self.dt

    def __getattr__(self, name):
        return getattr(time, name)


@contextlib.contextmanager
def later(dt, *mods):
    for m in mods:
        m.time = _Later(dt)
    try:
        yield
    finally:
        for m in mods:
            m.time = time


_HOST = re.compile(r"^/gnss\d+_(inject|records|combine)/")


class _Node(http.server.BaseHTTPRequestHandler):
    """A node whose halves' hosts do not serve by chain (404), and whose rows answer at their
    own paths with the path and body they were sent."""

    def _answer(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n)) if n else None
        self.server.asked.append((self.command, self.path, body))
        if _HOST.match(self.path):
            self.send_error(404)
            return
        out = json.dumps({"path": self.path, "body": body}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    do_GET = do_POST = _answer

    def log_message(self, *args):
        pass


def test_404_fallback_and_trim_targets():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Node)
    srv.asked = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    origin = "http://127.0.0.1:%d" % srv.server_address[1]
    combine0 = origin + "/gnss0_gps_l5_n2combine"
    inject0 = origin + "/gnss0_gps_l5_inject"
    inject1 = origin + "/gnss1_gps_l5_inject"
    half0 = {"url": origin + "/gnss0_inject/set_trim", "chain": "gps_l5"}
    half1 = {"url": origin + "/gnss1_inject/set_trim", "chain": "gps_l5"}
    node_api.set_hosts(
        [origin + "/gnss0_combine", origin + "/gnss0_inject", origin + "/gnss1_inject"]
    )
    try:
        url = node_api.url(combine0, "get_status", "window=3")
        check(
            url == origin + "/gnss0_combine/get_status?chain=gps_l5&window=3",
            "a hosted row's GET is asked by chain (%s)" % url,
        )
        try:
            got = transport._get(url, timeout=2.0)
        except Exception as e:
            got = repr(e)
        check(
            got == {"path": "/gnss0_gps_l5_n2combine/get_status?window=3", "body": None}
            and [a[:2] for a in srv.asked]
            == [
                ("GET", "/gnss0_combine/get_status?chain=gps_l5&window=3"),
                ("GET", "/gnss0_gps_l5_n2combine/get_status?window=3"),
            ],
            "a GET the host answers 404 is asked again at the row's own path, with its "
            "query (%r)" % (got,),
        )
        check(
            node_api.url(combine0, "get_status") == combine0 + "/get_status",
            "that host's rows are then asked at their own paths",
        )
        del srv.asked[:]
        try:
            transport._get(node_api.url(combine0, "get_status"), timeout=2.0)
        except Exception:
            pass
        check(
            [a[1] for a in srv.asked] == ["/gnss0_gps_l5_n2combine/get_status"],
            "and go there first (%r)" % srv.asked,
        )

        check(
            trimarm.trim_targets([inject0, inject1], by_half=True) == [half0, half1],
            "before any 404 both halves' trackers are their halves' targets",
        )
        trim = {"trims": {"8": 0.01}}
        url, payload = node_api.post_request(inject1, "set_trim", trim)
        del srv.asked[:]
        try:
            st = transport._post(url, payload, timeout=2.0)
        except Exception as e:
            st = repr(e)
        check(
            st == 200
            and srv.asked
            == [
                ("POST", "/gnss1_inject/set_trim", {"chain": "gps_l5", "body": trim}),
                ("POST", "/gnss1_gps_l5_inject/set_trim", trim),
            ],
            "a POST the host answers 404 is posted again at the row's own path, with the "
            "row's own body (%r, %r)" % (st, srv.asked),
        )
        check(
            node_api.post_request(inject1, "set_trim", trim)
            == (inject1 + "/set_trim", trim)
            and node_api.post_request(inject0, "set_trim", trim)
            == (half0["url"], {"chain": "gps_l5", "body": trim}),
            "only that host's rows are posted at their own paths",
        )
        check(
            trimarm.trim_targets([inject0, inject1], by_half=True)
            == [half0, inject1 + "/set_trim"],
            "trim_targets follows it: that half's tracker is its own set_trim, the other "
            "half's is its half's target (%r)"
            % trimarm.trim_targets([inject0, inject1], by_half=True),
        )

        with later(node_api.BY_ROW_S - 1.0, node_api):
            inside = (
                node_api.url(combine0, "get_status"),
                trimarm.trim_targets([inject1], by_half=True),
            )
        with later(node_api.BY_ROW_S + 1.0, node_api):
            after = (
                node_api.url(combine0, "get_status"),
                trimarm.trim_targets([inject1], by_half=True),
            )
        check(
            inside == (combine0 + "/get_status", [inject1 + "/set_trim"]),
            "for BY_ROW_S (%.0f s) after the 404 (%r)" % (node_api.BY_ROW_S, inside),
        )
        check(
            after == (origin + "/gnss0_combine/get_status?chain=gps_l5", [half1]),
            "and by chain again after it (%r)" % (after,),
        )
    finally:
        srv.shutdown()
        srv.server_close()
        node_api.set_hosts([])
        node_api._by_row.clear()


class _Started(Exception):
    """main() got past its start-up checks."""


def start(argv):
    """("started" or the SystemExit code, stderr) of main(argv) run to the end of its start-up
    checks."""

    def stop(entry, base):
        raise _Started()

    real = broker.resolve_prefix
    broker.resolve_prefix = stop
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            broker.main(argv, rx=object())
        got = "returned"
    except _Started:
        got = "started"
    except SystemExit as e:
        got = e.code
    finally:
        broker.resolve_prefix = real
    return got, err.getvalue()


def test_gather_at_start():
    gather = "http://127.0.0.1:1/fleet_trim"
    argv = [
        "--signal",
        "gps_l5",
        "--trackers",
        "http://node-a:12048/gnss0_gps_l5_inject",
        "--fleet-trim-url",
        gather,
        "--fleet-trim-half-targets",
    ]
    args = types.SimpleNamespace(
        fleet_trim_half_targets=True, fleet_trim_url=gather, transcript_read=None
    )
    golden = iface.golden().doc
    with_leaf = dict(golden, sections=dict(golden["sections"]))
    with_leaf["sections"]["rest"] = dict(
        golden["sections"]["rest"], **{trimarm.GATHER_BATCH: 1}
    )
    without = dict(golden, sections=dict(golden["sections"]))
    without["sections"]["rest"] = dict(golden["sections"]["rest"])
    without["sections"]["rest"].pop(trimarm.GATHER_BATCH, None)
    reads = []

    def serves(doc):
        def read_source(src, timeout=None):
            reads.append(src)
            if doc is None:
                raise iface.SourceError("connection refused")
            if doc == "legacy":
                raise iface.Legacy("no gnss interface")
            return iface.Source(doc, None, src)

        return read_source

    real_read = iface.read_source
    try:
        iface.read_source = serves(with_leaf)
        got, err = start(argv)
        n = len(reads)
        check(
            got == "started" and n == 1 and trimarm.half_targets(args) is True,
            "a gather that batches: the chain starts on the halves' targets (%s, %d "
            "read(s))" % (got, n),
        )
        check(len(reads) == 1, "and the gather is not read again (%d)" % len(reads))

        for doc, what in ((without, "lacks rest/fleet_trim_batch"), ("legacy", "")):
            trimarm._gather.clear()
            iface.read_source = serves(doc)
            got, err = start(argv)
            check(
                got == 2 and "--fleet-trim-half-targets: the gather at" in err,
                "a gather that %s refuses the chain (%s)"
                % (what or "serves no interface", got),
            )

        trimarm._gather.clear()
        iface.read_source = serves(None)
        got, err = start(argv)
        check(
            got == "started" and "ERROR --fleet-trim-half-targets" in err,
            "a gather that cannot be read starts the chain, with an ERROR (%s)" % got,
        )
        del reads[:]
        iface.read_source = serves(with_leaf)
        with later(trimarm.GATHER_RECHECK_S - 1.0, trimarm):
            within = trimarm.half_targets(args)
        n = len(reads)
        with later(trimarm.GATHER_RECHECK_S + 1.0, trimarm):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                after = trimarm.half_targets(args)
        check(
            within is False and n == 0,
            "on the trackers' own set_trim, not read again within GATHER_RECHECK_S "
            "(%s, %d reads)" % (within, n),
        )
        check(
            after is True and len(reads) == 1,
            "read again after it, and on the halves' targets once it batches (%s, %d "
            "reads)" % (after, len(reads)),
        )
    finally:
        iface.read_source = real_read
        trimarm._gather.clear()


def test_node_hosts_per_process():
    broker._NODE_HOSTS[:] = []
    node_api.set_hosts([])
    host_a = "http://node-a:12048/gnss0_inject"
    host_b = "http://node-b:12048/gnss0_inject"

    def argv(hosts):
        return [
            "--signal",
            "gps_l5",
            "--trackers",
            "http://node-a:12048/gnss0_gps_l5_inject",
            "--node-hosts",
            hosts,
        ]

    reached, go = [], threading.Event()

    def running(entry, base):
        reached.append(threading.current_thread().name)
        go.wait(10.0)
        raise _Started()

    def until(cond, s=5.0):
        t_end = time.monotonic() + s
        while not cond() and time.monotonic() < t_end:
            time.sleep(0.01)

    alive = {"a", "b", "c"}
    threads = {}

    def chain(name, hosts):
        t = threading.Thread(
            target=broker_multi.run_chain,
            args=(name, argv(hosts), object(), alive),
            name=name,
            daemon=True,
        )
        threads[name] = t
        t.start()

    real = broker.resolve_prefix
    broker.resolve_prefix = running
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            chain("a", host_a)
            until(lambda: "a" in reached)
            chain("b", host_b)
            threads["b"].join(5.0)
            hosts_then = node_api.hosts()
            chain("c", host_a)
            until(lambda: "c" in reached)
            a_running = threads["a"].is_alive() and "a" in alive
            go.set()
            for t in threads.values():
                t.join(5.0)
    finally:
        go.set()
        broker.resolve_prefix = real
    check("a" in reached, "chain a starts with its --node-hosts")
    check(
        not threads["b"].is_alive()
        and "b" not in reached
        and "b" not in alive
        and "--node-hosts differs from another chain's in this process"
        in err.getvalue(),
        "chain b, naming other hosts, refuses to start (reached %r)" % reached,
    )
    check(
        a_running and hosts_then == frozenset([host_a]),
        "chain a keeps running, on its hosts (%r)" % sorted(hosts_then),
    )
    check("c" in reached, "chain c, naming a's hosts, starts")
    broker._NODE_HOSTS[:] = []
    node_api.set_hosts([])


def main():
    print("a 404 from a half's host asked by chain, and the trim targets")
    test_404_fallback_and_trim_targets()
    print("the gather check at the start")
    test_gather_at_start()
    print("--node-hosts per process")
    test_node_hosts_per_process()
    if _fails:
        print("FAILED (%d)" % len(_fails))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
