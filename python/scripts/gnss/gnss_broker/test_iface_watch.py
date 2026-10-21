"""The interface watcher (iface_watch.py) against stub kotekan REST servers, and its hooks in
transport, telem, trimarm and the publisher.

    python3 -m gnss_broker.test_iface_watch            (from python/scripts/gnss)

The servers listen on 127.0.0.1 ephemeral ports; the watcher's passes run on a clock the test
sets, so the 10 s confirmation and the 60 s poll cost no waiting.
"""
import copy
import http.server
import io
import json
import os
import shutil
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

from gnss_broker import iface, iface_watch, publish, telem, transport

# Longer than the watcher's timeout in these tests, so a "hang" reply is a timeout.
TIMEOUT_S = 0.5
HANG_S = 1.5


def golden_doc():
    return copy.deepcopy(iface.golden().doc)


def version_body(doc, commit="c0ffee"):
    entry = {
        "schema": iface.SCHEMA,
        "digest": iface.doc_digest(doc),
        "sections": iface.section_digests(doc),
    }
    return json.dumps(
        {"git_commit_hash": commit, "interfaces": {"gnss": entry}}
    ).encode()


class Stub(object):
    """A kotekan REST stand-in: replies[path] is (code, body), "hang" (no answer within the
    watcher's timeout), or absent (404). Every request is recorded as (method, path, headers,
    body)."""

    def __init__(self):
        self.replies = {}
        self.requests = []
        stub = self

        class H(http.server.BaseHTTPRequestHandler):
            def _reply(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                stub.requests.append(
                    (self.command, self.path, sorted(self.headers.items()), body)
                )
                r = stub.replies.get(self.path)
                if r == "hang":
                    time.sleep(HANG_S)
                    return
                code, data = r if r is not None else (404, b'{"code": 404}')
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _reply

            def log_message(self, *a):
                pass

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.srv.block_on_close = False
        threading.Thread(
            target=self.srv.serve_forever, args=(0.05,), daemon=True
        ).start()
        self.netloc = "127.0.0.1:%d" % self.srv.server_address[1]
        self.url = "http://" + self.netloc

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()

    def paths(self):
        return [r[1] for r in self.requests]

    def serve(self, doc, commit="c0ffee"):
        self.replies["/version"] = (200, version_body(doc, commit))
        self.replies["/interfaces/gnss"] = (200, iface.canonical(doc))

    def legacy(self, commit="4a2f19654"):
        self.replies["/version"] = (
            200,
            json.dumps({"git_commit_hash": commit}).encode(),
        )


class Fleet(unittest.TestCase):
    """Two nodes (a, b), a gather (g) and an aggregator (s). gps_l5 and gal_e5a track on both
    nodes; gps_l5 also has the search."""

    def setUp(self):
        self.a, self.b, self.g, self.s = Stub(), Stub(), Stub(), Stub()
        for st in (self.a, self.b, self.g, self.s):
            self.addCleanup(st.close)
        self.d = tempfile.mkdtemp(prefix="iface_watch_")
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.mode_file = os.path.join(self.d, "iface_mode")
        self.addCleanup(transport.set_iface_fence, frozenset())
        self.addCleanup(transport.set_iface_error_hook, None)
        self.logs, self.fences = [], []
        a, b = self.a.url, self.b.url
        self.cfg = {
            "common": {
                "rest-url": a,
                "fleet-trim-url": self.g.url + "/fleet_trim",
                "publish-port": 12060,
            },
            "chains": {
                "gps_l5": {
                    "signal": "gps_l5",
                    "detectors": self.s.url + "/gps_search",
                    "trackers": "%s/gnss{0..1}_l5_inject,%s/gnss{0..1}_l5_inject"
                    % (a, b),
                    "dll-combiners": "gnss0_l5_n2combine,%s/gnss0_l5_n2combine" % b,
                },
                "gal_e5a": {
                    "signal": "gal_e5a",
                    "trackers": "%s/gnss0_e5a_inject,%s/gnss0_e5a_inject" % (a, b),
                },
            },
        }

    def watcher(self, mode="warn", **kw):
        kw.setdefault("timeout", TIMEOUT_S)
        kw.setdefault("log", self.logs.append)
        kw.setdefault("set_fence", self.fences.append)
        w = iface_watch.from_chains(self.cfg, mode_file=self.mode_file, **kw)
        w.yaml_mode = mode
        return w

    def all_match(self):
        for st in (self.a, self.b, self.g, self.s):
            st.serve(golden_doc())

    def logged(self, text):
        return [m for m in self.logs if text in m]


class TestOrigins(Fleet):
    def test_roles_chains_and_descriptors(self):
        o = iface_watch.origins_from_chains(self.cfg)
        self.assertEqual(
            sorted(o), sorted(x.netloc for x in (self.a, self.b, self.g, self.s))
        )
        a, b, g, s = (o[x.netloc] for x in (self.a, self.b, self.g, self.s))
        self.assertEqual((a.roles, b.roles), ({"node"}, {"node"}))
        self.assertEqual((g.roles, s.roles), ({"gather"}, {"aggregator"}))
        self.assertEqual(a.node_chains, {"gps_l5", "gal_e5a"})
        self.assertEqual(
            a.descriptors, {"GPS_L5_Q", "GPS_L5_Q_NH", "GAL_E5A_Q", "GAL_E5A_Q_CS"}
        )
        self.assertEqual(s.descriptors, {"GPS_L5_Q", "GPS_L5_Q_NH"})
        self.assertEqual(s.chains, {"gps_l5"})
        self.assertEqual(g.descriptors, set())
        # No rest-url: the broker's default, where bare names and the combiner go.
        cfg = {"chains": {"x": {"signal": "gps_l5", "trackers": "track_{0..1}"}}}
        self.assertEqual(
            sorted(iface_watch.origins_from_chains(cfg)), ["localhost:12048"]
        )

    def test_settings(self):
        st = iface_watch.settings
        self.assertEqual(st({}), ("warn", iface_watch.DEFAULT_MODE_FILE))
        self.assertEqual(st({"iface": "enforce"})[0], "enforce")
        self.assertEqual(
            st({"iface": {"mode": "off", "mode-file": "/x"}}), ("off", "/x")
        )
        self.assertEqual(st({"iface": {"mode-file": None}}), ("warn", None))
        for bad in ("on", {"mode": "warn", "mood": 1}, {"mode-file": 3}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                st({"iface": bad})


class TestVerdicts(Fleet):
    def test_match_needs_no_document(self):
        self.all_match()
        w = self.watcher()
        w.run_pass(now=0.0)
        self.assertEqual(
            set(o.state for o in w.origins.values()), {iface.MATCH}, self.logs
        )
        for st in (self.a, self.b, self.g, self.s):
            self.assertEqual(st.paths(), ["/version"])
        snap = w.snapshot()
        self.assertEqual(snap["origins"][self.a.netloc]["state"], iface.MATCH)
        self.assertEqual(snap["origins"][self.a.netloc]["kotekan"], "c0ffee")
        self.assertEqual(snap["mode"], "warn")
        # Polled again at 60 s, not before.
        w.run_pass(now=59.0)
        self.assertEqual(self.a.paths(), ["/version"])
        w.run_pass(now=60.0)
        self.assertEqual(self.a.paths(), ["/version", "/version"])

    def test_legacy_and_uncomparable_are_allowed_and_warned_once(self):
        self.all_match()
        self.a.legacy()
        self.b.replies["/version"] = (
            200,
            json.dumps({"interfaces": {"gnss": {"error": "no constants"}}}).encode(),
        )
        w = self.watcher(mode="enforce")
        for t in (0.0, 60.0, 120.0):
            w.run_pass(now=t)
        self.assertEqual(w.origins[self.a.netloc].state, iface.LEGACY)
        self.assertEqual(w.origins[self.b.netloc].state, iface.UNCOMPARABLE)
        self.assertEqual(len(self.logged("WARN %s is LEGACY" % self.a.netloc)), 1)
        self.assertEqual(len(self.logged("WARN %s is UNCOMPARABLE" % self.b.netloc)), 1)
        self.assertEqual(self.fences[-1], frozenset())

    def test_a_document_that_does_not_hash_to_its_digest_is_uncomparable(self):
        self.all_match()
        doc = golden_doc()
        doc["sections"]["rest"]["new_family"] = 1
        self.a.serve(doc)
        self.a.replies["/interfaces/gnss"] = (200, iface.canonical(golden_doc()))
        w = self.watcher()
        w.run_pass(now=0.0)
        self.assertEqual(w.origins[self.a.netloc].state, iface.UNCOMPARABLE)

    def test_compatible_with_extra_fields_fetches_once(self):
        self.all_match()
        doc = golden_doc()
        doc["sections"]["rest"]["new_family"] = 1
        doc["sections"]["signals"]["GPS_L5_Q"]["new_field"] = 7
        self.a.serve(doc)
        w = self.watcher()
        w.run_pass(now=0.0)
        o = w.origins[self.a.netloc]
        self.assertEqual(o.state, iface.COMPATIBLE)
        self.assertEqual(
            [iface.path_str(p) for p in o.extras], ["signals/GPS_L5_Q/new_field"]
        )
        w.run_pass(now=60.0)
        self.assertEqual(self.a.paths(), ["/version", "/interfaces/gnss", "/version"])
        self.assertEqual(len(self.logged("is COMPATIBLE")), 1)

    def test_a_change_outside_the_scope_is_a_match(self):
        self.all_match()
        doc = golden_doc()
        doc["sections"]["signals"]["BDS_B3I"]["code_length"] += 1
        doc["sections"]["cube"]["CUBE_VERSION"] += 1
        for st in (self.a, self.s):
            st.serve(doc)
        w = self.watcher()
        w.run_pass(now=0.0)
        self.assertEqual(w.origins[self.a.netloc].state, iface.MATCH)
        self.assertEqual(w.origins[self.s.netloc].state, iface.MATCH)

    def test_mismatch_needs_two_reads_10_s_apart(self):
        self.all_match()
        w = self.watcher()
        w.run_pass(now=0.0)
        doc = golden_doc()
        doc["sections"]["record.row"]["REC_CP"]["slot"] += 1
        self.a.serve(doc, commit="badc0de")
        w.run_pass(now=60.0)
        o = w.origins[self.a.netloc]
        self.assertEqual(o.state, iface.MATCH, "one read never changes the verdict")
        self.assertTrue(o.pending)
        w.run_pass(now=65.0)
        self.assertEqual(self.a.paths().count("/version"), 2, "not read before 10 s")
        w.run_pass(now=70.0)
        self.assertEqual(o.state, iface.MISMATCH)
        err = self.logged("ERROR %s MISMATCH" % self.a.netloc)
        self.assertEqual(len(err), 1)
        g = iface.golden()
        for want in (
            "record.row/REC_CP/slot: golden %d, binary %d"
            % (g.slot("record.row", "REC_CP"), g.slot("record.row", "REC_CP") + 1),
            "kotekan badc0de",
            "pin kotekan %s" % g.commit,
        ):
            self.assertIn(want, err[0])
        # The ERROR repeats every 10 min while it lasts; the document was fetched once.
        w.run_pass(now=80.0)
        self.assertEqual(len(self.logged("ERROR %s MISMATCH" % self.a.netloc)), 1)
        w.run_pass(now=670.0)
        self.assertEqual(len(self.logged("ERROR %s MISMATCH" % self.a.netloc)), 2)
        self.assertEqual(self.a.paths().count("/interfaces/gnss"), 1)
        # The first good read ends it.
        self.a.serve(golden_doc())
        w.run_pass(now=680.0)
        self.assertEqual(o.state, iface.MATCH)

    def test_a_different_second_read_is_not_a_confirmation(self):
        self.all_match()
        w = self.watcher()
        doc = golden_doc()
        doc["sections"]["record.row"]["REC_CP"]["slot"] += 1
        self.a.serve(doc)
        w.run_pass(now=0.0)
        doc["sections"]["telem"]["TELEM_VERSION"] += 1
        self.a.serve(doc)
        w.run_pass(now=10.0)
        self.assertEqual(w.origins[self.a.netloc].state, iface_watch.UNKNOWN)
        w.run_pass(now=20.0)
        self.assertEqual(w.origins[self.a.netloc].state, iface.MISMATCH)

    def test_failed_reads_keep_the_verdict(self):
        self.all_match()
        w = self.watcher(mode="enforce")
        w.run_pass(now=0.0)
        doc = golden_doc()
        doc["sections"]["telem"]["TELEM_VERSION"] += 1
        self.b.serve(doc)
        w.run_pass(now=60.0)
        w.run_pass(now=70.0)
        a, b = w.origins[self.a.netloc], w.origins[self.b.netloc]
        self.assertEqual((a.state, b.state), (iface.MATCH, iface.MISMATCH))
        fence = self.fences[-1]
        self.assertEqual(fence, frozenset({(self.b.netloc, None)}))
        t = 70.0
        for bad in (
            "hang",
            (404, b"{}"),
            (500, b"oops"),
            (200, b"not json {"),
            (200, b"[1, 2]"),
            (503, b'{"code": 503}'),
        ):
            for st in (self.a, self.b):
                st.replies["/version"] = bad
            t += 60.0
            w.run_pass(now=t)
            self.assertEqual((a.state, b.state), (iface.MATCH, iface.MISMATCH), bad)
            self.assertIsNotNone(a.last_error, bad)
            self.assertEqual(self.fences[-1], fence, "%r moved the fence" % (bad,))
        # A document that cannot be fetched is a failed read too.
        self.a.serve(doc)
        self.a.replies["/interfaces/gnss"] = (500, b"oops")
        w.run_pass(now=t + 60.0)
        self.assertEqual(a.state, iface.MATCH)

    def test_odd_one_out(self):
        self.all_match()
        doc = golden_doc()
        doc["sections"]["record.row"]["REC_CP"]["slot"] += 1
        self.a.serve(doc)
        self.b.serve(doc)
        w = self.watcher()
        w.run_pass(now=0.0)
        w.run_pass(now=10.0)
        self.assertTrue(w.odd_one_out)
        self.assertEqual(
            len(
                self.logged("broker pin is the odd one out (pin %s)" % w.golden.commit)
            ),
            1,
        )
        self.assertTrue(w.snapshot()["odd_one_out"])

    def test_an_exception_at_one_origin_is_isolated(self):
        self.all_match()
        w = self.watcher()
        real = w._compare

        def flaky(o, entry):
            if o.netloc == self.a.netloc:
                raise RuntimeError("boom")
            return real(o, entry)

        w._compare = flaky
        w.run_pass(now=0.0)
        self.assertEqual(w.origins[self.a.netloc].state, iface_watch.UNKNOWN)
        for st in (self.b, self.g, self.s):
            self.assertEqual(w.origins[st.netloc].state, iface.MATCH)
        self.assertTrue(self.logged("watcher error"))
        self.assertIn("boom", w.snapshot()["origins"][self.a.netloc]["last_error"])

    def test_check_fleet_decides_on_one_read(self):
        self.all_match()
        doc = golden_doc()
        doc["sections"]["record.row"]["REC_CP"]["slot"] += 1
        self.b.serve(doc)
        self.s.legacy()
        lines, bad = iface_watch.check_fleet(self.cfg, timeout=TIMEOUT_S)
        self.assertTrue(bad)
        row = {ln.split()[0]: ln for ln in lines[2:]}
        self.assertIn(" MISMATCH ", row[self.b.netloc])
        self.assertIn("record.row/REC_CP/slot", row[self.b.netloc])
        self.assertIn(" LEGACY ", row[self.s.netloc])
        self.assertIn(" MATCH ", row[self.a.netloc])

    def test_check_fleet_columns_fit_an_origin_with_every_role(self):
        self.all_match()
        self.cfg["chains"]["gps_l5"]["detectors"] = "gps_search"  # on rest-url, node a
        self.cfg["common"]["fleet-trim-url"] = "fleet_trim"  # and the gather too
        lines, bad = iface_watch.check_fleet(self.cfg, timeout=TIMEOUT_S)
        self.assertFalse(bad, lines)
        head, rows = lines[1], lines[2:]
        self.assertTrue(
            any(" aggregator,gather,node " in ln for ln in rows), "\n".join(lines)
        )
        # Every row's verdict and kotekan start under their headings.
        for col in ("verdict", "kotekan"):
            at = head.index(col)
            for ln in rows:
                self.assertEqual(
                    (ln[at - 1], ln[at] != " "),
                    (" ", True),
                    "%s misaligned:\n%s" % (col, "\n".join(lines)),
                )

    def test_check_fleet_cli(self):
        self.all_match()
        path = os.path.join(self.d, "chains.yaml")
        with open(path, "w") as fh:
            json.dump(self.cfg, fh)  # JSON is YAML
        out = io.StringIO()
        with redirect_stdout(out):
            rc = iface.main(["check-fleet", path, "--timeout", str(TIMEOUT_S)])
        self.assertEqual(rc, 0, out.getvalue())
        self.assertEqual(out.getvalue().count(" MATCH "), 4, out.getvalue())
        self.cfg["common"]["iface"] = "loud"
        with open(path, "w") as fh:
            json.dump(self.cfg, fh)
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            rc = iface.main(["check-fleet", path])
        self.assertEqual(rc, 2)
        self.assertIn("'loud' is not one of", err.getvalue())


class TestModesAndFence(Fleet):
    def mismatch_everywhere(self, section, key):
        doc = golden_doc()
        v = doc["sections"][section][key]
        if isinstance(v, dict):
            v["slot"] += 1
        else:
            doc["sections"][section][key] = v + 1
        for st in (self.a, self.b, self.g, self.s):
            st.serve(doc)

    def test_warn_never_fences(self):
        self.mismatch_everywhere("telem", "TELEM_VERSION")
        w = self.watcher(mode="warn")
        w.run_pass(now=0.0)
        w.run_pass(now=10.0)
        self.assertEqual(w.origins[self.a.netloc].state, iface.MISMATCH)
        self.assertEqual(set(self.fences), {frozenset()})

    def test_enforce_fences_nodes_never_the_gather_or_aggregator(self):
        # telem is in the node and gather scopes; signals in the node and aggregator scopes.
        self.mismatch_everywhere("telem", "TELEM_VERSION")
        doc = golden_doc()
        doc["sections"]["rest"]["search"] += 1
        self.s.serve(doc)
        w = self.watcher(mode="enforce")
        w.run_pass(now=0.0)
        w.run_pass(now=10.0)
        for st in (self.a, self.b, self.g, self.s):
            self.assertEqual(w.origins[st.netloc].state, iface.MISMATCH)
        self.assertEqual(
            self.fences[-1], frozenset({(self.a.netloc, None), (self.b.netloc, None)}),
        )
        snap = w.snapshot()["origins"]
        self.assertEqual(snap[self.a.netloc]["fenced"], "origin")
        self.assertIsNone(snap[self.g.netloc]["fenced"])
        self.assertIsNone(snap[self.s.netloc]["fenced"])

    def test_a_node_that_also_serves_the_search_is_never_fenced(self):
        self.cfg["chains"]["gps_l5"]["detectors"] = "gps_search"  # on rest-url, node a
        self.mismatch_everywhere("telem", "TELEM_VERSION")
        w = self.watcher(mode="enforce")
        self.assertEqual(w.origins[self.a.netloc].roles, {"node", "aggregator"})
        w.run_pass(now=0.0)
        w.run_pass(now=10.0)
        self.assertEqual(w.origins[self.a.netloc].state, iface.MISMATCH)
        self.assertEqual(self.fences[-1], frozenset({(self.b.netloc, None)}))

    def test_a_signals_failure_fences_only_its_chains(self):
        self.all_match()
        doc = golden_doc()
        doc["sections"]["signals"]["GAL_E5A_Q_CS"]["code_length"] += 1
        self.a.serve(doc)
        w = self.watcher(mode="enforce")
        w.run_pass(now=0.0)
        w.run_pass(now=10.0)
        self.assertEqual(self.fences[-1], frozenset({(self.a.netloc, "gal_e5a")}))
        self.assertEqual(
            w.chain_summary("gal_e5a")["fenced"], [self.a.netloc], w.chain_summary
        )
        # The summary's verdicts are the origins'; only the fence is per chain.
        self.assertNotIn("fenced", w.chain_summary("gps_l5"))
        self.assertEqual(w.chain_summary("gps_l5")["worst"], iface.MISMATCH)

    def test_the_mode_file_switches_without_a_restart(self):
        self.mismatch_everywhere("telem", "TELEM_VERSION")
        w = self.watcher(mode="warn")
        w.run_pass(now=0.0)
        w.run_pass(now=10.0)
        self.assertEqual(self.fences[-1], frozenset())
        with open(self.mode_file, "w") as fh:
            fh.write("enforce\n")
        w.run_pass(now=11.0)
        self.assertEqual(w.mode, "enforce")
        self.assertEqual(len(self.fences[-1]), 2)
        with open(self.mode_file, "w") as fh:
            fh.write("bogus\n")
        w.run_pass(now=12.0)
        self.assertEqual(w.mode, "warn", "a bad mode file falls back to the yaml")
        self.assertTrue(self.logged("WARN mode file"))
        with open(self.mode_file, "w") as fh:
            fh.write("off\n")
        n = len(self.a.requests)
        w.run_pass(now=100.0)
        self.assertEqual(len(self.a.requests), n, "off reads nothing")
        self.assertEqual(self.fences[-1], frozenset())
        os.unlink(self.mode_file)
        w.run_pass(now=101.0)
        self.assertEqual(w.mode, "warn")

    def test_stale_alarm(self):
        self.all_match()
        w = self.watcher()
        self.assertIsNone(w.stale_alarm(now=1e9))
        w.run_pass(now=1000.0)
        self.assertIsNone(w.stale_alarm(now=1599.0))
        self.assertIn("600 s old", w.stale_alarm(now=1600.0))
        self.assertIsNone(w.stale_alarm(now=1700.0))
        self.assertIsNotNone(w.stale_alarm(now=2200.0))


class TestHooks(Fleet):
    def calls(self, url):
        """The requests a fixed sequence of transport calls puts on the wire at self.a."""
        n = len(self.a.requests)
        self.a.replies["/gnss0_l5_inject/get_status"] = (200, b'{"ok": 1}')
        self.a.replies["/gnss0_l5_inject/set_seeds"] = (200, b"{}")
        transport._TR.get(url + "/gnss0_l5_inject/get_status", 2.0)
        transport._TR.post(
            url + "/gnss0_l5_inject/set_seeds", {"b": [1, 2], "a": 1}, 2.0
        )
        transport._TR.get(url + "/version", 2.0)
        return self.a.requests[n:]

    def test_warn_and_off_send_the_same_bytes(self):
        """The requests are byte for byte the same with no watcher, with a watcher's error hook
        and an empty fence (warn, off), and with a fence on another origin."""
        self.a.replies["/version"] = (200, b"{}")
        base = self.calls(self.a.url)
        w = self.watcher()
        transport.set_iface_error_hook(w.mark_dirty)
        self.assertEqual(self.calls(self.a.url), base)
        transport.set_iface_fence({("elsewhere:1", None), (self.b.netloc, "gps_l5")})
        self.assertEqual(self.calls(self.a.url), base)
        self.assertEqual(len(base), 3)

    def test_a_fenced_origin_is_refused_like_a_down_node(self):
        self.a.replies["/version"] = (200, b"{}")
        self.a.replies["/telescope/time0_ns"] = (200, b'{"time0_ns": 1}')
        hits = []
        transport.set_iface_error_hook(hits.append)
        transport.set_iface_fence({(self.a.netloc, None)})
        n = len(self.a.requests)
        with self.assertRaises(urllib.error.URLError) as e:
            transport._TR.get(self.a.url + "/gnss0_l5_inject/get_status", 2.0)
        self.assertIsInstance(e.exception, transport.IfaceRefused)
        with self.assertRaises(transport.IfaceRefused):
            transport._TR.post(self.a.url + "/gnss0_l5_inject/set_seeds", {}, 2.0)
        self.assertEqual(len(self.a.requests), n, "nothing reached the node")
        self.assertEqual(hits, [], "a refusal is not a transport failure")
        # Kotekan's own paths stay open, the time0 read among them.
        self.assertEqual(transport._TR.get(self.a.url + "/version", 2.0), {})
        self.assertEqual(
            transport._TR.get(self.a.url + "/telescope/time0_ns", 2.0), {"time0_ns": 1}
        )
        # A chain fence covers only that chain's thread.
        transport.set_iface_fence({(self.a.netloc, "gal_e5a")})
        url = self.a.url + "/gnss0_e5a_inject/get_status"
        transport.set_log_tag("gps_l5")
        self.addCleanup(transport.set_log_tag, "")
        self.assertFalse(transport.iface_fenced(url))
        transport.set_log_tag("gal_e5a")
        self.assertTrue(transport.iface_fenced(url))

    def test_a_failed_request_makes_its_origin_dirty(self):
        self.all_match()
        w = self.watcher()
        transport.set_iface_error_hook(w.mark_dirty)
        w.run_pass(now=0.0)
        self.a.replies["/gnss0_l5_inject/get_status"] = (500, b"oops")
        with self.assertRaises(urllib.error.HTTPError):
            transport._TR.get(self.a.url + "/gnss0_l5_inject/get_status", 2.0)
        w.mark_dirty("http://not-an-origin:1/x")  # ignored
        w.run_pass(now=5.0)
        self.assertEqual(self.a.paths().count("/version"), 1)
        w.run_pass(now=10.0)
        self.assertEqual(self.a.paths().count("/version"), 2, "re-read at 10 s, not 60")
        self.assertEqual(self.b.paths().count("/version"), 1)

    def test_telemetry_and_trim_targets(self):
        trackers = [
            "http://cx19:12048/gnss0_l5_inject",
            "http://cx27:12048/gnss0_l5_inject",
        ]
        self.assertIs(transport.iface_unfenced(trackers), trackers)
        tc = telem.TelemClient()

        def frame(inst, chain="gps_l5"):
            return types.SimpleNamespace(chain=chain, inst=inst, rx=0.0, seq=1, win=5)

        tc._store_frame(frame("cx19.0"))
        self.assertEqual((tc.frames, tc.iface_dropped), (1, 0))
        transport.set_iface_fence({("cx19:12048", None), ("cx27:12048", "gal_e5a")})
        transport.set_log_tag("gps_l5")
        self.addCleanup(transport.set_log_tag, "")
        self.assertEqual(transport.iface_unfenced(trackers), trackers[1:])
        tc._store_frame(frame("cx19.1"))
        tc._store_frame(frame("cx27.0"))
        tc._store_frame(frame("cx27.1", chain="gal_e5a"))
        self.assertEqual((tc.frames, tc.iface_dropped), (2, 2))


class TestPublisher(Fleet):
    def get(self, pub, path):
        port = pub._srv.server_address[1]
        with urllib.request.urlopen(
            "http://127.0.0.1:%d%s" % (port, path), timeout=5
        ) as r:
            return r.read()

    def test_get_iface_and_the_chains_summary(self):
        pub = publish.FleetPublisher(0, lambda m: None)
        self.addCleanup(pub._srv.server_close)
        self.addCleanup(pub._srv.shutdown)
        for c in ("gps_l5", "gal_e5a"):
            pub.register(c, signal=c, band="L5", meta={"constellation": "G"})
        before = self.get(pub, "/get_chains")
        self.assertIsNone(json.loads(self.get(pub, "/get_iface"))["mode"])
        self.all_match()
        w = self.watcher()
        w.run_pass(now=0.0)
        pub.iface = w
        after = json.loads(self.get(pub, "/get_chains"))
        self.assertEqual(
            [{k: v for k, v in d.items() if k != "iface"} for d in after],
            json.loads(before),
            "the chains list is unchanged apart from the added key",
        )
        self.assertEqual(after[0]["iface"]["worst"], iface.MATCH)
        self.assertEqual(after[0]["iface"]["states"], {iface.MATCH: 4})
        # wait_for_broker_chains.sh counts the lines holding "chain" after splitting on ','.
        count = lambda b: sum(
            '"chain"' in x for x in b.decode().split(",")
        )  # noqa: E731
        self.assertEqual(count(before), 2)
        self.assertEqual(count(self.get(pub, "/get_chains")), 2)
        snap = json.loads(self.get(pub, "/get_iface"))
        self.assertEqual(snap["golden"], iface.golden().digest)
        self.assertEqual(snap["origins"][self.a.netloc]["state"], iface.MATCH)
        self.assertIsNotNone(snap["age_s"])


if __name__ == "__main__":
    unittest.main()
