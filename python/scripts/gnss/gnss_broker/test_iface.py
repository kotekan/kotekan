"""The interface module (iface.py): canonical form and digests against the C++ registry's known
answers, the vendored golden against its pin and against kotekan's, and the comparison rules.

    python3 -m gnss_broker.test_iface            (from python/scripts/gnss)

The golden's copy in kotekan (lib/stages/gnss/gnss_interface.json) is found as kotekan_src
says. Without a kotekan tree holding it that check skips, and fails under
GNSS_REQUIRE_KOTEKAN=1.
"""
import copy
import http.server
import io
import json
import os
import shutil
import struct
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout

import kotekan_src
from gnss_broker import iface

# The C++ registry's own known answers (tests/boost/test_interfaceRegistry.cpp).
CPP_NESTED = {
    "z": {"b": [1, -2, 9223372036854775807, -9223372036854775808], "a": True},
    "A": 'quote " and backslash \\ and /slash',
    "m": {"empty_obj": {}, "empty_arr": [], "f": False, "u": 4294967295},
    "_": [[], [{}], "~"],
}
CPP_NESTED_BYTES = (
    b'{"A":"quote \\" and backslash \\\\ and /slash","_":[[],[{}],"~"],'
    b'"m":{"empty_arr":[],"empty_obj":{},"f":false,"u":4294967295},'
    b'"z":{"a":true,"b":[1,-2,9223372036854775807,-9223372036854775808]}}'
)
CPP_EXAMPLE = {
    "interface": "example",
    "schema": 3,
    "sections": {
        "zeta": {"b": -1, "a": [1, 2, 3]},
        "alpha": {"name": 'L5 "Q" \\', "flag": True, "carrier_hz": "1176450000"},
    },
}
CPP_DOUBLES = (
    (1176.45e6, "1176450000"),
    (1e-3, "0.001"),
    (0.1, "0.10000000000000001"),
    (-2.5e-12, "-2.4999999999999998e-12"),
    (5.12e-6, "5.1200000000000001e-06"),
    (1e22, "1e+22"),
    (-0.0, "-0"),
)
# What kotekan served for the golden at fe1936269 (/version and GET /interfaces/gnss of the
# build, recomputed there in Python), by document digest.
KOTEKAN_SERVED = {
    "fnv1a64:d952a1895d4b1588": {
        "cube": "fnv1a64:9d8189ee49526cd3",
        "enums": "fnv1a64:ce7f87e79a2838f6",
        "gpu_chain": "fnv1a64:1f5f16e80dbb5c93",
        "metadata": "fnv1a64:ff0b089558622884",
        "record.chan": "fnv1a64:df5e8f24a8e017fe",
        "record.cmb": "fnv1a64:1cb9fde616bbf0dc",
        "record.elem": "fnv1a64:b61468c5a83782d3",
        "record.row": "fnv1a64:436522cd6437ce2d",
        "rest": "fnv1a64:9cde53faca83ad60",
        "signals": "fnv1a64:87d44429c62b7ddb",
        "telem": "fnv1a64:4fc6ec77e47425b8",
    }
}


def golden_doc():
    """A copy of the golden document, free to change."""
    return copy.deepcopy(iface.golden().doc)


def entry_of(doc):
    """doc as `kotekan --interfaces-json` prints it, under "gnss"."""
    return {
        "schema": doc["schema"],
        "digest": iface.doc_digest(doc),
        "sections": iface.section_digests(doc),
        "doc": doc,
    }


def run_cli(*argv):
    """(exit status, stdout) of `python3 -m gnss_broker.iface argv...`."""
    out = io.StringIO()
    with redirect_stdout(out):
        rc = iface.main(list(argv))
    return rc, out.getvalue()


class TestCanonical(unittest.TestCase):
    def test_fnv_known_answers(self):
        self.assertEqual(iface.digest(b""), "fnv1a64:cbf29ce484222325")
        self.assertEqual(iface.digest(b"a"), "fnv1a64:af63dc4c8601ec8c")
        self.assertEqual(iface.digest(b"foobar"), "fnv1a64:85944171f73967e8")

    def test_canonical_bytes_are_the_registrys(self):
        self.assertEqual(iface.canonical(CPP_NESTED), CPP_NESTED_BYTES)

    def test_document_and_section_digests_are_the_registrys(self):
        self.assertEqual(iface.doc_digest(CPP_EXAMPLE), "fnv1a64:1dacc0ea03bc8e79")
        self.assertEqual(
            iface.section_digests(CPP_EXAMPLE),
            {"alpha": "fnv1a64:3c5995dc6b90b081", "zeta": "fnv1a64:9a999fafd7525af6"},
        )
        self.assertEqual(
            iface.canonical(CPP_EXAMPLE),
            b'{"interface":"example","schema":3,"sections":{"alpha":'
            b'{"carrier_hz":"1176450000","flag":true,"name":"L5 \\"Q\\" \\\\"},'
            b'"zeta":{"a":[1,2,3],"b":-1}}}',
        )

    def test_doubles_are_the_registrys_and_read_back(self):
        for x, s in CPP_DOUBLES:
            self.assertEqual(iface.canonical_double(x), s)
            self.assertEqual(iface.dbl(s), x)
        for bad in ("1e-3", "0.0010", "1176.45e6", 0.001, 1):
            with self.assertRaises(ValueError, msg=repr(bad)):
                iface.dbl(bad)

    def test_the_form_refuses_what_the_registry_refuses(self):
        for bad in (
            {"a": {"b": 1.0}},
            {"a": None},
            {"a": 1 << 63},
            {"a": -(1 << 63) - 1},
            {"a": "caf\u00e9"},
            {"caf\u00e9": 1},
            {"a": ["line\n"]},
            {"a\x7f": 1},
            {"a": (1, 2)},
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                iface.canonical(bad)
        iface.canonical({"a": (1 << 63) - 1, "b": -(1 << 63)})
        with self.assertRaises(ValueError) as cm:
            iface.canonical({"s": {"x": [0, 2.5]}})
        self.assertIn("'/s/x/1'", str(cm.exception))


class TestGolden(unittest.TestCase):
    def test_golden_loads_against_its_pin(self):
        g = iface.golden()
        self.assertEqual(g.digest, g.pin["digest"])
        self.assertEqual(sorted(g.sections), sorted(iface.SECTIONS))
        self.assertRegex(g.commit, r"^[0-9a-f]{40}$")
        self.assertEqual(g.pin["source"], iface.KOTEKAN_GOLDEN)
        with open(iface.GOLDEN_PATH) as fh:
            self.assertEqual(fh.read(), iface.format_golden(g.doc), "golden form")

    def test_digests_are_the_ones_kotekan_served(self):
        g = iface.golden()
        served = KOTEKAN_SERVED.get(g.digest)
        if served is None:
            self.skipTest("no served digests recorded for %s" % g.digest)
        self.assertEqual(g.section_digests, served)

    def test_vendored_golden_is_kotekans(self):
        path = None
        if kotekan_src.SRC:
            path = kotekan_src.src(iface.KOTEKAN_GOLDEN)
        if not path or not os.path.exists(path):
            msg = "no kotekan tree holding %s (KOTEKAN_SRC or beside this checkout)" % (
                iface.KOTEKAN_GOLDEN
            )
            if os.environ.get("GNSS_REQUIRE_KOTEKAN"):
                self.fail(msg)
            self.skipTest(msg)
        with open(path, "rb") as a, open(iface.GOLDEN_PATH, "rb") as b:
            self.assertEqual(
                a.read(),
                b.read(),
                "%s differs from the vendored golden: python3 -m gnss_broker.iface sync "
                "--from %s" % (path, kotekan_src.SRC),
            )

    def test_a_golden_that_is_not_its_pin_fails_at_load(self):
        d = tempfile.mkdtemp(prefix="iface_golden_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        doc = golden_doc()
        doc["sections"]["telem"]["TELEM_VERSION"] += 1
        gp, pp = os.path.join(d, "g.json"), os.path.join(d, "p.json")
        with open(gp, "w") as fh:
            fh.write(iface.format_golden(doc))
        with open(pp, "w") as fh:
            json.dump(iface.golden().pin, fh)
        with self.assertRaises(iface.GoldenError) as cm:
            iface.load_golden(gp, pp)
        self.assertIn("pins %s" % iface.golden().digest, str(cm.exception))
        with self.assertRaises(iface.GoldenError):
            iface.load_golden(os.path.join(d, "absent.json"), pp)
        del doc["sections"]["enums"]
        with open(gp, "w") as fh:
            fh.write(iface.format_golden(doc))
        with self.assertRaises(iface.GoldenError) as cm:
            iface.load_golden(gp, pp)
        self.assertIn("no section enums", str(cm.exception))

    def test_telem_header_format_is_the_one_the_broker_had(self):
        # The literal telem.py carried before it read the golden.
        self.assertEqual(
            iface.golden().struct_format("telem", "TelemHeader"),
            "<IHHHHHHIIQQqdIHH16s16s8H",
        )
        for name in ("FrameHdr", "PrnCtl"):
            fmt = iface.golden().struct_format("gpu_chain", name)
            self.assertEqual(
                struct.calcsize(fmt),
                iface.golden().sections["gpu_chain"][name]["size"],
            )

    def test_a_field_table_with_a_gap_or_a_wrong_size_is_refused(self):
        ok = {"size": 12, "fields": [["a", 0, 8, "i64"], ["b", 8, 4, "u32"]]}
        self.assertEqual(iface.struct_format(ok), "<qI")
        for bad in (
            {"size": 12, "fields": [["a", 0, 8, "i64"], ["b", 9, 4, "u32"]]},
            {"size": 12, "fields": [["a", 0, 8, "i64"], ["b", 8, 4, "u16"]]},
            {"size": 16, "fields": [["a", 0, 8, "i64"], ["b", 8, 4, "u32"]]},
            {"size": 8, "fields": [["a", 0, 8, "q64"]]},
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                iface.struct_format(bad)


class TestCompare(unittest.TestCase):
    def test_same_is_match(self):
        c = iface.compare(iface.golden(), golden_doc())
        self.assertEqual((c.state, c.fails, c.extras), (iface.MATCH, [], []))

    def test_an_extra_field_is_compatible(self):
        doc = golden_doc()
        doc["sections"]["signals"]["GPS_L5_I_NH"] = dict(
            doc["sections"]["signals"]["GPS_L5_Q_NH"]
        )
        doc["sections"]["telem"]["TELEM_NEW"] = 1
        doc["sections"]["telem"]["frame_bytes"].append(
            {"bytes": 240, "n_cols": 0, "n_prn": 1, "n_rec": 1}
        )
        c = iface.compare(iface.golden(), doc)
        self.assertEqual(c.state, iface.COMPATIBLE)
        self.assertEqual(
            sorted(iface.path_str(p) for p in c.extras),
            ["signals/GPS_L5_I_NH", "telem/TELEM_NEW", "telem/frame_bytes/3"],
        )
        self.assertTrue(iface.summary(c).startswith("compatible (extra: "))

    def test_a_changed_leaf_is_a_mismatch_naming_both_values(self):
        doc = golden_doc()
        doc["sections"]["record.row"]["RECORD_FLOATS"] = 30
        c = iface.compare(iface.golden(), doc)
        self.assertEqual(c.state, iface.MISMATCH)
        self.assertEqual(c.fails, [(("record.row", "RECORD_FLOATS"), 32, 30)])
        self.assertEqual(
            iface.describe_fails(c.fails),
            ["record.row/RECORD_FLOATS: golden 32, binary 30"],
        )

    def test_a_missing_leaf_is_a_mismatch(self):
        doc = golden_doc()
        del doc["sections"]["record.row"]["REC_PHI_DDOP"]
        c = iface.compare(iface.golden(), doc)
        self.assertEqual(c.state, iface.MISMATCH)
        self.assertEqual(
            [iface.path_str(p) for p, _, _ in c.fails],
            ["record.row/REC_PHI_DDOP/meaning", "record.row/REC_PHI_DDOP/slot"],
        )
        self.assertIn("binary missing", iface.describe_fails(c.fails)[0])

    def test_types_are_strict(self):
        doc = golden_doc()
        doc["sections"]["signals"]["GPS_L5_Q"]["time_multiplexed"] = 0
        doc["sections"]["telem"]["TELEM_VERSION"] = "8"
        c = iface.compare(iface.golden(), doc)
        self.assertEqual(
            sorted(iface.path_str(p) for p, _, _ in c.fails),
            ["signals/GPS_L5_Q/time_multiplexed", "telem/TELEM_VERSION"],
        )

    def test_a_role_sees_only_its_scope(self):
        g = iface.golden()
        cmb = golden_doc()
        cmb["sections"]["record.cmb"]["CMB_DEEP"]["slot"] = 9
        search = golden_doc()
        search["sections"]["rest"]["search"] = 2
        meta = golden_doc()
        meta["sections"]["metadata"]["GnssChanMetadata"]["size"] = 16
        cube = golden_doc()
        cube["sections"]["cube"]["CUBE_VERSION"] = 4
        for doc, want in (
            (cmb, {"node": "MATCH", "gather": "MATCH", "aggregator": "MATCH"},),
            (search, {"node": "MATCH", "gather": "MATCH", "aggregator": "MISMATCH"},),
            (meta, {"node": "MATCH", "gather": "MISMATCH", "cubearch": "MISMATCH"},),
            (cube, {"node": "MATCH", "gather": "MATCH", "cubearch": "MISMATCH"}),
        ):
            got = {r: iface.compare(g, doc, role=r).state for r in want}
            self.assertEqual(got, want)
            self.assertEqual(iface.compare(g, doc).state, iface.MISMATCH)

    def test_several_roles_compare_the_union_of_their_scopes(self):
        g = iface.golden()
        paths = iface.scope(["node", "gather"], ["GPS_L5_Q"])
        self.assertEqual(len(paths), len(set(paths)), "each path once")
        self.assertEqual(
            set(paths),
            set(iface.scope("node", ["GPS_L5_Q"]))
            | set(iface.scope("gather", ["GPS_L5_Q"])),
        )
        search = golden_doc()
        search["sections"]["rest"]["search"] = 2
        c = iface.compare(g, search, role=["node", "aggregator"], signals=["GPS_L5_Q"])
        self.assertEqual([iface.path_str(p) for p, _, _ in c.fails], ["rest/search"])
        with self.assertRaises(ValueError):
            iface.scope(["node", "nobody"])

    def test_the_signals_scope_is_the_chains_descriptors(self):
        g = iface.golden()
        doc = golden_doc()
        doc["sections"]["signals"]["GAL_E6_C"]["carrier_hz"] = "1278760000"
        del doc["sections"]["signals"]["SBAS_L5_Q_MC"]
        l5 = ["GPS_L5_Q", "GPS_L5_Q_NH"]
        self.assertEqual(iface.compare(g, doc, "node", signals=l5).state, iface.MATCH)
        c = iface.compare(g, doc, "node")
        self.assertEqual(c.state, iface.MISMATCH)
        bad = sorted(set(p[1] for p, _, _ in c.fails))
        self.assertEqual(bad, ["GAL_E6_C", "SBAS_L5_Q_MC"])
        c = iface.compare(g, doc, "aggregator", signals=["SBAS_L5_Q", "SBAS_L5_Q_MC"])
        self.assertEqual(c.state, iface.MISMATCH)
        self.assertTrue(all(p[1] == "SBAS_L5_Q_MC" for p, _, _ in c.fails))
        with self.assertRaises(ValueError):
            iface.compare(g, doc, "node", signals=["NO_SUCH"])
        with self.assertRaises(ValueError):
            iface.compare(g, doc, "viewer")

    def test_another_schema_or_a_float_is_uncomparable(self):
        for change in (
            lambda d: d.update(schema=2),
            lambda d: d.update(interface="other"),
            lambda d: d["sections"]["telem"].update(TELEM_VERSION=8.0),
            lambda d: d.update(extra=1),
        ):
            doc = golden_doc()
            change(doc)
            self.assertEqual(
                iface.compare(iface.golden(), doc).state, iface.UNCOMPARABLE
            )


class TestSourcesAndCli(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="iface_cli_")
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)

    def put(self, name, obj):
        path = os.path.join(self.d, name)
        with open(path, "w") as fh:
            fh.write(obj if isinstance(obj, str) else json.dumps(obj))
        return path

    def test_interfaces_json_output_is_read_and_checked(self):
        doc = golden_doc()
        s = iface.read_source(self.put("ok.json", {"gnss": entry_of(doc)}))
        self.assertEqual(iface.doc_digest(s.doc), iface.golden().digest)
        lie = entry_of(doc)
        lie["digest"] = "fnv1a64:0000000000000000"
        sec = entry_of(doc)
        sec["sections"]["telem"] = "fnv1a64:0000000000000000"
        for name, obj, exc in (
            ("lie.json", {"gnss": lie}, iface.Uncomparable),
            ("sec.json", {"gnss": sec}, iface.Uncomparable),
            ("err.json", {"gnss": {"error": "no constants"}}, iface.Uncomparable),
            ("none.json", {}, iface.Legacy),
            ("empty.json", "", iface.Legacy),
            ("old.json", "Invalid option, run with -h to see options", iface.Legacy),
            ("junk.json", "kotekan: unrecognized option", iface.SourceError),
        ):
            with self.assertRaises(exc, msg=name):
                iface.read_source(self.put(name, obj))

    def test_a_url_is_read_from_version_and_interfaces(self):
        doc = golden_doc()
        body = iface.canonical(doc)
        version = {"git_commit_hash": "abc", "interfaces": {"gnss": entry_of(doc)}}
        del version["interfaces"]["gnss"]["doc"]
        replies = {}

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                code, data = replies.get(self.path, (404, b"{}"))
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        url = "http://127.0.0.1:%d" % srv.server_address[1]

        replies["/version"] = (200, json.dumps(version).encode())
        replies["/interfaces/gnss"] = (200, body)
        s = iface.read_source(url)
        self.assertEqual((s.doc, s.commit), (doc, "abc"))
        replies["/interfaces/gnss"] = (200, json.dumps(doc, indent=1).encode())
        with self.assertRaises(iface.Uncomparable):
            iface.read_source(url)  # served, but not canonical
        replies["/version"] = (200, json.dumps({"git_commit_hash": "old"}).encode())
        with self.assertRaises(iface.Legacy):
            iface.read_source(url)
        replies["/version"] = (500, b"oops")
        with self.assertRaises(iface.SourceError):
            iface.read_source(url)

    def test_diff_says_same_compatible_or_flag_day(self):
        doc = golden_doc()
        a = self.put("a.json", doc)
        doc["sections"]["rest"]["new_family"] = 1
        b = self.put("b.json", doc)
        doc["sections"]["telem"]["TELEM_VERSION"] = 9
        c = self.put("c.json", doc)
        self.assertEqual(run_cli("diff", "golden", a), (0, "same\n"))
        rc, out = run_cli("diff", a, b)
        self.assertEqual((rc, out), (0, "compatible (extra: rest/new_family)\n"))
        rc, out = run_cli("diff", "golden", c)
        self.assertEqual(rc, 1)
        self.assertEqual(out, "flag day: telem/TELEM_VERSION: A 8, B 9\n")
        # Neither change is in the aggregator's scope (signals, rest/search).
        self.assertEqual(
            run_cli("diff", "golden", c, "--role", "aggregator"), (0, "same\n")
        )

    def test_preflight(self):
        g = iface.golden()
        good = self.put("good.json", {"gnss": entry_of(golden_doc())})
        doc = golden_doc()
        doc["sections"]["record.row"]["REC_CP"]["slot"] = 3
        bad = self.put("bad.json", {"gnss": entry_of(doc)})
        doc = golden_doc()
        doc["sections"]["cube"]["CUBE_HEADER_BYTES"] = 160
        cube = self.put("cube.json", {"gnss": entry_of(doc)})
        doc = golden_doc()
        del doc["sections"]["signals"]["SBAS_L5_Q_MC"]
        nosbas = self.put("nosbas.json", {"gnss": entry_of(doc)})
        legacy = self.put("legacy.json", "")
        body = "    signal: GPS_L5_Q_NH\n    signal: SBAS_L5_Q_MC\n"
        stamped = self.put(
            "stamped.yaml", "# GENERATED\n%s\nx:\n%s" % (iface.stamp_line(g), body)
        )
        unstamped = self.put("unstamped.yaml", "# GENERATED\nx:\n" + body)

        def pf(src, *extra):
            return run_cli("preflight", "--role", "node", "--from", src, *extra)

        rc, out = pf(good, "--config", stamped)
        self.assertEqual(rc, 0, out)
        self.assertIn("MATCH", out)
        self.assertIn("config stamp agrees", out)
        rc, out = pf(bad)
        self.assertEqual(rc, 1, out)
        self.assertIn("ERROR record.row/REC_CP/slot: golden 2, binary 3", out)
        # cube is a sizing section the node role does not compare: the stamp catches it.
        self.assertEqual(pf(cube)[0], 0)
        rc, out = pf(cube, "--config", stamped)
        self.assertEqual(rc, 1, out)
        self.assertIn("section cube", out)
        rc, out = pf(cube, "--config", unstamped)
        self.assertEqual(rc, 0, out)
        self.assertIn("carries no gnss_iface stamp", out)
        # The archiver's own scope holds the cube.
        rc, out = run_cli("preflight", "--role", "cubearch", "--from", cube)
        self.assertEqual(rc, 1, out)
        self.assertIn("ERROR cube/CUBE_HEADER_BYTES: golden 152, binary 160", out)
        self.assertIn("REFUSED", out)
        rc, out = run_cli("preflight", "--role", "cubearch", "--from", good)
        self.assertEqual(rc, 0, out)
        # The 10-20 failure: the config names a descriptor the binary does not resolve.
        rc, out = pf(nosbas, "--config", unstamped)
        self.assertEqual(rc, 1, out)
        self.assertIn("names descriptor SBAS_L5_Q_MC, which this binary does not", out)
        self.assertEqual(pf(nosbas, "--signals", "GPS_L5_Q,GPS_L5_Q_NH")[0], 0)
        rc, out = pf(legacy, "--config", stamped)
        self.assertEqual(rc, 0, out)
        self.assertIn("WARN LEGACY", out)
        self.assertEqual(pf(os.path.join(self.d, "absent.json"))[0], 0)
        # A failure inside the check is not a refusal: exit 2, never 1.
        real, iface.compare = iface.compare, None
        try:
            with redirect_stderr(io.StringIO()):
                self.assertEqual(pf(good)[0], 2)
        finally:
            iface.compare = real

    def test_stamp_round_trips_and_is_found_in_the_header_only(self):
        g = iface.golden()
        line = iface.stamp_line(g)
        st = iface.parse_stamp("# GENERATED\n%s\na: 1\n" % line)
        self.assertEqual(st["digest"], g.digest)
        self.assertEqual(st["pin"], g.commit)
        self.assertEqual(st["sizing"], {s: g.section_digests[s] for s in iface.SIZING})
        self.assertIsNone(iface.parse_stamp("# GENERATED\na: 1\n%s\n" % line))
        self.assertIsNone(iface.parse_stamp("# GENERATED\na: 1\n"))

    def test_sync_vendors_and_pins(self):
        gp, pp = os.path.join(self.d, "g.json"), os.path.join(self.d, "p.json")
        doc = golden_doc()
        doc["sections"]["rest"]["new_family"] = 1
        dump = self.put("dump.json", {"gnss": entry_of(doc)})
        rc, _ = run_cli("sync", "--from", dump, "--golden", gp, "--pin", pp)
        self.assertEqual(rc, 2, "a dump names no commit: --commit is required")
        rc, out = run_cli(
            "sync", "--from", dump, "--commit", "c0ffee", "--golden", gp, "--pin", pp
        )
        self.assertEqual(rc, 0, out)
        g = iface.load_golden(gp, pp)
        self.assertEqual((g.doc, g.commit), (doc, "c0ffee"))
        with open(gp) as fh:
            self.assertEqual(fh.read(), iface.format_golden(doc))
        rc, out = run_cli("sync", "--from", "golden", "--golden", gp, "--pin", pp)
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
