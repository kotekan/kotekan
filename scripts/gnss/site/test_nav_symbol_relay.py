#!/usr/bin/env python3
"""nav_symbol_relay: the record format, the age of a symbol, target discovery, and one record
from the sender's socket to a node endpoint."""
import http.server
import json
import os
import socket
import sys
import tempfile
import threading
import time
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav_symbol_relay as R  # noqa: E402


class Format(unittest.TestCase):
    def test_round_trip(self):
        signs = [1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1]
        rec = R.pack(7, "C", 2, 25, 3, True, 1_400_000_000_123, signs, 1.0, 2.0, 3.0)
        r = R.parse(rec[4:])
        self.assertEqual((r["seq"], r["sys"], r["sig"], r["prn"]), (7, "C", 2, 25))
        self.assertEqual((r["stream_id"], r["resolved"], r["n"]), (3, True, 11))
        self.assertEqual(r["sym0"], 1_400_000_000_123)
        self.assertEqual(r["signs"], "+--+++-+--+")
        self.assertEqual((r["t_dec"], r["t_out"], r["rx_utc"]), (1.0, 2.0, 3.0))

    def test_bad_magic(self):
        rec = bytearray(R.pack(1, "G", 1, 5, 0, True, 0, [1])[4:])
        rec[0] ^= 1
        with self.assertRaises(ValueError):
            R.parse(bytes(rec))

    def test_transmit_time(self):
        # B2b: index in BDT ms; 14 s behind GPST, then GPST - 18 s to UTC
        r = dict(sig=2, sys="C", sym0=1_400_000_000_000)
        self.assertAlmostEqual(R.tx_unix(r), R.GPS_EPOCH_UNIX + 1_400_000_000 + 14 - 18)
        r = dict(sig=1, sys="G", sym0=70_000_000_000)
        self.assertAlmostEqual(R.tx_unix(r), R.GPS_EPOCH_UNIX + 1_400_000_000 - 18)


class Targets(unittest.TestCase):
    def test_rows_with_the_endpoint(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "config", "generated"))
            cfg = {
                "gnss0_n2dual": {
                    "commands": [
                        {
                            "chains": {
                                "gnss0_bds_b2b": {
                                    "signal": "BDS_B2B_I",
                                    "set_nav_symbols_endpoint": "/gnss0_bds_b2b_inject/set_nav_symbols",
                                },
                                "gnss0_gal_e5b": {"signal": "GAL_E5B_Q"},
                            }
                        }
                    ]
                }
            }
            with open(os.path.join(root, R.GEN % "cx99"), "w") as f:
                json.dump(cfg, f)
            t = R.targets(root)
        self.assertEqual(list(t), ["BDS_B2B_I"])
        self.assertEqual(len(t["BDS_B2B_I"]), 1)
        self.assertTrue(
            t["BDS_B2B_I"][0].endswith("/gnss0_bds_b2b_inject/set_nav_symbols")
        )


class KeepAlive(unittest.TestCase):
    def test_batches_share_one_connection(self):
        peers = []

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                peers.append(self.client_address)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass

        node = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=node.serve_forever, daemon=True).start()
        t = R.Target(
            "http://127.0.0.1:%d/gnss0_bds_b2b_inject/set_nav_symbols"
            % node.server_port,
            1.0,
        )
        t.start()
        try:
            for _ in range(3):
                t.offer(b"[]")
            t0 = time.time()
            while t.sent + t.failed < 3 and time.time() - t0 < 2.0:
                time.sleep(0.01)
        finally:
            node.shutdown()
        self.assertEqual((t.sent, t.failed), (3, 0))
        self.assertEqual(len(set(peers)), 1)
        self.assertEqual(t.name, "127.0.0.1/gnss0_bds_b2b")


class EndToEnd(unittest.TestCase):
    def test_resolved_symbols_reach_the_node(self):
        got = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                got.append(
                    (
                        self.path,
                        json.loads(
                            self.rfile.read(int(self.headers["Content-Length"]))
                        ),
                    )
                )
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        node = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=node.serve_forever, daemon=True).start()
        url = (
            "http://127.0.0.1:%d/gnss0_bds_b2b_inject/set_nav_symbols"
            % node.server_port
        )
        orig = R.targets
        R.targets = lambda root, nodes=None: {"BDS_B2B_I": [url]}
        try:
            with tempfile.TemporaryDirectory() as cap:
                a = types.SimpleNamespace(
                    listen="127.0.0.1:0",
                    batch_ms=10.0,
                    timeout=1.0,
                    nodes=None,
                    capture=cap,
                    keep_hours=1.0,
                    accept_unresolved=False,
                    root=".",
                )
                relay = R.Relay(a)
                threading.Thread(target=relay.dispatch, daemon=True).start()
                a_sock, b_sock = socket.socketpair()
                th = threading.Thread(target=relay.serve, args=(b_sock,), daemon=True)
                th.start()
                a_sock.sendall(
                    R.pack(1, "C", 2, 25, 4, True, 1000, [1, -1, 1])
                    + R.pack(
                        2, "C", 2, 26, 4, False, 1000, [1, 1]
                    )  # unresolved: held back
                    + R.pack(3, "G", 1, 5, 0, True, 0, [])  # heartbeat
                    + R.pack(4, "G", 3, 9, 0, True, 500, [1])  # L5: no CHORD target
                )
                a_sock.close()
                th.join(2.0)
                t0 = time.time()
                while not got and time.time() - t0 < 2.0:
                    time.sleep(0.01)
                caps = [
                    r
                    for f in os.listdir(cap)
                    for _, r in R.read_capture(os.path.join(cap, f))
                ]
        finally:
            R.targets = orig
            node.shutdown()
        self.assertEqual(len(got), 1)
        path, rows = got[0]
        self.assertEqual(path, "/gnss0_bds_b2b_inject/set_nav_symbols")
        self.assertEqual(rows, [dict(prn=25, sym0=1000, signs="+-+", stream_id=4)])
        self.assertEqual([r["seq"] for r in caps], [1, 2, 3, 4])
        self.assertEqual(relay.gaps, 0)


if __name__ == "__main__":
    unittest.main()
