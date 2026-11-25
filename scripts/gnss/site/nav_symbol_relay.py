#!/usr/bin/env python3
"""Nav-symbol relay: data symbols streamed from another receiver (the bladeRF receiver on the
gx10, through an SSH tunnel to this host) to the node chains that wipe them from their replicas
(gnssTrackChain.hpp, set_nav_symbols).

  run   [--listen HOST:PORT] [--capture DIR] [--keep-hours H] [--accept-unresolved]
        Accept the sender on --listen (127.0.0.1:12080) and parse its records. Every
        --batch-ms, POST the new symbols of each signal to every chain row that takes them:
        the rows with set_nav_symbols_endpoint in the generated node configs, keyed by their
        `signal`. Only symbols whose sign the sender has resolved (CRC) go out, unless
        --accept-unresolved. Every record, with its arrival time, goes to an hourly capture
        file in DIR; files older than H hours are deleted. Once a minute the log gives, per
        signal, how old the symbols were on arrival (arrival - transmit time of their start)
        and what was dropped.
  parse FILE
        Print the records of a capture file.

Stream (agreed with the gx10 session, 2026-11-24), little-endian, one TCP connection, each
record after a u32 byte length:
  u32 magic 0x4D59534E ("NSYM"), u16 version 1, u32 seq (per connection), u8 sys (ASCII G E C
  R S W), u8 sig (1 L2C_CM, 2 B2B_I, 3 L5_I, 4 E5A_I, 5 E5B_I, 6 B2A_D, 7 B3I_D1, 8 GLO_L2OF,
  9 GW_1202), u16 prn, u32 stream_id, u16 flags (bit 0 resolved, bit 1 conf), u16 n, i64 sym0,
  f64 t_dec, f64 t_out, f64 rx_utc_sym0; then ceil(n/8) bytes of signs (bit 1 = -1, LSB
  first) and, with flag bit 1, n u8 conf (|proj|/sigma x 16). n = 0 is a heartbeat.
  sym0 = floor(t / T_sym), t the satellite-clock transmit time in seconds since 1980-01-06 on
  the system's own scale: GPST, GST, or BDT = GPST - 14 s. A capture record is the arrival
  time (f64 UTC) and the record's length and bytes.
"""
import argparse
import glob
import http.client
import json
import os
import queue
import re
import socket
import struct
import sys
import threading
import time
import urllib.parse

import yaml

K = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
GEN = "config/generated/chord_gnss_%s_multi.yaml"
PORT = 12048
MAGIC = 0x4D59534E
HEAD = struct.Struct("<IHIBBHIHHqddd")
# The chain descriptor of each stream signal CHORD wipes, and every signal's symbol period
SIG_DESC = {1: "GPS_L2C_CM", 2: "BDS_B2B_I"}
SIG_NAME = {
    1: "L2C_CM",
    2: "B2B_I",
    3: "L5_I",
    4: "E5A_I",
    5: "E5B_I",
    6: "B2A_D",
    7: "B3I_D1",
    8: "GLO_L2OF",
    9: "GW_1202",
}
SIG_T = {1: 0.02, 2: 0.001, 3: 0.01, 4: 0.02, 5: 0.004, 6: 0.005, 7: 0.02, 8: 0.01}
GPS_EPOCH_UNIX = 315964800.0
GPS_MINUS_UTC_S = 18.0  # leap seconds since the GPS epoch, unchanged since 2017
BDT_MINUS_GPST_S = -14.0


def log(msg):
    sys.stdout.write(
        "%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), msg)
    )
    sys.stdout.flush()


def pack(
    seq, sys_, sig, prn, stream_id, resolved, sym0, signs, t_dec=0.0, t_out=0.0, rx=0.0
):
    """One record, with its length prefix (the sender's side; for tests and replays)."""
    n = len(signs)
    bits = bytearray((n + 7) // 8)
    for k, s in enumerate(signs):
        if s < 0:
            bits[k // 8] |= 1 << (k % 8)
    body = HEAD.pack(
        MAGIC,
        1,
        seq,
        ord(sys_),
        sig,
        prn,
        stream_id,
        1 if resolved else 0,
        n,
        sym0,
        t_dec,
        t_out,
        rx,
    ) + bytes(bits)
    return struct.pack("<I", len(body)) + body


def parse(body):
    """A record's fields; signs as a string of '+' and '-'."""
    (
        magic,
        ver,
        seq,
        sys_,
        sig,
        prn,
        stream_id,
        flags,
        n,
        sym0,
        t_dec,
        t_out,
        rx,
    ) = HEAD.unpack_from(body)
    if magic != MAGIC or ver != 1:
        raise ValueError("not an NSYM v1 record (magic %#x, version %d)" % (magic, ver))
    nb = (n + 7) // 8
    if len(body) < HEAD.size + nb + (n if flags & 2 else 0):
        raise ValueError("record shorter than its %d symbols" % n)
    bits = body[HEAD.size : HEAD.size + nb]
    signs = "".join("-" if bits[k // 8] >> (k % 8) & 1 else "+" for k in range(n))
    return dict(
        seq=seq,
        sys=chr(sys_) if sys_ else "",
        sig=sig,
        prn=prn,
        stream_id=stream_id,
        resolved=bool(flags & 1),
        n=n,
        sym0=sym0,
        signs=signs,
        t_dec=t_dec,
        t_out=t_out,
        rx_utc=rx,
    )


def tx_unix(rec):
    """UTC (unix seconds) of the transmit time of the record's first symbol, or None."""
    T = SIG_T.get(rec["sig"])
    if T is None:
        return None
    gpst = rec["sym0"] * T - (BDT_MINUS_GPST_S if rec["sys"] == "C" else 0.0)
    return GPS_EPOCH_UNIX + gpst - GPS_MINUS_UTC_S


def targets(root, nodes=None):
    """{descriptor: [url, ...]}: every chain row with set_nav_symbols_endpoint, at the row's
    own path, which a node serves under --inject-host too."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, GEN % "cx*"))):
        node = re.sub(r".*chord_gnss_(cx\d+)_multi\.yaml$", r"\1", path)
        if nodes is not None and node not in nodes:
            continue
        cfg = yaml.safe_load(open(path)) or {}
        stack = [cfg]
        while stack:
            d = stack.pop()
            if isinstance(d, dict):
                ep = d.get("set_nav_symbols_endpoint")
                if ep and d.get("signal"):
                    out.setdefault(d["signal"], []).append(
                        "http://%s:%d%s" % (node, PORT, ep)
                    )
                stack.extend(d.values())
            elif isinstance(d, list):
                stack.extend(d)
    for desc in out:
        out[desc] = sorted(set(out[desc]))
    return out


class Target(threading.Thread):
    """One endpoint's POSTs, in order, from a bounded queue over one kept-alive connection: a
    slow node drops its own batches and nobody else's."""

    def __init__(self, url, timeout):
        super().__init__(daemon=True)
        self.url, self.timeout = url, timeout
        u = urllib.parse.urlsplit(url)
        self.host, self.port, self.path = u.hostname, u.port, u.path
        self.name = "%s%s" % (self.host, self.path.rsplit("_inject/", 1)[0])
        self.conn = None
        self.q = queue.Queue(maxsize=50)
        self.sent = self.dropped = self.failed = 0
        self.logged = (0, 0)  # dropped, failed at the last report

    def offer(self, data):
        try:
            self.q.put_nowait(data)
        except queue.Full:
            self.dropped += 1

    def post(self, data):
        if self.conn is None:
            self.conn = http.client.HTTPConnection(
                self.host, self.port, timeout=self.timeout
            )
        self.conn.request(
            "POST", self.path, body=data, headers={"Content-Type": "application/json"}
        )
        r = self.conn.getresponse()
        r.read()
        if r.status != 200:
            raise http.client.HTTPException("HTTP %d" % r.status)

    def run(self):
        while True:
            data = self.q.get()
            try:
                self.post(data)
                self.sent += 1
            except Exception:
                self.failed += 1
                if self.conn is not None:
                    self.conn.close()
                self.conn = None


class Relay:
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()
        self.pending = {}  # descriptor -> rows
        self.ages = {}  # signal name -> ages (s) this minute
        self.counts = {}  # signal name -> [records, unresolved, symbols]
        self.gaps = 0
        nodes = set(a.nodes.split(",")) if a.nodes else None
        self.urls = targets(a.root, nodes)
        self.targets = {
            d: [Target(u, a.timeout) for u in us] for d, us in self.urls.items()
        }
        for ts in self.targets.values():
            for t in ts:
                t.start()
        self.cap = None
        self.cap_hour = None

    def capture(self, t_arr, body):
        if not self.a.capture:
            return
        hour = time.strftime("%Y%m%dT%H", time.gmtime(t_arr))
        if hour != self.cap_hour:
            if self.cap:
                self.cap.close()
            os.makedirs(self.a.capture, exist_ok=True)
            self.cap = open(
                os.path.join(self.a.capture, "nav_symbols_%s.bin" % hour), "ab"
            )
            self.cap_hour = hour
            cut = t_arr - self.a.keep_hours * 3600
            for f in glob.glob(os.path.join(self.a.capture, "nav_symbols_*.bin")):
                if os.path.getmtime(f) < cut:
                    os.remove(f)
        self.cap.write(struct.pack("<dI", t_arr, len(body)) + body)
        self.cap.flush()

    def take(self, rec, t_arr):
        name = SIG_NAME.get(rec["sig"], str(rec["sig"]))
        c = self.counts.setdefault(name, [0, 0, 0])
        c[0] += 1
        c[1] += 0 if rec["resolved"] else 1
        c[2] += rec["n"]
        tx = tx_unix(rec)
        if tx is not None:
            self.ages.setdefault(name, []).append(t_arr - tx)
        desc = SIG_DESC.get(rec["sig"])
        if desc in self.targets and (rec["resolved"] or self.a.accept_unresolved):
            row = dict(
                prn=rec["prn"],
                sym0=rec["sym0"],
                signs=rec["signs"],
                stream_id=rec["stream_id"],
            )
            with self.lock:
                self.pending.setdefault(desc, []).append(row)

    def dispatch(self):
        while True:
            time.sleep(self.a.batch_ms / 1000.0)
            with self.lock:
                batch, self.pending = self.pending, {}
            for desc, rows in batch.items():
                data = json.dumps(rows, separators=(",", ":")).encode()
                for t in self.targets[desc]:
                    t.offer(data)

    def report(self):
        while True:
            time.sleep(60.0)
            parts = []
            for name, c in sorted(self.counts.items()):
                ages = sorted(self.ages.get(name, []))
                q = (
                    lambda f: "%.0f"
                    % (1e3 * ages[min(len(ages) - 1, int(f * len(ages)))])
                    if ages
                    else "-"
                )
                parts.append(
                    "%s %d rec %d sym (%d unresolved) age ms p50 %s p99 %s max %s"
                    % (name, c[0], c[2], c[1], q(0.5), q(0.99), q(1.0))
                )
            self.counts, self.ages = {}, {}
            tg = [t for ts in self.targets.values() for t in ts]
            slow = []
            for t in tg:
                d, f = t.dropped - t.logged[0], t.failed - t.logged[1]
                t.logged = (t.dropped, t.failed)
                if d or f:
                    slow.append("%s %d/%d" % (t.name, d, f))
            log(
                "NAVSYM 60s: %s | seq gaps %d | posts %d dropped %d failed %d over %d targets"
                " (totals) | dropped/failed this minute: %s"
                % (
                    "; ".join(parts) or "nothing",
                    self.gaps,
                    sum(t.sent for t in tg),
                    sum(t.dropped for t in tg),
                    sum(t.failed for t in tg),
                    len(tg),
                    ", ".join(slow) or "none",
                )
            )

    def serve(self, conn):
        last = None
        buf = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buf += chunk
            while len(buf) >= 4:
                (ln,) = struct.unpack_from("<I", buf)
                if ln < HEAD.size or ln > 1 << 20:
                    raise ValueError("record length %d" % ln)
                if len(buf) < 4 + ln:
                    break
                body, buf = buf[4 : 4 + ln], buf[4 + ln :]
                t_arr = time.time()
                rec = parse(body)
                self.capture(t_arr, body)
                if last is not None and rec["seq"] != last + 1:
                    self.gaps += 1
                last = rec["seq"]
                if rec["n"]:
                    self.take(rec, t_arr)

    def run(self):
        host, port = self.a.listen.rsplit(":", 1)
        for d, us in sorted(self.urls.items()):
            log("targets %s: %s" % (d, " ".join(us)))
        threading.Thread(target=self.dispatch, daemon=True).start()
        threading.Thread(target=self.report, daemon=True).start()
        srv = socket.create_server((host, int(port)))
        log("listening on %s" % self.a.listen)
        while True:
            conn, peer = srv.accept()
            log("sender connected from %s:%d" % peer)
            try:
                self.serve(conn)
                log("sender closed the connection")
            except Exception as e:
                log("sender connection dropped: %s" % e)
            finally:
                conn.close()


def read_capture(path):
    with open(path, "rb") as f:
        raw = f.read()
    off = 0
    while off + 12 <= len(raw):
        t_arr, ln = struct.unpack_from("<dI", raw, off)
        off += 12
        yield t_arr, parse(raw[off : off + ln])
        off += ln


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["run", "parse"])
    ap.add_argument("file", nargs="?", help="parse: a capture file")
    ap.add_argument("--listen", default="127.0.0.1:12080")
    ap.add_argument(
        "--batch-ms", type=float, default=20.0, help="ms between POSTs (20)"
    )
    ap.add_argument("--timeout", type=float, default=0.5, help="s per POST (0.5)")
    ap.add_argument("--nodes", help="only these nodes: NODE,NODE (default every node)")
    ap.add_argument("--capture", help="directory for the hourly capture files")
    ap.add_argument(
        "--keep-hours", type=float, default=24.0, help="capture kept, h (24)"
    )
    ap.add_argument(
        "--accept-unresolved",
        action="store_true",
        help="also pass symbols whose sign the sender has not resolved",
    )
    ap.add_argument(
        "--root", default=K, help="the checkout whose config/generated/ to read"
    )
    a = ap.parse_args()
    if a.mode == "parse":
        for t_arr, rec in read_capture(a.file):
            print(json.dumps(dict(t_arr=t_arr, **rec)))
    else:
        Relay(a).run()


if __name__ == "__main__":
    main()
