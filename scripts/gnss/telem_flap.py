#!/usr/bin/env python3
"""Flap harness for GnssTelemGather: a gather fed a telemetry capture while a client
connects to its serve port, takes nothing, and reconnects at a fixed rate.

    telem_flap.py --kotekan <build>/kotekan/kotekan \\
                  --config config/generated/chord_gnss_gather.yaml \\
                  --capture <telemetry capture>.bin \\
                  [--duration 600] [--flaps-per-min 20] [--clients 1] [--read] [--no-sender]
                  [--port-offset 1000] [--workdir DIR] [--env VAR=VALUE ...]

The gather is started from a copy of the config with every port moved by --port-offset, so it
can run beside a live stack. The capture is the gather's own serve stream ([uint32 length][frame],
as gnss_broker/telem.py reads it); each frame is sent to the gather's bufferRecv listener with
the bufferSend wire header and a GnssChanMetadata blob, paced by the frames' utc0, and looped for
the whole run. A flapping client reads nothing, so the gather's send deadline expires once the
socket buffer is full and the gather drops it; with --read the client drains the stream instead
and the gather sees its close (the reap path, or a failed send if a frame is in flight). With
--no-sender no frames flow, so only the reaper can notice a closed client.

Exit 0 if the gather is still running at the end, 1 if it died (the exit status and the tail of
its log are printed), 2 if it never came up.
"""
import argparse
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import json

# bufferSend header without the config tracker (use_config_tracker: false in the gather config):
# metadata_size, frame_size. The metadata is GnssChanMetadata: int64 sample_seq, uint32 reserved.
_SEND_HDR = struct.Struct("<II")
_META = struct.Struct("<qI")
_META_BYTES = _META.size


def read_capture(path):
    """Every (utc0, wstart0, frame) in the capture, in order."""
    frames = []
    with open(path, "rb") as fh:
        while True:
            raw = fh.read(4)
            if len(raw) < 4:
                break
            (n,) = struct.unpack("<I", raw)
            fr = fh.read(n)
            if len(fr) < n:
                break
            (wstart0,) = struct.unpack_from("<q", fr, 40)
            (utc0,) = struct.unpack_from("<d", fr, 48)
            frames.append((utc0, wstart0, fr))
    return frames


def port_moved_config(src, dst, offset, workdir):
    """Copy the gather config with listen/serve/rest ports moved and the trim state file in
    workdir. Returns (listen_port, serve_port, rest_port)."""
    text = open(src).read()
    ports = {}

    def bump(key, m):
        p = int(m.group(2)) + offset
        ports[key] = p
        return "%s%d" % (m.group(1), p)

    text = re.sub(r"(listen_port:\s*)(\d+)", lambda m: bump("listen", m), text)
    text = re.sub(r"(serve_port:\s*)(\d+)", lambda m: bump("serve", m), text)
    text = re.sub(r"(\n\s+port:\s*)(\d+)", lambda m: bump("rest", m), text)
    text = re.sub(
        r"(trim_state_file:\s*)\S+",
        lambda m: m.group(1) + os.path.join(workdir, "fleet_trim.json"),
        text,
    )
    if set(ports) != {"listen", "serve", "rest"}:
        sys.exit("config %s: found ports %s, expected listen/serve/rest" % (src, ports))
    with open(dst, "w") as fh:
        fh.write(text)
    return ports["listen"], ports["serve"], ports["rest"]


def wait_listening(port, timeout):
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


class Sender(threading.Thread):
    """Stream the capture into the gather's listener, paced by utc0, looping until stopped."""

    def __init__(self, frames, port, stop):
        super().__init__(daemon=True)
        self.frames = frames
        self.port = port
        self.stop = stop
        self.sent = 0
        self.reconnects = 0

    def run(self):
        utcs = [u for u, _, _ in self.frames if u > 0]
        span = (max(utcs) - min(utcs)) if utcs else 0.0
        # A frame whose slot 0 is absent has utc0 0: it goes out with the previous frame.
        t_first = min(utcs) if utcs else 0.0
        sock = None
        while not self.stop.is_set():
            t_loop = time.time()
            last_rel = 0.0
            for utc0, wstart0, fr in self.frames:
                if self.stop.is_set():
                    break
                if utc0 > 0:
                    last_rel = utc0 - t_first
                wait = t_loop + last_rel - time.time()
                if wait > 0:
                    time.sleep(wait)
                payload = (
                    _SEND_HDR.pack(_META_BYTES, len(fr)) + _META.pack(wstart0, 0) + fr
                )
                while sock is None and not self.stop.is_set():
                    try:
                        sock = socket.create_connection(
                            ("127.0.0.1", self.port), timeout=5
                        )
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        self.reconnects += 1
                    except OSError:
                        time.sleep(0.5)
                if sock is None:
                    break
                try:
                    sock.sendall(payload)
                    self.sent += 1
                except OSError:
                    sock.close()
                    sock = None
            # Keep the pace across loops: the next pass starts where this one would have ended.
            rest = t_loop + span - time.time()
            if rest > 0:
                time.sleep(rest)
        if sock is not None:
            sock.close()


class Flapper(threading.Thread):
    """Connect to the serve port, hold the connection for 60/flaps_per_min seconds reading
    nothing (or draining the stream with `read`), then close and reconnect. The gather drops a
    client that reads nothing once the socket buffer is full and its send deadline expires; a
    client whose period ends first is closed here and reaped there."""

    def __init__(self, port, flaps_per_min, stop, read):
        super().__init__(daemon=True)
        self.port = port
        self.period = 60.0 / flaps_per_min
        self.stop = stop
        self.read = read
        self.connects = 0
        self.refused = 0
        self.received = 0

    def run(self):
        while not self.stop.is_set():
            try:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=2)
            except OSError:
                self.refused += 1
                self.stop.wait(min(self.period, 1.0))
                continue
            self.connects += 1
            t_end = time.time() + self.period
            if self.read:
                s.settimeout(0.1)
                while not self.stop.is_set() and time.time() < t_end:
                    try:
                        b = s.recv(1 << 16)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not b:
                        break
                    self.received += len(b)
            else:
                self.stop.wait(self.period)
            s.close()


def gather_stats(rest_port):
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:%d/telem_gather/get_stats" % rest_port, timeout=2
        ) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001 -- any failure reads as "no stats"
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kotekan", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--flaps-per-min", type=float, default=20.0)
    ap.add_argument(
        "--clients", type=int, default=1, help="flapping clients, staggered"
    )
    ap.add_argument("--read", action="store_true", help="clients drain the stream")
    ap.add_argument("--no-sender", action="store_true", help="no frames flow")
    ap.add_argument("--port-offset", type=int, default=1000)
    ap.add_argument(
        "--workdir", default="/var/tmp/%s-telem-flap" % os.environ.get("USER", "x")
    )
    ap.add_argument(
        "--env", action="append", default=[], help="VAR=VALUE for the gather"
    )
    a = ap.parse_args()

    os.makedirs(a.workdir, exist_ok=True)
    cfg = os.path.join(a.workdir, "gather_flap.yaml")
    listen, serve, rest = port_moved_config(a.config, cfg, a.port_offset, a.workdir)
    frames = read_capture(a.capture)
    if not frames:
        sys.exit("no frames in %s" % a.capture)
    print(
        "capture: %d frames, config: listen %d serve %d rest %d, workdir %s"
        % (len(frames), listen, serve, rest, a.workdir),
        flush=True,
    )

    env = dict(os.environ)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    log_path = os.path.join(a.workdir, "gather.log")
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        [a.kotekan, "-c", cfg, "-b", "127.0.0.1:%d" % rest],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=a.workdir,
    )
    if not wait_listening(serve, 60):
        proc.send_signal(signal.SIGTERM)
        proc.wait(10)
        print("gather never listened on %d; log %s" % (serve, log_path))
        return 2

    stop = threading.Event()
    sender = Sender(frames, listen, stop)
    if not a.no_sender:
        sender.start()
    flappers = []
    for i in range(a.clients):
        f = Flapper(serve, a.flaps_per_min, stop, a.read)
        flappers.append(f)
        f.start()
        time.sleep(60.0 / a.flaps_per_min / a.clients)

    t_end = time.time() + a.duration
    t_report = time.time() + 30
    rc = None
    while time.time() < t_end:
        rc = proc.poll()
        if rc is not None:
            break
        if time.time() >= t_report:
            t_report += 30
            st = gather_stats(rest)
            print(
                "t=%4.0f s sender sent %d (reconnects %d); flaps connects %d refused %d "
                "received %d MB; gather clients %s drops %s hangups %s bad %s"
                % (
                    a.duration - (t_end - time.time()),
                    sender.sent,
                    sender.reconnects,
                    sum(f.connects for f in flappers),
                    sum(f.refused for f in flappers),
                    sum(f.received for f in flappers) >> 20,
                    st.get("clients"),
                    st.get("client_drops"),
                    st.get("client_hangups"),
                    st.get("bad_frames"),
                ),
                flush=True,
            )
        time.sleep(1.0)
    stop.set()
    if sender.is_alive():
        sender.join(5)
    for f in flappers:
        f.join(5)

    if rc is None:
        st = gather_stats(rest)
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print(
            "SURVIVED %.0f s: sender sent %d frames; flaps connects %d; gather stats %s"
            % (
                a.duration,
                sender.sent,
                sum(f.connects for f in flappers),
                json.dumps(
                    {
                        k: st.get(k)
                        for k in (
                            "clients",
                            "client_drops",
                            "client_hangups",
                            "bad_frames",
                        )
                    }
                ),
            )
        )
        print("shutdown exit status %s; log %s" % (proc.returncode, log_path))
        return 0

    log.flush()
    tail = subprocess.run(
        ["tail", "-n", "40", log_path], capture_output=True, text=True
    ).stdout
    print(
        "DIED after %.0f s with status %s (signal %s): sender sent %d; flaps connects %d"
        % (
            a.duration - (t_end - time.time()),
            rc,
            -rc if rc < 0 else None,
            sender.sent,
            sum(f.connects for f in flappers),
        )
    )
    print("--- last 40 log lines (%s) ---\n%s" % (log_path, tail))
    return 1


if __name__ == "__main__":
    sys.exit(main())
