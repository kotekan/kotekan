#!/usr/bin/env python3
"""power_dump.py -- record a kotekan networkPowerStream to disk.

A headless, modern successor to the old pySaveTCP.py: listen for a kotekan
``networkPowerStream`` (which connects OUT to us, same as livebeam_server.py),
parse the handshake, and append every integration's power spectra to a
self-describing file. This is the "reduced" tens-of-ms spectral data product --
useful when you don't want to dig into the full raw voltages.

Point a kotekan networkPowerStream at this listener, e.g. in the yaml:
    stream_dump:
        kotekan_stage: networkPowerStream
        in_buf: power_buffer
        destination_port: 23500
        destination_ip: <this host>
then:  python3 power_dump.py --port 23500 --source squirrel

File format (magic ``POWERDUMP1``), little-endian:
    line 1:  b"POWERDUMP1\n"
    line 2:  JSON metadata + b"\n"   (nfreq, nvis, freqs_hz, elem stokes, cadence...)
    then:    fixed-size records, one per integration --
             struct "<q d i" (frame_idx int64, utc float64, samples_summed int32)
             followed by nvis*nfreq float32 power values.
    `power` holds the raw wire values; per-sample mean power = power / samples_summed.

Read it back (numpy only):
    meta, data = read_power_dump("power_squirrel_20260819-101500.pdump")
    data["power"][t, vis, freq]   # float32 ;   data["utc"][t] ;   meta["freqs_hz"]
"""
import argparse
import json
import socket
import struct
import time

import numpy as np

MAGIC = b"POWERDUMP1"
# IntensityHeader (see lib/utils/powerStreamUtil.hpp), 48 bytes.
HEADER_FMT = "=iiiidiiiId"
HEADER_LEN = struct.calcsize(HEADER_FMT)
# IntensityPacketHeader (per subframe): frame_idx, elem_idx, samples_summed.
SUBHDR_FMT = "=iii"
SUBHDR_LEN = struct.calcsize(SUBHDR_FMT)
# Per-integration record scalar prefix: frame_idx, utc, samples_summed.
REC_PREFIX = "<qdi"
STOKES = {-8: "YX", -7: "XY", -6: "YY", -5: "XX", -4: "LR", -3: "RL",
          -2: "LL", -1: "RR", 1: "I", 2: "Q", 3: "U", 4: "V"}


def _recv_exact(conn, n):
    chunks, got = [], 0
    while got < n:
        b = conn.recv(min(n - got, 65536))
        if not b:
            raise ConnectionError("kotekan stream closed")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def read_power_dump(path):
    """Load a POWERDUMP1 file -> (meta: dict, data: structured ndarray).

    ``data`` fields: ``frame_idx`` (i8), ``utc`` (f8), ``samples_summed`` (i4),
    ``power`` (f4, shape (nvis, nfreq)). A truncated final record (recorder
    killed mid-write) is dropped automatically.
    """
    with open(path, "rb") as f:
        magic = f.readline().rstrip(b"\n")
        if magic != MAGIC:
            raise ValueError(f"{path}: not a POWERDUMP1 file (got {magic!r})")
        meta = json.loads(f.readline().decode())
        rec = np.dtype([("frame_idx", "<i8"), ("utc", "<f8"),
                        ("samples_summed", "<i4"),
                        ("power", "<f4", (meta["nvis"], meta["nfreq"]))])
        data = np.fromfile(f, dtype=rec)
    return meta, data


def _open_outfile(args, meta):
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = args.outfile or f"power_{args.source or 'aro'}_{ts}.pdump"
    out = open(path, "wb")
    out.write(MAGIC + b"\n")
    out.write((json.dumps(meta) + "\n").encode())
    out.flush()
    return path, out


def _handle(conn, args):
    dtype = np.uint32 if args.power_dtype == "uint32" else np.float32
    # Protocol version: v2 prepends a small int; v1's first int is packet_length
    # (large), so a small leading int marks v2.
    first4 = _recv_exact(conn, 4)
    (v0,) = struct.unpack("=i", first4)
    if 0 < v0 < 64:
        version, hdr = v0, _recv_exact(conn, HEADER_LEN)
    else:
        version, hdr = 1, first4 + _recv_exact(conn, HEADER_LEN - 4)
    (pkt_len, sub_len, nsamp, samp_type, raw_cad, nfreq, nvis,
     samples_summed, idx0, utc0) = struct.unpack(HEADER_FMT, hdr)
    if version >= 2:
        # per-element freq map (nvis x nfreq x [lo, hi]) + one stokes per element
        nfb = nvis * nfreq * 4 * 2
        info = _recv_exact(conn, nfb + nvis)
        freqs = np.frombuffer(info[:nfb], dtype=np.float32).reshape(nvis, nfreq, 2)
        elems = np.frombuffer(info[nfb:], dtype=np.int8)
    else:
        info = _recv_exact(conn, nfreq * 4 * 2 + nvis)
        freqs = np.frombuffer(info[:nfreq * 4 * 2], dtype=np.float32).reshape(-1, 2)
        elems = np.frombuffer(info[nfreq * 4 * 2:], dtype=np.int8)
    period = abs(raw_cad) * samples_summed  # seconds between integrations

    meta = {
        "magic": MAGIC.decode(), "version": 1,
        "wire_protocol_version": int(version),
        "source": args.source, "power_dtype": args.power_dtype,
        "nfreq": int(nfreq), "nvis": int(nvis),
        "sample_type": int(samp_type),
        "raw_cadence_s": abs(float(raw_cad)),
        "samples_summed": int(samples_summed),
        "frame_period_s": float(period),
        "handshake_frame_idx": int(idx0), "handshake_utc": float(utc0),
        "freqs_hz": freqs.tolist(),                    # per-bin [lo, hi]
        "elem_stokes": [int(e) for e in elems],
        "elem_labels": [STOKES.get(int(e), f"elem{int(e)}") for e in elems],
        "created_utc": time.time(),
        "note": "power holds raw wire values; per-sample mean = power/samples_summed",
    }
    path, out = _open_outfile(args, meta)
    print(f"recording -> {path}  (nvis={nvis} nfreq={nfreq} "
          f"period={period * 1e3:.1f} ms, labels={meta['elem_labels']})", flush=True)

    buf = np.zeros((nvis, nfreq), dtype=np.float32)
    subframe_len = sub_len + pkt_len
    split_s = args.split_minutes * 60.0
    n, t_report, t_open, fidx, ss = 0, time.time(), time.time(), idx0, samples_summed
    while True:
        for i in range(nvis):
            d = _recv_exact(conn, subframe_len)
            fidx, eidx, ss = struct.unpack(SUBHDR_FMT, d[:sub_len])
            if 0 <= eidx < nvis:
                buf[eidx] = np.frombuffer(d[sub_len:], dtype=dtype).astype(np.float32)
        utc = utc0 + abs(raw_cad) * samples_summed * (fidx - idx0)
        out.write(struct.pack(REC_PREFIX, int(fidx), float(utc), int(ss)))
        out.write(np.ascontiguousarray(buf, dtype="<f4").tobytes())
        n += 1
        now = time.time()
        if now - t_report >= 5.0:
            out.flush()
            print(f"  {n} integrations written ({n * period:.0f}s of data)", flush=True)
            t_report = now
        if split_s and now - t_open >= split_s:
            out.close()
            path, out = _open_outfile(args, meta)
            print(f"rotated -> {path}", flush=True)
            t_open = now


def serve(args):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(f"power_dump listening on {args.bind}:{args.port} "
          f"(kotekan networkPowerStream connects out to us)", flush=True)
    while True:
        conn, peer = srv.accept()
        print(f"connected: {peer}", flush=True)
        try:
            _handle(conn, args)
        except (ConnectionError, OSError, RuntimeError) as e:
            print(f"stream ended: {e}", flush=True)
        finally:
            conn.close()
        if args.exit_on_disconnect:
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=23500,
                    help="TCP port to listen on (kotekan connects out to it)")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--source", default=None, help="label used in the filename + metadata")
    ap.add_argument("--outfile", default=None,
                    help="output path (default: power_<source>_<UTC>.pdump)")
    ap.add_argument("--power-dtype", choices=["uint32", "float32"], default="uint32",
                    help="wire sample dtype (ARO computeDualpolPower = uint32)")
    ap.add_argument("--split-minutes", type=float, default=0.0,
                    help="start a new file every N minutes (0 = one file per connection)")
    ap.add_argument("--exit-on-disconnect", action="store_true",
                    help="exit when the stream drops (for a respawn-loop supervisor)")
    ap.add_argument("--read", metavar="FILE", default=None,
                    help="don't record; print a summary of an existing .pdump file")
    args = ap.parse_args()

    if args.read:
        meta, data = read_power_dump(args.read)
        print(json.dumps(meta, indent=2))
        print(f"\n{len(data)} integrations; power shape "
              f"(t, nvis, nfreq) = {(len(data),) + data['power'].shape[1:]}")
        if len(data):
            span = data["utc"][-1] - data["utc"][0]
            print(f"time span: {span:.1f}s "
                  f"({time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(data['utc'][0]))} UTC + )")
        return

    serve(args)


if __name__ == "__main__":
    main()
