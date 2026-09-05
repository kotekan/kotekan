#!/usr/bin/env python3
"""Speak bufferSend's wire protocol and push synthetic beam-cube frames at an archiver.

Gate, not a toy: it proves frame_size agreement (bufferRecv CLOSES on a mismatch and delivers
nothing rather than bad data), the writer's file bundling, and that gnss_cube_read.py can parse
what actually lands -- all without a node, a GPU or a satellite.

Wire (use_config_tracker false, use_frame_desc false), per frame:
    uint32 metadata_size | metadata bytes | frame bytes
GnssChanMetadata serializes as int64 sample_seq | uint32 n_chan_scale | n floats.
"""
import socket, struct, sys, time
sys.path.insert(0, "/home/kvand/gnss/kotekan/config")
from gnss_record_layout import cube_frame_bytes, cube_header_bytes  # noqa: E402

HOST, PORT = "127.0.0.1", int(sys.argv[1])
MP, MB, NE = 32, 8, 32
NPRN, NBIN = 9, 7
NFRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 12
CHAIN = sys.argv[3] if len(sys.argv) > 3 else "faketest/gnss0_n2assemble"
GPU = int(sys.argv[4]) if len(sys.argv) > 4 else 0
HDR = cube_header_bytes()
FB = cube_frame_bytes(MP, MB, NE)
WIN = 96 * 2048 * 16384
RATE = 3.2e9


def build(idx, dropped):
    f = bytearray(FB)
    struct.pack_into("<8q", f, 0, 2, idx, idx * WIN, idx * WIN + WIN - 32768,
                     dropped, NPRN, NBIN, NE)
    struct.pack_into("<2d", f, 64, float(WIN), RATE)
    struct.pack_into("<4i", f, 80, GPU, 1, MP, MB)
    f[96:96 + len(CHAIN)] = CHAIN.encode()
    o = HDR
    struct.pack_into("<%dd" % MP, f, o, *([0.25] * MP)); o += MP * 8
    struct.pack_into("<%di" % MB, f, o, *([5972 + i for i in range(MB)])); o += MB * 4
    struct.pack_into("<%di" % MB, f, o, *([5972 + i for i in range(MB)])); o += MB * 4
    struct.pack_into("<%di" % MP, f, o, *([3, 10, 17, 18, 19, 20, 23, 26, 27] + [0] * (MP - NPRN)))
    o += MP * 4
    struct.pack_into("<%di" % MP, f, o, *([96] * NPRN + [0] * (MP - NPRN))); o += MP * 4
    struct.pack_into("<%di" % MP, f, o, *([0] * MP)); o += MP * 4
    struct.pack_into("<%df" % (MP * MB), f, o, *([96.0] * (MP * MB))); o += MP * MB * 4
    struct.pack_into("<%df" % (MP * MB), f, o, *([1.5] * (MP * MB))); o += MP * MB * 4
    n = MP * MB * NE
    struct.pack_into("<%df" % n, f, o, *([0.5] * n)); o += n * 4
    struct.pack_into("<%df" % n, f, o, *([-0.25] * n)); o += n * 4
    # incoh: a recognisable per-element ramp, so a readback proves the ELEMENT axis survived
    vals = [float((e % NE) + 1) for _ in range(MP * MB) for e in range(NE)]
    struct.pack_into("<%df" % n, f, o, *vals)
    return bytes(f)


s = socket.create_connection((HOST, PORT), timeout=10)
meta = struct.pack("<qI", 0, 0)
sent = 0
for i in range(NFRAMES):
    frame = build(1716400 + i, 0)
    s.sendall(struct.pack("<II", len(meta), FB) + meta + frame)
    sent += 1
    time.sleep(0.02)
s.close()
print(f"sent {sent} frame(s) of {FB} B to {HOST}:{PORT} as {CHAIN} gpu{GPU}")
