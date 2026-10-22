#!/usr/bin/env python3
"""End-to-end gate for the tracker->broker telemetry transport: offline, ~20 s, no sky.

Drives the shipped code, not a model of it: real GnssTelemPack, bufferSend, bufferRecv,
GnssTelemGather and gnss_broker/telem.py. Only the record source is synthetic: rawFileRead
replays record frames this script writes, complete with the GnssChanMetadata (sample_seq)
the packer keys on.

    python3 scripts/gnss/telem_e2e.py [--keep] [--timeout S]

test_telem.py checks the parser against gnssTelem.hpp and the window store's behaviour. It
cannot show that the two ends agree: the packer's slot addressing, the bufferSend/bufferRecv
frame-size and config_tracker contract, the gather's length-prefixed stream and the client's
grouping are four places where a mismatch delivers plausible data with the wrong address,
which on sky reads as physics.

What it tests:

  [1] PROVENANCE. Every checked value that comes out the far end must be the one written for
      that (instance, hop, PRN), and the wire must carry exactly the despread PRNs. The
      payload encodes its own address, so a mis-paired frame is caught rather than averaged.
      The comb columns are checked the same way, with their freq_id labels.

  [2] GROUPING BY ABSOLUTE WINDOW, NOT ARRIVAL ORDER. One simulated instance starts late, so
      its Nth frame is not its peers' Nth frame. Grouped by arrival order it would be paired
      with the wrong window, and the fleet sum would look fine and be wrong. Grouped by
      wstart, it lands where it belongs and simply has no data for the earliest windows.

  [3] A DROPPED RECORD IS A HOLE, NOT A SHIFT. One instance omits one record slot. Its
      neighbours must keep their own hops, and the fleet's hop sets must still agree on the
      records that are there.
"""
import argparse
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

K = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
sys.path.insert(0, os.path.join(K, "python", "scripts", "gnss"))
sys.path.insert(0, os.path.join(K, "config"))

import kotekan_src  # noqa: E402
from gnss_broker import telem  # noqa: E402
from gnss_record_layout import (
    record_stride,
    telem_frame_bytes,  # noqa: E402
    telem_max_chan,
)

# Ports clear of the site's live kotekan and broker ports (11025-11070, 12048-12060). A gate
# that collides with production is worse than no gate.
PORT_RECV = 11890
PORT_SERVE = 11891
PORT_REST = 12890

HOPS_PER_RECORD = 2048
FFT_LEN = 16384
REC_PER_FRAME = 4
# Row compaction: the wire carries fewer rows (max_prn) than the record buffer has PRN slots,
# and GnssTelemPack fills them with the PRNs actually despread. MAX_PRN < N_PRN, so slot-indexed
# rows cannot pass this gate.
MAX_PRN = 3
N_PRN = 4
N_ELEM = 2
N_WINDOWS = 12
PRNS = [3, 7, 10, 26]
# One configured PRN that is not being despread. It must occupy no row at all: silence is not a
# row of zeros, and a zero row would be indistinguishable downstream from a satellite that was
# tracked and found nothing.
DEAD_PRN = 7
LIVE_PRNS = [p for p in PRNS if p != DEAD_PRN]

# The three simulated instances. cx51.1 starts late; see [2] above.
INSTANCES = [("cx19.0", 0), ("cx42.1", 0), ("cx51.1", 3)]

REC_STRIDE = record_stride(N_ELEM)
N_CHAN = 5  # < TELEM_MAX_CHAN: the sender's row is narrower than the format's
# ceiling, so the receive buffer is bigger than the frames and this
# exercises the short-frame path through bufferRecv
CHAN_FLOATS = 9  # re, im, energy for each of P, E, L (gnssRecord.hpp)
CHAN_IDS = [5972 + 16 * k for k in range(N_CHAN)]


def _mark(inst, wstart, prn, slot):
    """The value written into row slot `slot`: it encodes its own address.

    instance index x 4e6 + record x 2000 + PRN index x 500 + slot, reproducible on the far
    side, so "did this row come from where the header says it did?" is a direct comparison.
    The PRN term is its index in PRNS with a stride above the largest slot (comb slots run to
    404), so no two addresses the fixture writes share a mark; main() checks that before it
    runs (duplicate_marks).

    The value must stay below 2^24 so it survives the float32 record slot exactly (above that
    the ulp is 2 and correct rows fail the comparison); the largest here is about 8.2e6. The
    instance term must be a fixed index, never hash(): PYTHONHASHSEED is randomised per
    process, so a hash-derived mark would make the result depend on the run.
    """
    ih = [i for i, (n, _) in enumerate(INSTANCES) if n == inst][0]
    rec = wstart // (HOPS_PER_RECORD * FFT_LEN)
    return float(ih * 4000000 + rec * 2000 + PRNS.index(prn) * 500 + slot)


def duplicate_marks():
    """Pairs of addresses the fixture writes the same mark at, over every row and comb slot
    write_record_files fills: a mis-addressing between such a pair would pass. Empty when
    every mark is distinct."""
    slots = list(range(telem._ROW_FLOATS)) + [
        k + ch for k in (100, 200, 300, 400) for ch in range(N_CHAN)
    ]
    seen, dup = {}, []
    for inst, start in INSTANCES:
        for w in range(start, start + N_WINDOWS):
            for r in range(REC_PER_FRAME):
                wstart = (w * REC_PER_FRAME + r) * HOPS_PER_RECORD * FFT_LEN
                for prn in PRNS:
                    for s in slots:
                        addr = (inst, wstart, prn, s)
                        m = _mark(*addr)
                        if m in seen:
                            dup.append((seen[m], addr))
                        seen[m] = addr
    return dup


def write_record_files(dirpath, inst, start_win, drop=()):
    """Write one rawFileWrite-format file of record frames for `inst`, omitting the
    (window, record slot) pairs in `drop`; return its path.

    Each frame is [u32 meta_size][meta][frame], the layout rawFileWrite produces and
    rawFileRead checks on every frame, so the replay stands in for the assembler's output
    rather than taking a special path.

    The comb marks are not a signal. They encode their own (instance, hop, PRN, channel)
    address so a stride error is caught by value, but the resulting E/P/L are arbitrary and
    any discriminator built from them is meaningless.
    """
    meta_size = 12  # GnssChanMetadata: int64 sample_seq + uint32 reserved (0)
    path = os.path.join(dirpath, "%s_0000000.raw" % inst.replace(".", "_"))
    with open(path, "wb") as fh:
        for w in range(start_win, start_win + N_WINDOWS):
            for r in range(REC_PER_FRAME):
                if (w, r) in drop:
                    continue
                wstart = (w * REC_PER_FRAME + r) * HOPS_PER_RECORD * FFT_LEN
                fh.write(struct.pack("<I", meta_size))
                fh.write(struct.pack("<qI", wstart, 0))
                body = [0.0] * (N_PRN * REC_STRIDE + N_PRN * N_CHAN * CHAN_FLOATS)
                for p, prn in enumerate(PRNS):
                    base = p * REC_STRIDE
                    for s in range(telem._ROW_FLOATS):
                        body[base + s] = _mark(inst, wstart, prn, s)
                    body[base + telem.REC_PRN] = float(prn)
                    # Replica energy is what marks a PRN as despread, and it is what the pack
                    # compacts on. DEAD_PRN gets zero and must therefore reach no wire row.
                    body[base + telem.REC_P_ENERGY] = (
                        0.0 if prn == DEAD_PRN else float(prn)
                    )
                    # Element blocks: filled with a sentinel that must never appear on the
                    # wire. The transport ships the record header only, and a stride bug would
                    # smuggle these through looking like data.
                    for s in range(telem._ROW_FLOATS, REC_STRIDE):
                        body[base + s] = -12345.0
                    # The comb block, after the last PRN record, per PRN then channel, exactly
                    # as the assembler writes it (gnssRecord.hpp chan_offset). Each column
                    # encodes its own address, so a stride error anywhere between here and the
                    # broker is caught by value, not by plausibility.
                    cb0 = N_PRN * REC_STRIDE + (p * N_CHAN) * CHAN_FLOATS
                    for ch in range(N_CHAN):
                        cb = cb0 + ch * CHAN_FLOATS
                        body[cb + 0] = _mark(inst, wstart, prn, 100 + ch)  # P
                        body[cb + 1] = _mark(inst, wstart, prn, 200 + ch)
                        body[cb + 2] = float(ch + 1)
                        body[cb + 3] = _mark(inst, wstart, prn, 300 + ch)  # E
                        body[cb + 5] = float(ch + 1)
                        body[cb + 6] = _mark(inst, wstart, prn, 400 + ch)  # L
                        body[cb + 8] = float(ch + 1)
                fh.write(struct.pack("<%df" % len(body), *body))
    return path


def write_config(dirpath):
    """One kotekan config: a sender leg per instance plus the gather, joined over localhost.

    Both halves run in one process on purpose: bufferSend to 127.0.0.1 still exercises the real
    socket path (framing, header contract, frame-size check) without needing two machines.
    """
    lines = [
        "type: config",
        "log_level: info",
        # Root-level default: kotekan requires cpu_affinity on every stage and resolves it by
        # walking up the config tree, so one entry here covers all of them.
        "cpu_affinity: [0, 1, 2, 3]",
        "gnss_pool: {kotekan_metadata_pool: GnssChanMetadata, num_metadata_objects: 2048}",
        "telescope: {name: ICETelescope, num_polarizations: 1, num_dishes: 1,"
        " query_gps: false, require_gps: false}",
        "rest_server: {port: %d}" % PORT_REST,
        # The receive buffer is the ceiling and the senders are narrower, as deployed, hence
        # `allow_short_frames`: it holds the widest row the format admits while every sender
        # here ships N_CHAN columns.
        "telem_buf: {kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: 256,"
        " frame_size: %d}"
        % telem_frame_bytes(REC_PER_FRAME, MAX_PRN, telem_max_chan()),
        "telem_recv: {kotekan_stage: bufferRecv, buf: telem_buf, listen_port: %d,"
        " num_threads: 2, drop_frames: false, use_config_tracker: false,"
        " allow_short_frames: true}" % PORT_RECV,
        "telem_gather: {kotekan_stage: GnssTelemGather, in_buf: telem_buf,"
        " serve_host: 127.0.0.1, serve_port: %d}" % PORT_SERVE,
    ]
    for i, (inst, _start) in enumerate(INSTANCES):
        tag = "s%d" % i
        lines += [
            "%s_rec_buf: {kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: 16,"
            " frame_size: %d}"
            % (tag, (N_PRN * REC_STRIDE + N_PRN * N_CHAN * CHAN_FLOATS) * 4),
            "%s_out_buf: {kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: 64,"
            " frame_size: %d}"
            % (tag, telem_frame_bytes(REC_PER_FRAME, MAX_PRN, N_CHAN)),
            "%s_read: {kotekan_stage: rawFileRead, buf: %s_rec_buf, base_dir: %s,"
            " file_name: %s, file_ext: raw, prefix_hostname: false, end_interrupt: false}"
            % (tag, tag, dirpath, inst.replace(".", "_")),
            "%s_pack: {kotekan_stage: GnssTelemPack, in_buf: %s_rec_buf, out_buf: %s_out_buf,"
            " chain: gps_l5, inst: %s, n_prn: %d, n_elements: %d, max_prn: %d,"
            " records_per_frame: %d, hops_per_record: %d, fft_len: %d, n_chan: %d,"
            " chan_export: true, channel_ids: [%s]}"
            % (
                tag,
                tag,
                tag,
                inst,
                N_PRN,
                N_ELEM,
                MAX_PRN,
                REC_PER_FRAME,
                HOPS_PER_RECORD,
                FFT_LEN,
                N_CHAN,
                ", ".join(str(c) for c in CHAN_IDS),
            ),
            "%s_send: {kotekan_stage: bufferSend, buf: %s_out_buf, server_ip: 127.0.0.1,"
            " server_port: %d, drop_frames: false, use_config_tracker: false}"
            % (tag, tag, PORT_RECV),
        ]
    path = os.path.join(dirpath, "telem_e2e.yaml")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def kotekan_binary():
    """The DPDK build on a cx node, the DPDK-free one elsewhere, falling back to whichever
    exists: /home is NFS with two build trees, and the wrong one fails in a way that reads as a
    config problem."""
    host = socket.gethostname().split(".")[0]
    cand = (
        [kotekan_src.src("build", "kotekan", "kotekan")]
        if host.startswith("cx")
        else [kotekan_src.src("build_nodpdk", "kotekan", "kotekan")]
    )
    cand.append(kotekan_src.src("build_nodpdk", "kotekan", "kotekan"))
    cand.append(kotekan_src.src("build", "kotekan", "kotekan"))
    for c in cand:
        if os.path.exists(c):
            return c
    sys.exit("no kotekan binary found under %s" % kotekan_src.SRC)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--keep",
        action="store_true",
        help="keep the scratch dir (config, record files, kotekan.log) after a pass; "
        "a failure always keeps it",
    )
    ap.add_argument("--timeout", type=float, default=25.0)
    a = ap.parse_args()

    dup = duplicate_marks()
    if dup:
        raise SystemExit(
            "%d fixture marks are written at two addresses, e.g. %s and %s: a "
            "mis-addressing between them would pass" % ((len(dup),) + dup[0])
        )

    d = tempfile.mkdtemp(prefix="telem_e2e-")
    # cx51.1 drops record slot 2 of absolute window 8 (its sixth window). See [3].
    drops = {"cx51.1": {(5 + 3, 2)}}
    for inst, start in INSTANCES:
        write_record_files(d, inst, start, drop=drops.get(inst, ()))
    cfg = write_config(d)
    log = os.path.join(d, "kotekan.log")
    binary = kotekan_binary()
    print("kotekan  %s" % binary)
    print("scratch  %s" % d)

    # --bind-address is required: without it kotekan's REST server binds its built-in default
    # 0.0.0.0:12048, which on a cx node is the production REST port.
    proc = subprocess.Popen(
        [binary, "--config", cfg, "--bind-address", "127.0.0.1:%d" % PORT_REST],
        stdout=open(log, "wb"),
        stderr=subprocess.STDOUT,
    )
    client = telem.TelemClient(host="127.0.0.1", port=PORT_SERVE, depth=64, retry_s=0.5)
    fails = []
    try:
        client.start()
        t0 = time.time()
        # Wait until every sender has delivered, or the timeout. The senders replay a finite
        # file and then stop, so "enough" is a fixed expected frame count, not a duration.
        want = len(INSTANCES) * N_WINDOWS
        while time.time() - t0 < a.timeout:
            if client.frames >= want:
                break
            time.sleep(0.25)
        time.sleep(1.0)  # let the tail arrive so `lag` has a newer window to hide

        st = client.stats()
        print(
            "frames   %d (wanted %d)  gaps %d  bad %d  connected %s"
            % (client.frames, want, st["gaps"], st["bad"], st["connected"])
        )
        if client.frames == 0:
            fails.append("NO FRAMES arrived -- see %s" % log)

        # -- the gather's own view, over REST ------------------------------------------------
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:%d/telem_gather/get_stats" % PORT_REST, timeout=5
            ) as r:
                gs = json.load(r)
            print(
                "gather   %d senders, %d bad, %d client drops"
                % (len(gs["senders"]), gs["bad_frames"], gs["client_drops"])
            )
            if gs["bad_frames"]:
                fails.append("gather rejected %d frames" % gs["bad_frames"])
            if len(gs["senders"]) != len(INSTANCES):
                fails.append(
                    "gather saw %d senders, expected %d"
                    % (len(gs["senders"]), len(INSTANCES))
                )
        except Exception as e:
            fails.append("gather /get_stats unreachable: %s" % e)

        wins = client.windows("gps_l5", lag=1)
        print(
            "windows  %d held: %s..%s"
            % (len(wins), wins[0] if wins else "-", wins[-1] if wins else "-")
        )

        # -- [1] PROVENANCE: every checked value is the one written for its address ----------
        checked = 0
        for w in wins:
            for inst, f in client.frame_set("gps_l5", w).items():
                for r in range(f.n_rec):
                    if not f.has_record(r):
                        continue
                    wstart = f.wstart0 + r * HOPS_PER_RECORD * FFT_LEN
                    # [1a] Compaction: exactly the despread PRNs, and nothing else. prns() is the
                    # frame's row map, read from record slot 0; a map rebuilt mid-frame would
                    # show in the per-slot marks below, not here.
                    got_prns = f.prns()
                    if got_prns != LIVE_PRNS:
                        fails.append(
                            "%s w%d r%d: wire carries PRNs %s, expected %s "
                            "(row compaction)" % (inst, w, r, got_prns, LIVE_PRNS)
                        )
                    if DEAD_PRN in got_prns:
                        fails.append(
                            "%s w%d r%d: PRN %d was never despread but occupies a wire "
                            "row -- silence is being shipped as data"
                            % (inst, w, r, DEAD_PRN)
                        )
                    for prn in LIVE_PRNS:
                        row = f.row(r, prn)
                        if row is None:
                            fails.append(
                                "%s w%d r%d: PRN %d missing" % (inst, w, r, prn)
                            )
                            continue
                        for s in (
                            telem.REC_DOPPLER,
                            telem.REC_CP,
                            telem.REC_P_RE,
                            telem.REC_CPHASE,
                            telem.REC_TRIM_INC,
                            telem.REC_SKY_IM,
                        ):
                            want_v = _mark(inst, wstart, prn, s)
                            # Exact: the mark is an integer below 2^24, so a float32 round trip
                            # is lossless and any difference is a real mis-addressing.
                            if row[s] != want_v:
                                fails.append(
                                    "%s w%d r%d PRN %d slot %d: %g != %g (the row came "
                                    "from a different record or instance)"
                                    % (inst, w, r, prn, s, row[s], want_v)
                                )
                        if any(v == -12345.0 for v in row):
                            fails.append(
                                "%s w%d r%d PRN %d: an ELEMENT block leaked into the "
                                "wire row -- the stride is wrong" % (inst, w, r, prn)
                            )
                        # [1b] The comb. Same provenance rule: each column must carry the value
                        # written for that (instance, hop, PRN, channel), divided by its replica
                        # energy, and its label must be the freq_id the sender was configured
                        # with. Columns that survive under wrong labels are worse than no comb:
                        # the delay fit downstream would be confidently wrong.
                        cmb = [
                            (fid, P, en[1]) for fid, _E, P, _L, en in f.comb_epl(r, prn)
                        ]
                        if len(cmb) != N_CHAN:
                            fails.append(
                                "%s w%d r%d PRN %d: %d comb columns, expected %d"
                                % (inst, w, r, prn, len(cmb), N_CHAN)
                            )
                        # E/L too: a tap that survives with the wrong value is the failure a
                        # shape-only check would miss.
                        for ch, (fid, E, P, L, _en) in enumerate(f.comb_epl(r, prn)):
                            for tag, got, off in (
                                ("E", E.real, 300),
                                ("L", L.real, 400),
                            ):
                                wv = _mark(inst, wstart, prn, off + ch) / (ch + 1)
                                if abs(got - wv) > 1e-3 * max(1.0, abs(wv)):
                                    fails.append(
                                        "%s win%d r%d PRN %d ch%d %s: %g != %g"
                                        % (inst, w, r, prn, ch, tag, got, wv)
                                    )
                        for ch, (fid, A, e) in enumerate(cmb):
                            if fid != CHAN_IDS[ch]:
                                fails.append(
                                    "%s w%d r%d PRN %d ch%d: freq_id %d != %d"
                                    % (inst, w, r, prn, ch, fid, CHAN_IDS[ch])
                                )
                            want_re = _mark(inst, wstart, prn, 100 + ch) / (ch + 1)
                            if abs(A.real - want_re) > 1e-3 * max(1.0, abs(want_re)):
                                fails.append(
                                    "%s w%d r%d PRN %d ch%d: comb %g != %g"
                                    % (inst, w, r, prn, ch, A.real, want_re)
                                )
                        checked += 1
        print(
            "rows     %d checked against their own written address (record + %d-column comb)"
            % (checked, N_CHAN)
        )
        if checked == 0:
            fails.append("no rows were checked -- the gate could not have failed")

        # -- [2] GROUPING BY ABSOLUTE WINDOW, not arrival order ------------------------------
        late = INSTANCES[2][0]
        late_start = INSTANCES[2][1]
        early_only = [w for w in wins if w < late_start]
        both = [w for w in wins if w >= late_start]
        for w in early_only:
            if late in client.frame_set("gps_l5", w):
                fails.append(
                    "%s appears in window %d, before it started: frames are being "
                    "grouped by ARRIVAL ORDER, which is the whole defect" % (late, w)
                )
        joined = [w for w in both if late in client.frame_set("gps_l5", w)]
        print(
            "late     %s absent from %d early windows, present in %d/%d shared"
            % (late, len(early_only), len(joined), len(both))
        )
        if both and not joined:
            fails.append("%s never joined a shared window" % late)
        if not early_only:
            fails.append("no early-only windows survived: [2] could not have failed")

        # -- hop sets agree exactly across instances, with no tolerance ----------------------
        agreed = 0
        for w in both:
            fs = client.frame_set("gps_l5", w)
            if len(fs) < 2:
                continue
            sets = {
                inst: set(f.hop(r) for r in range(f.n_rec) if f.has_record(r))
                for inst, f in fs.items()
            }
            full = max(sets.values(), key=len)
            for inst, s in sets.items():
                if not s <= full:
                    fails.append(
                        "window %d: %s has hops its peers do not -- %s"
                        % (w, inst, sorted(s - full))
                    )
            if all(s == full for s in sets.values()):
                agreed += 1
        shared = len([w for w in both if len(client.frame_set("gps_l5", w)) > 1])
        # The one expected exception is the window whose record this script deliberately drops
        # ([3]); everything else must agree exactly, with no tolerance.
        print(
            "aligned  %d/%d shared windows have identical hop sets across instances "
            "(1 exception expected: the dropped record)" % (agreed, shared)
        )
        if shared and agreed < shared - 1:
            fails.append(
                "%d shared windows disagree on hops; only the deliberate drop should"
                % (shared - agreed)
            )

        # -- [3] the dropped record is a hole, and its neighbours keep their hops -------------
        dw = 5 + 3
        fs = client.frame_set("gps_l5", dw)
        if late in fs:
            f = fs[late]
            if f.has_record(2):
                fails.append(
                    "window %d: %s should have dropped record slot 2" % (dw, late)
                )
            ref = [g for i, g in fs.items() if i != late]
            if ref:
                mine = set(f.hop(r) for r in range(f.n_rec) if f.has_record(r))
                theirs = set(
                    ref[0].hop(r) for r in range(ref[0].n_rec) if ref[0].has_record(r)
                )
                if len(mine) != 3 or not mine < theirs:
                    fails.append(
                        "window %d: the dropped record SHIFTED the others (%s vs %s)"
                        % (dw, sorted(mine), sorted(theirs))
                    )
                else:
                    print(
                        "hole     window %d slot 2 missing; the other 3 kept their own hops"
                        % dw
                    )
        else:
            fails.append(
                "window %d not received from %s -- [3] could not have failed"
                % (dw, late)
            )

        # -- the shape the broker consumes ---------------------------------------------------
        got, now = client.coherent_source("gps_l5", prns=LIVE_PRNS)
        print(
            "coherent %d instances, %d PRNs on cx19.0, fleet_now hop %d"
            % (len(got), len(got.get("cx19.0", {})), now)
        )
        if len(got) != len(INSTANCES):
            fails.append(
                "coherent_source saw %d instances, expected %d"
                % (len(got), len(INSTANCES))
            )
        hopsets = [
            set(g.get(LIVE_PRNS[0], {})) for g in got.values() if g.get(LIVE_PRNS[0])
        ]
        if len(hopsets) > 1:
            common = set.intersection(*hopsets)
            if not common:
                fails.append(
                    "no hop is shared by every instance -- the fleet cannot combine"
                )
            print(
                "common   %d hops shared by all %d instances"
                % (len(common), len(hopsets))
            )
    finally:
        client.stop()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    if fails:
        print("\nFAILED (%d):" % len(fails))
        for m in fails[:20]:
            print("  - %s" % m)
        print("\nlog: %s" % log)
        return 1
    print(
        "\nPASS -- packer, bufferSend/Recv, gather, socket and client all agree end to end"
    )
    if not a.keep:
        shutil.rmtree(d, ignore_errors=True)
    else:
        print("kept %s" % d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
