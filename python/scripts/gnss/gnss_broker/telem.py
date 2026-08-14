"""TASK #59: the broker's end of the frame-synced tracker telemetry stream.

Connects to the GATHER instance (a small kotekan on the broker host: one bufferRecv for the
whole fleet, plus GnssTelemGather) and keeps a short ring of recent WINDOWS, each holding
whichever instances have reported it.

WHAT THIS REPLACES, AND WHY IT IS NOT JUST FASTER. The broker used to make ~60 REST round
trips per cycle -- 12 instances x 5 chains -- each landing at a different wall time, and then
had to work out afterwards which instance and which window each reply described. Every one of
those inferences went wrong somewhere in a single week:

  * #53 -- /get_spectrum windows were "whatever accumulated since your last GET", so no two
    instances ever summed the same records, and there was no way to ask for the past.
  * #52 -- the cross-instance delay fit then absorbed that misalignment into a free phase per
    instance: it FITTED what it should have DERIVED.
  * #33 -- the carrier-rate feed differenced `res_cycles` across a "served row" that silently
    changed instance. One instance reads 0.82 Hz; the served row read 4.92. Six times worse,
    from addressing alone, and it looked exactly like physics.
  * #46 -- 0.105 s of record-time spread between instances that nothing in the design buffers.

Here the address travels WITH the data: every frame carries its chain, its instance, and an
ABSOLUTE window index computed from the F-engine's own sample counter
(wstart / (records_per_frame * hops_per_record * fft_len)) by each sender from the same three
configured integers. Grouping instances is an exact integer match. Nothing is inferred.

TWO RULES THE REST OF THE BROKER MUST KEEP:

  1. ⚠️ NEVER DIFFERENCE AN ACCUMULATOR ACROSS INSTANCES. What arrives here is per-record
     INCREMENTS (gnssRecord.hpp slots 15/19) and per-record correlations. Accumulate ONCE, per
     instance, over a run this module has certified contiguous (`record_stream` returns the
     gap flags for exactly that). `res_cycles` from /get_status is the thing that was wrong;
     do not rebuild it here.

  2. ⚠️ A WINDOW IS NOT COMPLETE THE MOMENT IT APPEARS. Senders are independent, so the newest
     window is always still filling. Every accessor takes `lag` (default 1 window, ~168 ms) and
     skips that many of the newest -- cheaper and more honest than waiting on a quorum that a
     down instance would never deliver.
"""
import array
import collections
import socket
import struct
import threading
import time

from .transport import _log, _log_rl

# gnssTelem.hpp -- keep in step with TelemHeader. The magic and version are checked on every
# frame, so a drift here surfaces immediately as "no frames" rather than as wrong numbers.
_MAGIC = 0x314C5447
_VERSION = 1
_HDR = struct.Struct("<IHHHHHHIIQQqdII16s16s")
_HDR_BYTES = 96
# gnssRecord.hpp RECORD_FLOATS. Verified against every frame's own n_row field, because a
# tracker rebuilt with a wider record and a broker that was not is precisely the silent
# mis-stride this transport exists to stop tolerating.
_ROW_FLOATS = 26

# Row slots we name here (gnssRecord.hpp). The rest of the row travels intact and is available
# through `row()` -- this is a transport, not a schema.
REC_PRN = 0
REC_DOPPLER = 1
REC_CP = 2
REC_P_RE = 3
REC_P_IM = 4
REC_P_ENERGY = 5
REC_NCHAN = 6
REC_E_ENERGY = 7
REC_L_ENERGY = 8
REC_UTC = 9
REC_E_RE = 11
REC_E_IM = 12
REC_L_RE = 13
REC_L_IM = 14
REC_CPHASE = 15
REC_PH_RE = 16
REC_PH_IM = 17
REC_PH_ENERGY = 18
REC_TRIM_INC = 19
REC_SKY_RE = 24
REC_SKY_IM = 25

assert _HDR.size == _HDR_BYTES, "TelemHeader struct format does not match gnssTelem.hpp"


class TelemFrame(object):
    """One sender's records for one window. Rows are decoded LAZILY.

    At 60 senders x 23.84 frames/s the reader thread sees ~1430 frames/s, and unpacking 4160
    floats on every one of them would cost more than the whole broker cycle. The header is 96
    bytes of struct.unpack; the payload is kept as bytes and decoded only for the PRNs someone
    actually asks about.
    """

    __slots__ = ("chain", "inst", "win", "seq", "n_rec", "n_prn", "n_chan", "n_elem",
                 "hops_per_record", "fft_len", "wstart0", "utc0", "present", "_buf", "_idx",
                 "rx")

    def __init__(self, hdr, buf, rx):
        (_magic, _ver, self.n_rec, self.n_prn, _n_row, self.n_chan, self.n_elem,
         self.hops_per_record, self.fft_len, self.win, self.seq, self.wstart0, self.utc0,
         self.present, _pad, chain, inst) = hdr
        self.chain = chain.split(b"\0", 1)[0].decode("ascii", "replace")
        self.inst = inst.split(b"\0", 1)[0].decode("ascii", "replace")
        self._buf = buf
        self._idx = None  # prn -> row index, built on first use
        self.rx = rx

    def has_record(self, r):
        """Was record slot r filled? A missing record is a HOLE AT A KNOWN INDEX, never a shift."""
        return bool(self.present & (1 << r))

    def hop(self, r):
        """Absolute F-engine hop index of record slot r -- the same key /get_records used."""
        return (self.wstart0 + r * self.hops_per_record * self.fft_len) // self.fft_len

    def _index(self):
        if self._idx is None:
            idx = {}
            # PRN lives in row slot 0 of every row (the assembler writes it even for a PRN that
            # did not run this window), so the map is read from the data rather than assumed
            # from a configured PRN list the broker would have to keep in step.
            for p in range(self.n_prn):
                off = _HDR_BYTES + p * _ROW_FLOATS * 4
                prn = int(struct.unpack_from("<f", self._buf, off)[0] + 0.5)
                if prn > 0:
                    idx[prn] = p
            self._idx = idx
        return self._idx

    def prns(self):
        return sorted(self._index().keys())

    def row(self, r, prn):
        """The full gnssRecord.hpp header row for `prn` in record slot r, or None.

        Returns an array('f') of RECORD_FLOATS. None when the slot was not filled or the PRN is
        not carried by this sender.
        """
        p = self._index().get(int(prn))
        if p is None or not self.has_record(r):
            return None
        off = _HDR_BYTES + ((r * self.n_prn) + p) * _ROW_FLOATS * 4
        a = array.array("f")
        a.frombytes(self._buf[off:off + _ROW_FLOATS * 4])
        return a

    def utc(self, r, prn):
        """Capture UTC of this record, as the assembler stamped it (row slots 9-10, a double).

        ⚠️ DIAGNOSTIC ONLY. Never a collation key: #46 measured 0.105 s of spread between
        instances on records whose wstart -- the thing this module keys on -- was identical.
        """
        p = self._index().get(int(prn))
        if p is None or not self.has_record(r):
            return 0.0
        off = _HDR_BYTES + ((r * self.n_prn) + p) * _ROW_FLOATS * 4 + REC_UTC * 4
        return struct.unpack_from("<d", self._buf, off)[0]


class TelemClient(object):
    """Reader thread + a bounded per-chain window ring.

    Reconnects forever with a fixed backoff. A gather that is down must cost the broker
    NOTHING: every accessor simply returns empty, and the caller falls back to the REST path
    it already has. That is the whole migration strategy -- both feeds live side by side until
    the new one has been shown, on sky, to be at least as good.
    """

    def __init__(self, host="127.0.0.1", port=11061, depth=64, retry_s=5.0):
        self.host = host
        self.port = port
        self.depth = int(depth)
        self.retry_s = float(retry_s)
        self._lock = threading.Lock()
        # chain -> OrderedDict{win: {inst: TelemFrame}}, oldest first, capped at `depth`
        self._store = {}
        self._thread = None
        self._stop = threading.Event()
        self.connected = False
        self.frames = 0
        self.bad = 0
        self.connects = 0
        self.last_rx = 0.0
        self._seen_seq = {}   # (chain, inst) -> last seq
        self.gaps = 0

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="telem", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.create_connection((self.host, self.port), timeout=10.0)
                sock.settimeout(30.0)
                self.connected = True
                self.connects += 1
                _log("telem: connected to gather %s:%d" % (self.host, self.port))
                self._read_loop(sock)
            except Exception as e:
                _log_rl("telem-conn", "telem: gather %s:%d unavailable (%s) -- the REST path is "
                        "unaffected" % (self.host, self.port, e))
            finally:
                self.connected = False
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
            if not self._stop.is_set():
                self._stop.wait(self.retry_s)

    def _recv_exactly(self, sock, n):
        chunks = []
        got = 0
        while got < n:
            b = sock.recv(n - got)
            if not b:
                raise IOError("gather closed the connection")
            chunks.append(b)
            got += len(b)
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    def _read_loop(self, sock):
        while not self._stop.is_set():
            (length,) = struct.unpack("<I", self._recv_exactly(sock, 4))
            # A frame is delivered whole or the gather closes the connection (it never
            # half-writes), so a length outside the plausible range means the stream is not
            # what we think it is -- reconnect rather than parse garbage.
            if length < _HDR_BYTES or length > (1 << 24):
                raise IOError("implausible frame length %d" % length)
            buf = self._recv_exactly(sock, length)
            hdr = _HDR.unpack_from(buf, 0)
            if hdr[0] != _MAGIC or hdr[1] != _VERSION or hdr[4] != _ROW_FLOATS:
                self.bad += 1
                _log_rl("telem-bad", "telem: rejecting a frame (magic %#x v%d n_row %d, want "
                        "%#x v%d %d) -- a tracker and this broker are on different builds"
                        % (hdr[0], hdr[1], hdr[4], _MAGIC, _VERSION, _ROW_FLOATS))
                continue
            self._store_frame(TelemFrame(hdr, buf, time.time()))

    def _store_frame(self, f):
        with self._lock:
            self.frames += 1
            self.last_rx = f.rx
            key = (f.chain, f.inst)
            prev = self._seen_seq.get(key)
            # The SENDER'S OWN counter is the only thing that can say a frame was lost. A rate
            # that looks right can still be missing every fourth frame.
            if prev is not None and f.seq > prev + 1:
                self.gaps += f.seq - prev - 1
            self._seen_seq[key] = f.seq
            ring = self._store.get(f.chain)
            if ring is None:
                ring = self._store[f.chain] = collections.OrderedDict()
            if f.win not in ring:
                # Eviction must drop the OLDEST WINDOW, not the least-recently-written one: a
                # laggard instance opening a window its peers finished long ago would otherwise
                # push it to the front of the ring and evict a NEWER window instead. Senders
                # are independent processes, so out-of-order window opens are normal, not an
                # error -- which is why the ring is re-sorted rather than assumed monotone.
                newest = next(reversed(ring)) if ring else None
                ring[f.win] = {}
                if newest is not None and f.win < newest:
                    for w in sorted(ring):
                        ring.move_to_end(w)
                while len(ring) > self.depth:
                    ring.popitem(last=False)
            slot = ring.get(f.win)
            if slot is None:
                return  # just evicted as too old to matter
            slot[f.inst] = f

    # -- accessors ---------------------------------------------------------------------------
    def chains(self):
        with self._lock:
            return sorted(self._store.keys())

    def windows(self, chain, lag=1):
        """Window indices held for `chain`, oldest first, excluding the `lag` newest.

        ⚠️ THE LAG IS NOT OPTIONAL PADDING. Senders are independent processes on six machines;
        the newest window is always still arriving. Taking it would silently combine a
        four-instance version of a window with the twelve-instance version of its neighbours,
        which reads as a fleet that keeps changing size.
        """
        with self._lock:
            ws = sorted(self._store.get(chain, {}).keys())
        return ws[:-lag] if lag > 0 else ws

    def frame_set(self, chain, win):
        """{inst: TelemFrame} for one window."""
        with self._lock:
            return dict(self._store.get(chain, {}).get(win, {}))

    def coherent_source(self, chain, prns=None, n_win=8, lag=1):
        """The `got` structure fleet_coherent builds from /get_records, from telemetry instead.

        Returns (got, fleet_now_hop) with got = {inst: {prn: {hop: (A, energy)}}} where
        A = (P_re + i P_im)/P_energy and energy = P_energy -- the SAME two numbers the combiner
        publishes (it forms ar = gr/energy and exports that), so the estimator downstream is
        untouched and the two feeds are directly comparable. That is the point: one estimator,
        two transports, and the difference between them is then a measurement rather than an
        argument.

        Records with zero energy are dropped exactly as /get_records drops them (a PRN that did
        not run is silence, not a row of zeros).
        """
        wins = self.windows(chain, lag=lag)
        if not wins:
            return {}, 0
        wins = wins[-int(n_win):]
        want = None if prns is None else set(int(p) for p in prns)
        got = {}
        fleet_now = 0
        for w in wins:
            for inst, f in self.frame_set(chain, w).items():
                per = got.setdefault(inst, {})
                for r in range(f.n_rec):
                    if not f.has_record(r):
                        continue
                    hop = f.hop(r)
                    if hop > fleet_now:
                        fleet_now = hop
                    for prn in f.prns():
                        if want is not None and prn not in want:
                            continue
                        row = f.row(r, prn)
                        if row is None:
                            continue
                        e = row[REC_P_ENERGY]
                        if e <= 0.0:
                            continue
                        per.setdefault(prn, {})[hop] = (
                            complex(row[REC_P_RE] / e, row[REC_P_IM] / e), e)
        return got, fleet_now

    def record_stream(self, chain, inst, prn, n_win=16, lag=1):
        """One instance's records for one PRN, in hop order, with the gaps marked.

        THE POINT OF THIS FUNCTION. Carrier phase must be accumulated over a run of records
        that is provably contiguous AND provably from ONE instance -- both were assumptions
        before, and both were wrong (see the module note, #33). Here the instance is the key
        and contiguity is checked against the hop grid rather than against a record count or a
        wall-clock span.

        Returns a list of dicts, oldest first:
            {hop, dphi_cmd, trim_inc, A, energy, gap}
        where `gap` is the number of MISSING records immediately before this one (0 = the arc
        continues). An accumulation must reset wherever gap != 0; nothing here does that for
        the caller, because what to do at a break is a loop decision and loops live upstream.
        """
        out = []
        step = None
        prev_hop = None
        for w in self.windows(chain, lag=lag)[-int(n_win):]:
            f = self.frame_set(chain, w).get(inst)
            if f is None:
                continue
            if step is None:
                step = f.hops_per_record
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                row = f.row(r, prn)
                if row is None:
                    continue
                e = row[REC_P_ENERGY]
                if e <= 0.0:
                    continue
                hop = f.hop(r)
                gap = 0
                if prev_hop is not None and step:
                    gap = int((hop - prev_hop) // step) - 1
                    if gap < 0:
                        gap = 0
                prev_hop = hop
                out.append({"hop": hop,
                            "utc": f.utc(r, prn),
                            "dphi_cmd": row[REC_CPHASE],
                            "trim_inc": row[REC_TRIM_INC],
                            "A": complex(row[REC_P_RE] / e, row[REC_P_IM] / e),
                            "energy": e,
                            "doppler_hz": row[REC_DOPPLER],
                            "code_phase_chips": row[REC_CP],
                            "e_pow": row[REC_E_ENERGY],
                            "l_pow": row[REC_L_ENERGY],
                            "gap": gap})
        return out

    def stats(self, stale_after_s=5.0):
        """Transport health -- and the alignment check, served rather than inferred.

        `spread` is max(win) - min(win) over the LIVE instances of a chain. Zero or one is the
        transport working; anything larger is the misalignment this whole change exists to make
        visible instead of leaving it to be found six weeks later as a physics anomaly.

        ⚠️ LIVE ONLY, and that is load-bearing. A stopped instance keeps its last window
        forever, so one dead sender drives the raw spread to the number of windows since it
        died -- measured 984 within a minute on 2026-08-14 while the nine live instances sat at
        1. Reported raw, this number would cry wolf on every instance death and be ignored by
        the second week. The stale ones are listed BY NAME instead, because "who left" is the
        actionable half.
        """
        now = time.time()
        with self._lock:
            per_chain = {}
            for chain, ring in self._store.items():
                last, seen = {}, {}
                for win, insts in ring.items():
                    for inst, f in insts.items():
                        if win > last.get(inst, -1):
                            last[inst] = win
                        if f.rx > seen.get(inst, 0.0):
                            seen[inst] = f.rx
                live = {i: w for i, w in last.items()
                        if stale_after_s <= 0 or (now - seen.get(i, 0.0)) <= stale_after_s}
                stale = sorted(set(last) - set(live))
                row = {"instances": len(last), "live": len(live), "stale": stale,
                       "windows_held": len(ring)}
                if live:
                    row.update({"win_min": min(live.values()),
                                "win_max": max(live.values()),
                                "spread": max(live.values()) - min(live.values())})
                per_chain[chain] = row
            return {"connected": self.connected,
                    "frames": self.frames,
                    "gaps": self.gaps,
                    "bad": self.bad,
                    "connects": self.connects,
                    "age_s": (now - self.last_rx) if self.last_rx else None,
                    "chains": per_chain}


_shared = {}
_shared_lock = threading.Lock()


def shared_client(host, port, depth=64):
    """ONE reader thread per process, shared by every chain.

    broker_multi runs all five chains as threads of one process (task #27), and the gather
    stream carries every chain on one connection -- so a client per chain would open five
    connections and decode the same 24 MB/s five times over, four of them discarded. The store
    is already keyed by chain, so sharing costs nothing and the accessors are unchanged.
    """
    key = (host, int(port))
    with _shared_lock:
        c = _shared.get(key)
        if c is None:
            c = _shared[key] = TelemClient(host=host, port=int(port), depth=depth).start()
    return c


def parse_endpoint(s, default_port=11061):
    """"host:port" / "host" / ":port" -> (host, port). Empty or None -> None."""
    if not s:
        return None
    s = s.strip()
    if ":" in s:
        h, _, p = s.rpartition(":")
        return (h or "127.0.0.1", int(p))
    return (s, default_port)
