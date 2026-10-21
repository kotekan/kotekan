"""The broker's end of the frame-synced tracker telemetry stream.

Connects to the gather instance (a small kotekan on the broker host: one bufferRecv for the
whole fleet, plus GnssTelemGather) and keeps a short ring of recent windows per chain, each
holding whichever instances have reported it.

Every frame carries its own address: its chain, its instance, and an absolute window index
that each sender computes from the F-engine's sample counter,
wstart / (records_per_frame * hops_per_record * fft_len), with the same three configured
integers. Grouping instances is an exact integer match; nothing is inferred from when a reply
arrived.

Two rules for consumers:

  1. Never difference an accumulator across instances: each instance's running total has its
     own origin. What arrives here is per-record increments (gnssRecord.hpp slots 15/19) and
     per-record correlations. Accumulate once, per instance, over a run known to be
     contiguous; do not rebuild a running total such as /get_status res_cycles here.

  2. A window is not complete when it first appears. Senders are independent, so the newest
     window is always still filling. windows() and coherent_source() take `lag` (default 1
     window, ~42 ms) and skip that many of the newest, rather than wait on a quorum that a
     down instance would never deliver.
"""
import array
import collections
import socket
import struct
import threading
import time

from . import iface
from .transport import _log, _log_rl

# The wire format, from the interface golden (gnssTelem.hpp TELEM_MAGIC, TELEM_VERSION and
# TelemHeader, gnssRecord.hpp RECORD_FLOATS and the slots). Magic, version and n_row are checked
# on every frame, so a sender on another format shows as rejected frames, not wrong numbers.
_G = iface.golden()
_MAGIC = _G.const("telem", "TELEM_MAGIC")
_VERSION = _G.const("telem", "TELEM_VERSION")
_HDR = struct.Struct(_G.struct_format("telem", "TelemHeader"))
_HDR_BYTES = _G.const("telem", "TELEM_HEADER_BYTES")
_MAX_CHAN = _G.const("telem", "TELEM_MAX_CHAN")
_MAX_REC = _G.const("telem", "TELEM_MAX_REC")  # record slots a frame may carry
_CHAN_FLOATS = _G.const("record.chan", "CHAN_FLOATS")
# A comb column: prompt, early, late, each re, im, replica energy.
CHAN_RE = _G.slot("record.chan", "CHAN_RE")
CHAN_IM = _G.slot("record.chan", "CHAN_IM")
CHAN_ENERGY = _G.slot("record.chan", "CHAN_ENERGY")
CHAN_E_RE = _G.slot("record.chan", "CHAN_E_RE")
CHAN_E_IM = _G.slot("record.chan", "CHAN_E_IM")
CHAN_E_ENERGY = _G.slot("record.chan", "CHAN_E_ENERGY")
CHAN_L_RE = _G.slot("record.chan", "CHAN_L_RE")
CHAN_L_IM = _G.slot("record.chan", "CHAN_L_IM")
CHAN_L_ENERGY = _G.slot("record.chan", "CHAN_L_ENERGY")
# Checked against every frame's own n_row: a tracker built with a wider record than this broker
# would otherwise be read at the wrong stride.
_ROW_FLOATS = _G.const("record.row", "RECORD_FLOATS")

# Row slots we name here. The rest of the row travels intact and is available through `row()`
# -- this is a transport, not a schema. The golden carries each slot's meaning.
REC_PRN = _G.slot("record.row", "REC_PRN")
REC_DOPPLER = _G.slot("record.row", "REC_DOPPLER")
REC_CP = _G.slot("record.row", "REC_CP")
REC_P_RE = _G.slot("record.row", "REC_P_RE")
REC_P_IM = _G.slot("record.row", "REC_P_IM")
REC_P_ENERGY = _G.slot("record.row", "REC_P_ENERGY")
REC_NCHAN = _G.slot("record.row", "REC_NCHAN")
REC_E_ENERGY = _G.slot("record.row", "REC_E_ENERGY")
REC_L_ENERGY = _G.slot("record.row", "REC_L_ENERGY")
REC_UTC = _G.slot("record.row", "RECORD_UTC_SLOT")
REC_E_RE = _G.slot("record.row", "REC_E_RE")
REC_E_IM = _G.slot("record.row", "REC_E_IM")
REC_L_RE = _G.slot("record.row", "REC_L_RE")
REC_L_IM = _G.slot("record.row", "REC_L_IM")
REC_CPHASE = _G.slot("record.row", "REC_CPHASE")
# Per-(instance, PRN) carrier state that no external measurement reaches: ang0 as the kernel
# used it, and the Phi-cache staleness.
REC_ANG0 = _G.slot("record.row", "REC_ANG0")
REC_PHI_DDOP = _G.slot("record.row", "REC_PHI_DDOP")
# The rotation exp(-i*phi0) applied to slots 3/4 and to every comb column. Its origin is
# arbitrary per instance, so multiply by exp(+i*phi0) to put instances on one reference before
# combining them within a record.
REC_PHI0 = _G.slot("record.row", "REC_PHI0")
REC_PH_RE = _G.slot("record.row", "REC_PH_RE")
REC_PH_IM = _G.slot("record.row", "REC_PH_IM")
REC_PH_ENERGY = _G.slot("record.row", "REC_PH_ENERGY")
REC_TRIM_INC = _G.slot("record.row", "REC_TRIM_INC")
REC_SKY_RE = _G.slot("record.row", "REC_SKY_RE")
REC_SKY_IM = _G.slot("record.row", "REC_SKY_IM")
# Bright-satellite projection cost on this row, cos^2 in [0, 1] of the satellite's live element
# weights inside the projected subspace (the row kept (1 - cost)^2 of its prompt power); -1 =
# no basis in force, a source, unsteered or cold. See gnssRecord.hpp REC_PROJ_COST.
REC_PROJ_COST = _G.slot("record.row", "REC_PROJ_COST")
# The rank of the projection basis in force on the record (0 = none) and the OR of its
# sources over the sender's channels: 1 own row, 2 sibling chain, 4 probe stack (gnssRecord.hpp).
REC_PROJ_K = _G.slot("record.row", "REC_PROJ_K")
REC_PROJ_SRC = _G.slot("record.row", "REC_PROJ_SRC")

# Widest row the format admits: gnss::telem_row_floats(TELEM_MAX_CHAN), i.e. RECORD_FLOATS +
# TELEM_MAX_CHAN * CHAN_FLOATS; TELEM_MAX_CHAN is also the size of the header's freq_id array.
# Not the stride: each sender ships only the comb columns it despreads and declares that width
# in the header (`row_total`). Striding by this constant reads a narrower sender's rows at the
# wrong offsets: plausible numbers, wrong PRNs.
_MAX_ROW_TOTAL = _ROW_FLOATS + _MAX_CHAN * _CHAN_FLOATS

assert _HDR.size == _HDR_BYTES, "TelemHeader struct format does not match its size"


class TelemFrame(object):
    """One sender's records for one window. Rows are decoded lazily.

    Every sender delivers one frame per window (~24 per second), and decoding every payload on
    arrival would spend most of the work on rows nobody reads. The 112-byte header is unpacked
    once; the payload is kept as bytes and decoded only for the PRNs someone asks about.
    """

    __slots__ = (
        "chain",
        "inst",
        "win",
        "seq",
        "n_rec",
        "n_prn",
        "n_chan",
        "n_elem",
        "hops_per_record",
        "fft_len",
        "wstart0",
        "utc0",
        "present",
        "_buf",
        "_idx",
        "rx",
        "max_chan",
        "row_total",
        "chan_ids",
    )

    def __init__(self, hdr, buf, rx):
        (
            _magic,
            _ver,
            self.n_rec,
            self.n_prn,
            _n_row,
            self.n_chan,
            self.n_elem,
            self.hops_per_record,
            self.fft_len,
            self.win,
            self.seq,
            self.wstart0,
            self.utc0,
            self.present,
            self.max_chan,
            self.row_total,
            chain,
            inst,
        ) = hdr[:18]
        # The comb's column labels ride on the frame and are never configured here: a
        # configured copy can drift out of step with the node it describes, and a delay fit
        # over mislabelled frequencies is confidently wrong rather than absent.
        self.chan_ids = list(hdr[18 : 18 + _MAX_CHAN])[: self.n_chan]
        self.chain = chain.split(b"\0", 1)[0].decode("ascii", "replace")
        self.inst = inst.split(b"\0", 1)[0].decode("ascii", "replace")
        self._buf = buf
        self._idx = None  # prn -> row index, built on first use
        self.rx = rx

    def has_record(self, r):
        """Was record slot r filled? A missing record is a hole at a known index, never a shift."""
        return bool(self.present & (1 << r))

    def hop(self, r):
        """Absolute F-engine hop index of record slot r: the same key /get_records uses."""
        return (self.wstart0 + r * self.hops_per_record * self.fft_len) // self.fft_len

    def _index(self):
        if self._idx is None:
            idx = {}
            # Each row's PRN (REC_PRN) is read from the first filled record slot. GnssTelemPack
            # fixes the row map for the whole window, writes rows only into the slots whose
            # record arrived and leaves the rest zero, so every filled slot carries the same map
            # and slot 0 may carry none. The map is read from the data rather than from a
            # configured PRN list the broker would have to keep in step.
            r0 = next((r for r in range(self.n_rec) if self.has_record(r)), None)
            if r0 is not None:
                for p in range(self.n_prn):
                    off = _HDR_BYTES + ((r0 * self.n_prn) + p) * self.row_total * 4
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
        off = _HDR_BYTES + ((r * self.n_prn) + p) * self.row_total * 4
        a = array.array("f")
        a.frombytes(self._buf[off : off + _ROW_FLOATS * 4])  # the record header only
        return a

    def comb_epl(self, r, prn):
        """The comb with all three taps: [(freq_id, E, P, L, (eE, eP, eL)), ...], or [].

        The input of the comb DLL (combdll.py). Each tap is normalised by its own replica
        energy, and all three were element-combined and NCO-rotated identically in the
        assembler: a discriminator built from taps combined even slightly differently measures
        the difference between the combines rather than the code offset.

        Summing P*eP over the columns does not exactly reproduce the header's prompt.
        ElemCal::combine is linear, so the channel sum and the element combine commute; but
        while the element cal adapts, the assembler calls ec.update() between the header's
        combine (the previous record's weights, causal) and the comb's (weights already updated
        with this record). So the comb is combined with weights one record newer, which have
        seen the record they are applied to; the /get_spectrum export shares this. Every column
        of one instance shares the same weights, so a delay fit across channels (the phase
        ramp) is unaffected. The residual is a known offset, not a transport error: do not use
        it as a check of the header.
        """
        p = self._index().get(int(prn))
        if p is None or not self.has_record(r) or not self.n_chan:
            return []
        base = (
            _HDR_BYTES + ((r * self.n_prn) + p) * self.row_total * 4 + _ROW_FLOATS * 4
        )
        a = array.array("f")
        a.frombytes(self._buf[base : base + self.n_chan * _CHAN_FLOATS * 4])
        out = []
        for ch in range(self.n_chan):
            b = ch * _CHAN_FLOATS
            eP = a[b + CHAN_ENERGY]
            if eP <= 0.0:
                continue
            eE = a[b + CHAN_E_ENERGY]
            eL = a[b + CHAN_L_ENERGY]
            out.append(
                (
                    self.chan_ids[ch],
                    complex(a[b + CHAN_E_RE] / eE, a[b + CHAN_E_IM] / eE),
                    complex(a[b + CHAN_RE] / eP, a[b + CHAN_IM] / eP),
                    complex(a[b + CHAN_L_RE] / eL, a[b + CHAN_L_IM] / eL),
                    (eE, eP, eL),
                )
            )
        return out


class TelemClient(object):
    """Socket thread + store thread + a bounded per-chain window ring.

    Reconnects forever. A gather that is down never blocks the broker: the window accessors
    return empty, fleet_coherent and the comb DLL fall back to their REST polls, and what only
    the stream feeds (the fleet ADRs, the telemetry C/N0 estimators) has no input until it
    returns.

    The socket is drained by a thread that does nothing else. The gather delivers a frame
    whole or drops the client (GnssTelemGather closes a client that cannot take a frame within
    its fixed 200 ms send_deadline_ms), and one connection carries every chain, so a stall of
    the reading thread breaks every fleet-ADR arc on every chain at once. The kernel's socket
    buffer covers only a fraction of a second at the stream's rate, and the ring lock is
    contended by every chain's control pass. So the socket thread reads bytes into a queue,
    and a second thread parses them and takes the ring lock; nothing the consumers do can stop
    the socket draining.
    """

    def __init__(
        self,
        host="127.0.0.1",
        port=11061,
        depth=64,
        retry_s=5.0,
        chains=None,
        read_timeout_s=30.0,
        epoch_margin=64,
        epoch_strikes=8,
        pending_max_bytes=128 << 20,
        fast_retry_after_s=10.0,
        rcvbuf_bytes=32 << 20,
    ):
        self.host = host
        self.port = port
        self.depth = int(depth)
        # An F-engine re-base moves the window counter far backwards. The margin and strike
        # count are gnss::FleetDll's (gnssFleetDll.hpp, run by GnssFleetTrim on the gather), but
        # the counting is not, so the broker re-anchors a few windows after the gather; see
        # _store_frame.
        self.epoch_margin = int(epoch_margin)
        self.epoch_strikes = int(epoch_strikes)
        self.epoch_resets = 0
        self.far_behind = (
            0  # frames more than epoch_margin behind the newest window (counted, kept)
        )
        self._backwards = (
            {}
        )  # chain -> consecutive such frames; epoch_strikes of them re-anchor
        # Optional chain filter, applied before a frame is stored. The stream carries every
        # chain on one connection, so without it a long collection for one chain also holds
        # every other chain's frames. None = keep everything (the broker's case).
        self.chains_filter = set(chains) if chains else None
        self.retry_s = float(retry_s)
        # Socket read timeout. Before a frame starts, a timeout is silence and the loop only
        # re-checks the stop flag (see _read_loop); part-way through a frame it is fatal. A
        # parameter so tests can exercise the quiet path in a fraction of a second.
        self.read_timeout_s = float(read_timeout_s)
        self._lock = threading.Lock()
        # chain -> OrderedDict{win: {inst: TelemFrame}}, oldest first, capped at `depth`
        self._store = {}
        self._thread = None
        self._store_thread = None
        self._stop = threading.Event()
        # Frames read off the socket and not yet parsed into the ring: (bytes, rx_time). The
        # socket thread appends, the store thread pops; the condition's lock is held for one
        # deque operation and never across a parse or the ring lock. Bounded in bytes because
        # senders ship different frame sizes: past the bound the oldest frames are dropped and
        # counted, which costs the affected windows (the fold breaks those arcs on the gap) but
        # never the connection.
        self._pending = collections.deque()
        self._pending_bytes = 0
        self._pending_cv = threading.Condition(threading.Lock())
        self.pending_max_bytes = int(pending_max_bytes)
        self.pending_peak = 0  # most frames ever queued at once
        self.pending_dropped = (
            0  # frames discarded because the queue was over its bound
        )
        # A connection that lived at least fast_retry_after_s and then ended is reconnected at
        # once: a dropped client is a momentary stall on our side, and every second before the
        # reconnect is a second of records lost on every chain. The backoff (retry_s) is for a
        # gather that is down, i.e. a connection that failed or died young.
        self.fast_retry_after_s = float(fast_retry_after_s)
        self.fast_reconnects = 0
        # Receive buffer asked for on each new socket; see _connect. 0 leaves the kernel's
        # autotuning alone. `rcvbuf_actual` is what the kernel granted, which is the number
        # worth reading: the request is clamped to net.core.rmem_max and then doubled.
        self.rcvbuf_bytes = int(rcvbuf_bytes)
        self.rcvbuf_actual = None
        self.connected = False
        self.frames = 0
        self.bad = 0
        self.connects = 0
        self.last_rx = 0.0
        self._seen_seq = {}  # (chain, inst) -> last seq
        self.gaps = 0

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self):
        if self._thread is not None:
            return self
        self._store_thread = threading.Thread(
            target=self._store_loop, name="telem-store", daemon=True
        )
        self._store_thread.start()
        self._thread = threading.Thread(target=self._run, name="telem", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        with self._pending_cv:
            self._pending_cv.notify_all()

    def _connect(self):
        """Open the socket with its receive buffer sized before connect().

        SO_RCVBUF has to precede connect(): the window scale is negotiated in the handshake
        from the buffer size at that moment, so a buffer enlarged afterwards cannot be
        advertised in full. The size matters most just after a connect: the kernel starts a
        fresh connection near 128 kB and autotunes upwards over tens of seconds, while this
        stream carries every chain at tens of MB/s and the gather drops a client that cannot
        take a frame within its 200 ms send deadline. A dropped connection that lived is
        reconnected at once (see fast_retry_after_s), so leaving the size to autotuning would
        re-enter that ramp on every drop, exactly when the link is already in trouble.
        The kernel clamps the request to net.core.rmem_max, so a host without the sysctl gets
        its maximum rather than an error.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if self.rcvbuf_bytes > 0:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf_bytes)
            sock.settimeout(10.0)
            sock.connect((self.host, self.port))
            self.rcvbuf_actual = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        except Exception:
            sock.close()
            raise
        return sock

    def _run(self):
        while not self._stop.is_set():
            sock = None
            t_up = None
            fast = False
            try:
                sock = self._connect()
                sock.settimeout(self.read_timeout_s)
                # Keepalive: a read timeout is tolerated as silence (see _read_loop), so the
                # kernel's probe is what notices a gather whose host vanished without closing
                # the socket.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                for opt, val in (
                    ("TCP_KEEPIDLE", 30),
                    ("TCP_KEEPINTVL", 10),
                    ("TCP_KEEPCNT", 3),
                ):
                    if hasattr(socket, opt):
                        sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)
                self.connected = True
                self.connects += 1
                t_up = time.monotonic()
                _log(
                    "telem: connected to gather %s:%d (connection #%d, receive buffer "
                    "%.1f MB%s)"
                    % (
                        self.host,
                        self.port,
                        self.connects,
                        (self.rcvbuf_actual or 0) / 1048576.0,
                        ""
                        if (self.rcvbuf_actual or 0) >= self.rcvbuf_bytes
                        else " -- CLAMPED by net.core.rmem_max, which is the ceiling on how "
                        "long a reader stall the link survives",
                    )
                )
                self._read_loop(sock)
            except Exception as e:
                lived = None if t_up is None else time.monotonic() - t_up
                fast = lived is not None and lived >= self.fast_retry_after_s
                if fast:
                    self.fast_reconnects += 1
                    # Not rate-limited: each one breaks every chain's arcs, and every
                    # timestamp matters.
                    _log(
                        "telem: gather %s:%d closed a connection that lived %.0f s (%s) -- "
                        "reconnecting now; the frames in between are lost on every chain and "
                        "the REST path is unaffected" % (self.host, self.port, lived, e)
                    )
                else:
                    _log_rl(
                        "telem-conn",
                        "telem: gather %s:%d unavailable (%s) -- retry in "
                        "%.0f s; the REST path is unaffected"
                        % (self.host, self.port, e, self.retry_s),
                    )
            finally:
                self.connected = False
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
            if not self._stop.is_set() and not fast:
                self._stop.wait(self.retry_s)

    def _recv_exactly(self, sock, n, quiet_ok=False):
        """Read exactly n bytes. `quiet_ok` tolerates a read timeout before the first byte.

        Only before the first byte: a timeout part-way through a frame leaves the stream
        desynchronised, and continuing would resume parsing at an offset, giving plausible
        numbers on the wrong rows. So the caller may treat "nothing arrived" as silence, and
        everything else stays fatal.
        """
        chunks = []
        got = 0
        while got < n:
            try:
                b = sock.recv(n - got)
            except socket.timeout:
                if quiet_ok and got == 0:
                    return None
                raise
            if not b:
                raise IOError("gather closed the connection")
            chunks.append(b)
            got += len(b)
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    def _read_loop(self, sock):
        """Drain the socket into the pending queue. Bytes in, bytes out, nothing else here.

        The only work between two recv() calls is one length check and one deque append under
        a lock that the store thread holds for one pop. A parse, a log line or the ring lock
        do not belong on this thread: see the class note on why a stall here is fleet-wide.
        """
        while not self._stop.is_set():
            # A quiet gather is not a broken one: with the fleet down it has nothing to
            # forward. A timeout before a frame starts is silence, so keep waiting rather than
            # churn reconnects for the length of an outage; keepalive (see _run) detects a link
            # that is gone.
            raw = self._recv_exactly(sock, 4, quiet_ok=True)
            if raw is None:
                continue
            (length,) = struct.unpack("<I", raw)
            # A frame is delivered whole or the gather closes the connection (it never
            # half-writes), so a length outside the plausible range means the stream is not
            # what we think it is -- reconnect rather than parse garbage.
            if length < _HDR_BYTES or length > (1 << 24):
                raise IOError("implausible frame length %d" % length)
            buf = self._recv_exactly(sock, length)
            self._enqueue(buf, time.time())

    def _enqueue(self, buf, rx):
        with self._pending_cv:
            self._pending.append((buf, rx))
            self._pending_bytes += len(buf)
            n = len(self._pending)
            if n > self.pending_peak:
                self.pending_peak = n
            # Over the bound: shed the oldest, keep the connection. Never shed the frame just
            # appended: a bound smaller than one frame must still let frames through.
            while (
                self._pending_bytes > self.pending_max_bytes and len(self._pending) > 1
            ):
                old, _ = self._pending.popleft()
                self._pending_bytes -= len(old)
                self.pending_dropped += 1
            self._pending_cv.notify()

    def _store_loop(self):
        while not self._stop.is_set():
            with self._pending_cv:
                while not self._pending and not self._stop.is_set():
                    self._pending_cv.wait(0.5)
                if not self._pending:
                    continue
                buf, rx = self._pending.popleft()
                self._pending_bytes -= len(buf)
            try:
                self._parse_and_store(buf, rx)
            except Exception as e:  # one bad frame must not take the store thread down
                self.bad += 1
                _log_rl(
                    "telem-parse", "telem: dropping a frame the parser refused (%s)" % e
                )

    def _parse_and_store(self, buf, rx):
        hdr = _HDR.unpack_from(buf, 0)
        if hdr[0] != _MAGIC or hdr[1] != _VERSION or hdr[4] != _ROW_FLOATS:
            self.bad += 1
            _log_rl(
                "telem-bad",
                "telem: rejecting a frame (magic %#x v%d n_row %d, want "
                "%#x v%d %d) -- a tracker and this broker are on different builds"
                % (hdr[0], hdr[1], hdr[4], _MAGIC, _VERSION, _ROW_FLOATS),
            )
            return
        # The shape is the sender's, checked against the length prefix. Senders ship
        # different widths (their own comb columns) and different row counts (their chain's),
        # so the stride cannot be assumed, and a stride that disagrees with the bytes on the
        # wire reads plausible numbers off the wrong rows. n_rec, n_prn, n_chan, max_chan,
        # row_total are hdr[2], [3], [5], [14], [15]. The bounds are gnss::telem_shape_ok's: a
        # header with no records or no rows matches a bare 112 bytes and would open a window.
        length = len(buf)
        if (
            hdr[2] == 0
            or hdr[2] > _MAX_REC
            or hdr[3] == 0
            or hdr[14] > _MAX_CHAN
            or hdr[5] > hdr[14]
            or hdr[15] != _ROW_FLOATS + hdr[14] * _CHAN_FLOATS
            or _HDR_BYTES + hdr[2] * hdr[3] * hdr[15] * 4 != length
        ):
            self.bad += 1
            _log_rl(
                "telem-shape",
                "telem: rejecting a frame whose header shape does not "
                "match its %d bytes (n_rec %d n_prn %d n_chan %d max_chan %d row_total "
                "%d)" % (length, hdr[2], hdr[3], hdr[5], hdr[14], hdr[15]),
            )
            return
        self._store_frame(TelemFrame(hdr, buf, rx))

    def _store_frame(self, f):
        if self.chains_filter is not None and f.chain not in self.chains_filter:
            return
        with self._lock:
            self.frames += 1
            self.last_rx = f.rx
            key = (f.chain, f.inst)
            prev = self._seen_seq.get(key)
            # Only the sender's own counter can say a frame was lost: a rate that looks right
            # can still be missing every fourth frame.
            if prev is not None and f.seq > prev + 1:
                self.gaps += f.seq - prev - 1
            self._seen_seq[key] = f.seq
            ring = self._store.get(f.chain)
            if ring is None:
                ring = self._store[f.chain] = collections.OrderedDict()
            if f.win not in ring:
                # Eviction must drop the oldest window, not the least recently written one: a
                # laggard instance opening a window its peers finished long ago would otherwise
                # push it to the front of the ring and evict a newer window instead. Senders
                # are independent processes, so out-of-order window opens are normal, which is
                # why the ring is re-sorted rather than assumed monotone.
                newest = next(reversed(ring)) if ring else None
                # An epoch reset is a large, persistent backwards jump. A laggard is a few
                # windows behind while its peers are not; a frame0 move lands the whole stream
                # far below everything held and keeps it there. Without the reset, post-re-base
                # windows are sorted to the front of a full ring and evicted first, so every
                # frame after an F-engine restart is discarded while frames and gaps look
                # healthy. One corrupt header cannot trigger it: epoch_strikes consecutive
                # window-opening frames must be far behind, and one that is not clears the run.
                # Until the run completes the frame is stored like any other. The margin and
                # strike count are the gather's (gnssFleetDll.hpp) but the counting is not: the
                # gather counts every far-behind frame and clears the run only on an in-order
                # one, so a re-base re-anchors the gather after epoch_strikes frames and the
                # broker after epoch_strikes windows.
                if newest is not None and newest - f.win > self.epoch_margin:
                    self.far_behind += 1
                    self._backwards[f.chain] = self._backwards.get(f.chain, 0) + 1
                    if self._backwards[f.chain] >= self.epoch_strikes:
                        self._backwards[f.chain] = 0
                        self.epoch_resets += 1
                        _log(
                            "telem: chain %s window counter went back %d -> %d and stayed "
                            "there: epoch reset #%d, ring cleared (an F-engine re-base)"
                            % (f.chain, newest, f.win, self.epoch_resets)
                        )
                        ring.clear()
                        newest = None
                else:
                    self._backwards[f.chain] = 0
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

        The lag is not padding. Senders are independent processes on several machines, so the
        newest window is always still arriving; taking it would combine a partial window with
        complete neighbours, which reads as a fleet that keeps changing size.
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
        A = (P_re + i P_im)/P_energy and energy = P_energy: the same two numbers the combiner
        publishes (it forms ar = gr/energy and exports that), so the estimator downstream is
        unchanged and a difference between the two feeds measures the transport.

        Records with zero energy are dropped exactly as /get_records drops them (a PRN that did
        not run is silence, not a row of zeros).
        """
        wins = self.windows(chain, lag=lag)
        if not wins:
            return {}, 0
        wins = wins[-int(n_win) :]
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
                            complex(row[REC_P_RE] / e, row[REC_P_IM] / e),
                            e,
                        )
        return got, fleet_now

    def stats(self, stale_after_s=5.0):
        """Transport health, including the cross-instance alignment check.

        `spread` is max(win) - min(win) over the live instances of a chain. Zero or one is the
        transport working; anything larger is misalignment between senders.

        Live only: a stopped instance keeps its last window forever, so one dead sender would
        drive the raw spread to the number of windows since it died. Instances not heard from
        within stale_after_s are listed by name in `stale` instead.
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
                live = {
                    i: w
                    for i, w in last.items()
                    if stale_after_s <= 0 or (now - seen.get(i, 0.0)) <= stale_after_s
                }
                stale = sorted(set(last) - set(live))
                row = {
                    "instances": len(last),
                    "live": len(live),
                    "stale": stale,
                    "windows_held": len(ring),
                }
                if live:
                    row.update(
                        {
                            "win_min": min(live.values()),
                            "win_max": max(live.values()),
                            "spread": max(live.values()) - min(live.values()),
                        }
                    )
                per_chain[chain] = row
            return {
                "connected": self.connected,
                "frames": self.frames,
                "gaps": self.gaps,
                "bad": self.bad,
                "far_behind": self.far_behind,
                "epoch_resets": self.epoch_resets,
                "connects": self.connects,
                "fast_reconnects": self.fast_reconnects,
                "rcvbuf": self.rcvbuf_actual,
                "pending": len(self._pending),
                "pending_peak": self.pending_peak,
                "pending_dropped": self.pending_dropped,
                "age_s": (now - self.last_rx) if self.last_rx else None,
                "chains": per_chain,
            }


_shared = {}
_shared_lock = threading.Lock()


def shared_client(host, port, depth=256):
    """One reader thread per process, shared by every chain.

    broker_multi runs every chain as a thread of one process, and the gather stream carries
    every chain on one connection, so a client per chain would open one connection per chain
    and parse the whole stream once per chain. The store is keyed by chain, so sharing changes
    no accessor.
    """
    # Depth is in windows, and a window is one frame (4 records, ~42 ms at the configured
    # packing), so 256 windows hold ~10.7 s. The fleet-ADR fold reads every window since its
    # last pass, so the ring must outlast the longest control pass: a window evicted before it
    # is folded breaks every fleet-ADR arc on the chain (a FADR BREAK line).
    key = (host, int(port))
    with _shared_lock:
        c = _shared.get(key)
        if c is None:
            c = _shared[key] = TelemClient(
                host=host, port=int(port), depth=depth
            ).start()
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
