#!/usr/bin/env python3
"""livebeam_server.py - bridge from kotekan's networkPowerStream to a browser viewer.

This server has four responsibilities, all wired up by :func:`main`:

  1. **TCP listener (default 23401)** for kotekan's outgoing power stream. The
     kotekan side connects once at startup and sends a 48-byte handshake header
     (see :class:`KotekanPowerStream`), then streams power-spectrum subframes
     -- ``frame_nvis`` of them per integration. For an autocorr pipeline this
     is one subframe per integration; for a cross-corr pipeline it is four
     (AA, BB, Re{AB*}, Im{AB*}). All subframes for one integration are
     accumulated into :attr:`KotekanPowerStream.databuf`.

  2. **WebSocket (default 8539)** for browser clients. On open each client
     receives a JSON ``viewer_config`` describing the pipeline mode and the
     UI modules to enable, then a binary FREQLIST message, then a binary
     TIMESTEP message per integration. See :data:`MSG_FREQLIST` /
     :data:`MSG_TIMESTEP` for the frame layout.

  3. **HTTP (default 8080)** serving the static viewer files from this
     directory (index.html, JS, CSS).

  4. **GET /mode** (under the HTTP server) returning a small JSON blob
     identifying the kotekan pipeline mode -- so the static front-end can
     pick the right JS to load before opening the WebSocket.

The kotekan TCP receive runs as a Twisted ``callLater`` loop. That's good
enough for one-producer / many-WS-clients but means
:meth:`KotekanPowerStream.start` blocks on ``accept`` at startup -- launch
this server before kotekan starts streaming.
"""

import argparse
import json
import logging
import os
import select
import signal
import socket
import struct
import sys
import time

import numpy as np
from autobahn.twisted.websocket import WebSocketServerFactory, WebSocketServerProtocol
from twisted.internet import reactor, threads
from twisted.internet.interfaces import IPushProducer
from twisted.python import log
from twisted.web import resource, server, static
from zope.interface import implementer


# Binary WebSocket message types (first byte of each binary push).
MSG_FREQLIST = 1  # bytes[1:] = nfreq * float32 frequency-bin centres (MHz)
MSG_TIMESTEP = 2  # bytes[1:9] = float64 sample-time (UTC sec)
#                  # bytes[9:]  = (nvis * nfreq) float32 power-spectrum bins

# Viewer protocol version. Bump on any breaking change to the WebSocket
# protocol or viewer_config shape; the JS client compares against its own
# compiled-in version and warns the user on mismatch (see app/socket.js).
VIEWER_PROTOCOL_VERSION = 1

# Default target browser-facing integration time (ms). kotekan often emits
# far faster than a viewer needs (e.g. 6.55 ms); from this target and the
# handshake's kotekan period we derive an integer N = round(target /
# kotekan_period) and average exactly N kotekan integrations per emitted
# frame. The *effective* period (N x kotekan_period, exact in the
# instrument clock) is reported to the client as ``frame_period_s``.
# Overridable with --viewer-integration-ms.
DEFAULT_VIEWER_INTEGRATION_MS = 50

# Cap on how many kotekan integrations a single _tick will drain from the
# socket. Draining to the newest buffered frame is what keeps sample_time
# pinned to ~now (the actual fix for the "fresh browser only ever sees
# stale frames" bug); the cap just bounds how long one tick can block the
# reactor while catching up from a large backlog. The 1 ms reschedule
# means even a big backlog clears in a fraction of a second.
MAX_DRAIN_PER_TICK = 256

# WebSocket keepalive: send a ping every N seconds; if no pong comes back
# within M seconds, drop the connection. The dropped connection triggers
# the JS client's auto-reconnect path.
WS_PING_INTERVAL_S = 20
WS_PING_TIMEOUT_S = 10

# Backpressure tuning. This is a *real-time* viewer, so we want frames to
# drop quickly when the network can't keep up rather than queuing and
# falling seconds behind. Both the kernel send buffer (SO_SNDBUF) and
# Twisted's per-transport ``bufferSize`` (the pauseProducing threshold)
# are sized to a small number of *frames*, computed per-connection from
# the actual frame size -- a fixed byte cap was the bug: 16 KB is ~4
# autocorr frames but ~1 crosscorr frame (4 streams x 1024 x f32 =
# 16 KB), so every single crosscorr frame filled the buffer and tripped
# pause/resume even though throughput was trivial.
#
# A client whose buffer genuinely can't keep up sees frames dropped
# server-side; the JS gap detector then paints the missed span grey.
WS_BUFFER_FRAMES = 4  # frames of slack before backpressure
WS_BUFFER_MIN_BYTES = 16 * 1024  # floor (also the autocorr value)

# Defense-in-depth: if a client stays paused (TCP buffer full, slow / hung
# downstream) for this many seconds, abort the connection. AutoPing already
# closes truly dead TCPs in ~30s, but a stuck-paused state can occur where
# the kernel still ACKs while the browser stops draining, and we'd rather
# kick the client and let it auto-reconnect than sit silently.
WS_STUCK_PAUSED_S = 30


# Minimum interval between /gps_sky recomputes of satellite az/el. MEO satellites move
# ~0.01 deg/s, so a few seconds is plenty. The recompute (the broker's /get_sky, plus TLE for
# GLONASS or as the fallback) runs in a worker thread and render_GET serves only the cached
# result, so the Twisted reactor never blocks on it.
SKY_REFRESH_S = 5.0

# Mode is decided by the number of visibility streams in the kotekan handshake.
MODE_BY_NVIS = {1: "autocorr", 4: "crosscorr"}
VIS_LABELS_BY_MODE = {
    "autocorr": ["I"],
    "crosscorr": ["AA", "BB", "Re{AB*}", "Im{AB*}"],
}


log_ = logging.getLogger("livebeam")


class KotekanPowerStream:
    """One kotekan-side TCP connection.

    Owns the listening socket, the accepted connection, and the latest
    accumulated frame (:attr:`databuf`, shape ``(nvis, nfreq)`` float32).

    Wire format
    -----------
    On accept, kotekan sends a 48-byte handshake header followed by
    ``frame_nfreq * 2 * 4`` bytes of frequency-bin ``(lo, hi)`` pairs (float32,
    Hz) and ``frame_nvis`` element-id bytes (int8).

    Then a stream of subframes; each subframe is a 12-byte sub-header
    (``frame_idx``, ``elem_idx``, ``samples_summed``: 3 uint32) followed by
    ``frame_nfreq`` float32 power values. ``frame_nvis`` consecutive subframes
    make one integration.
    """

    HEADER_FMT = "=iiiidiiiId"
    HEADER_LEN = struct.calcsize(HEADER_FMT)
    SUBFRAME_HDR_FMT = "III"
    SUBFRAME_HDR_LEN = struct.calcsize(SUBFRAME_HDR_FMT)

    def __init__(
        self,
        host="0.0.0.0",
        port=23401,
        on_frame=None,
        emit_interval_s=DEFAULT_VIEWER_INTEGRATION_MS / 1000.0,
    ):
        self.host = host
        self.port = port
        self.on_frame = on_frame  # callable, fired once per emitted (averaged) frame
        self.connected = False
        # Browser-facing integration window; integrations drained inside one
        # window are averaged into a single emitted frame.
        self.emit_interval_s = emit_interval_s
        # Populated by start() once kotekan's handshake is parsed.
        self.frame_nvis = 0
        self.frame_nfreq = 0

    def start(self):
        """Bind, ``accept`` (blocking), parse the handshake, and prime the loop."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(1)
        log_.info("Waiting for kotekan on %s:%d", self.host, self.port)
        self._listen_sock = sock
        self.connection, peer = sock.accept()
        log_.info("Connected to kotekan: %s", peer)

        self.packed_header = self._recv_exact(self.HEADER_LEN, self.connection)
        (
            self.frame_length,  # packet_length: bytes of float32 spectrum per subframe
            self.subframe_hdr_len,  # bytes of subframe header (should == SUBFRAME_HDR_LEN)
            self.frame_samples,  # samples_per_packet (== nfreq)
            self.frame_dtype,  # sample_type tag
            self.frame_raw_cad,  # raw_cadence (seconds per sample)
            self.frame_nfreq,  # number of freq bins
            self.frame_nvis,  # number of visibilities / elements per integration
            self.frame_int_len,  # samples_summed per integration
            self.frame_idx0,  # handshake_idx (zero offset for frame_idx)
            self.frame_utc0,  # handshake_utc (zero offset for sample_time)
        ) = struct.unpack(self.HEADER_FMT, self.packed_header)

        info_len = self.frame_nfreq * 4 * 2 + self.frame_nvis * 1
        info = self._recv_exact(info_len, self.connection)
        self.frame_freqs = np.frombuffer(
            info[: self.frame_nfreq * 4 * 2], dtype=np.float32
        ).reshape(-1, 2)
        self.frame_elems = np.frombuffer(
            info[self.frame_nfreq * 4 * 2 :], dtype=np.int8
        )

        self.databuf = np.zeros((self.frame_nvis, self.frame_nfreq), dtype=np.float32)
        self.frame_idx = self.frame_idx0
        self.samples_summed = 0
        self.connected = True

        # Fixed-N integration accumulator. We average exactly ``integrate_n``
        # kotekan integrations into one emitted frame -- a data-domain decimation,
        # *not* a wall-clock window -- so the emitted-frame spacing is
        # exactly ``N x kotekan_period`` in the instrument's own clock. The
        # instrument clock (frame_idx / raw_cad / int_len / utc0) is the
        # single source of truth; the host wall clock is never consulted.
        #
        # ``_emit_idx`` is the kotekan frame index of the newest integration
        # in the most recently finalized block; sample_time() reconstructs
        # the timestamp from it. Caught up, consecutive emits' indices
        # differ by exactly N -> spacing == reported frame_period_s. Behind,
        # the drain loop finalizes several blocks per tick but only keeps
        # the last -> the index jumps by k*N and the client's gap detector
        # correctly paints the (k-1) dropped blocks grey.
        self.kotekan_period_s = float(self.frame_raw_cad) * float(self.frame_int_len)
        self.integrate_n = max(1, round(self.emit_interval_s / self.kotekan_period_s))
        self._acc = np.zeros((self.frame_nvis, self.frame_nfreq), dtype=np.float64)
        self._acc_n = 0
        self.outbuf = np.zeros((self.frame_nvis, self.frame_nfreq), dtype=np.float32)
        self._emit_idx = self.frame_idx
        self._block_ready = False
        log_.info(
            "Viewer integration: averaging N=%d kotekan integrations "
            "(kotekan period %.3f ms -> effective %.1f ms / frame)",
            self.integrate_n,
            self.kotekan_period_s * 1e3,
            self.integrate_n * self.kotekan_period_s * 1e3,
        )

        log_.info(
            "Handshake: mode=%s nvis=%d nfreq=%d",
            self.mode,
            self.frame_nvis,
            self.frame_nfreq,
        )

        self.connection.settimeout(0.1)
        reactor.callLater(0.001, self._tick)

    @property
    def mode(self):
        """``'autocorr'`` for nvis=1, ``'crosscorr'`` for nvis=4, else ``'unknown'``."""
        return MODE_BY_NVIS.get(self.frame_nvis, "unknown")

    @property
    def vis_labels(self):
        """Per-vis labels for the configured mode."""
        return VIS_LABELS_BY_MODE.get(
            self.mode, [f"vis{i}" for i in range(self.frame_nvis)]
        )

    @staticmethod
    def _recv_exact(n, conn):
        """Receive exactly ``n`` bytes from ``conn`` or raise."""
        chunks = []
        got = 0
        while got < n:
            chunk = conn.recv(min(n - got, 2048))
            if not chunk:
                raise RuntimeError("kotekan TCP connection closed")
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def _read_one_integration(self):
        """Read ``frame_nvis`` subframes into :attr:`databuf`. May raise
        socket.timeout (no data right now) or on a closed connection."""
        for i in range(self.frame_nvis):
            d = self._recv_exact(
                self.subframe_hdr_len + self.frame_length, self.connection
            )
            self.frame_idx, elem_idx, self.samples_summed = struct.unpack(
                self.SUBFRAME_HDR_FMT, d[: self.subframe_hdr_len]
            )
            if elem_idx != i:
                log_.warning(
                    "Out-of-order subframe: expected elem %d, got %d", i, elem_idx
                )
                self.databuf[i, :] = 0
                break
            self.databuf[elem_idx, :] = np.frombuffer(
                d[self.subframe_hdr_len :], dtype=np.float32
            )

    def _socket_has_data(self):
        """True if at least one more byte is buffered on the kotekan socket
        (so another integration is waiting and we shouldn't fall behind)."""
        try:
            r, _, _ = select.select([self.connection], [], [], 0)
            return bool(r)
        except (OSError, ValueError):
            return False

    def _tick(self):
        """One iteration of the receive loop.

        Reads a kotekan integration, then *drains* any further integrations
        already buffered on the socket (up to MAX_DRAIN_PER_TICK), summing
        them into the averaging accumulator. This is the core fix for the
        stale-data bug: by always catching up to the newest buffered frame,
        ``frame_idx`` (hence sample_time) stays pinned to ~now even if the
        reactor briefly fell behind kotekan's emit rate.

        Once at least ``emit_interval_s`` of wall-clock has passed and we
        have something accumulated, the mean is emitted as a single frame
        (timestamped at the newest integration) and on_frame fires.
        """
        try:
            try:
                self._read_one_integration()
            except socket.timeout:
                pass
            else:
                self._fold_integration()
                drained = 0
                while drained < MAX_DRAIN_PER_TICK and self._socket_has_data():
                    self._read_one_integration()
                    self._fold_integration()
                    drained += 1
        except Exception as e:
            log_.warning("kotekan recv: %s", e)
            self.close()
            return

        # At most one on_frame per tick, carrying the most recent finalized
        # block. If several blocks finalized this tick (we were behind), the
        # older ones were intentionally dropped; _emit_idx jumped by k*N so
        # the client renders the skipped span as grey.
        if self._block_ready:
            self._block_ready = False
            if self.on_frame is not None:
                self.on_frame()

        if self.connected:
            reactor.callLater(0.001, self._tick)

    def _fold_integration(self):
        """Add the just-read integration to the current block; finalize the
        block (-> outbuf, ready to emit) once N have accumulated."""
        self._acc += self.databuf
        self._acc_n += 1
        if self._acc_n >= self.integrate_n:
            np.divide(self._acc, self._acc_n, out=self._acc)
            self.outbuf[:] = self._acc.astype(np.float32)
            self._emit_idx = self.frame_idx  # newest integration in block
            self._acc.fill(0.0)
            self._acc_n = 0
            self._block_ready = True

    def sample_time(self):
        """UTC seconds of the newest integration in the last emitted block,
        reconstructed purely from the instrument clock (handshake utc0 +
        frame index x raw cadence x samples-per-integration). No host wall
        clock involved -- this is the single source of truth, and the drain
        loop keeps _emit_idx ~current."""
        return self.frame_utc0 + self.frame_raw_cad * self.frame_int_len * (
            self._emit_idx - self.frame_idx0
        )

    def close(self):
        if not self.connected:
            return
        self.connected = False
        try:
            self.connection.close()
        except Exception:
            pass
        try:
            self._listen_sock.close()
        except Exception:
            pass
        log_.info("Closed kotekan connection")


def build_viewer_config(kotekan, args):
    """Assemble the JSON the server ships to each browser client at WS-open.

    The browser uses this to pick which UI modules to wire up, what labels to
    show for each visibility stream, where to reach kotekan's REST endpoints,
    and which optional integrations (CCERA pointing, galaxy view) to enable.

    ``kotekan`` is None in GPS-only mode (--no-power-stream): there is no power
    waterfall, so mode is "gps" and the client builds only the GNSS panels.
    """
    if kotekan is None:
        # GPS-only: no power stream, so no waterfall or spectrum fields.
        mode = "gps"
        airspy_stages, frame_period_s = [], 0.0
        nfreq, nvis, vis_labels = 0, 0, []
        ui = {}
    else:
        mode = kotekan.mode

        # Pipeline-shaped defaults; the user can override the lists / values via CLI.
        default_airspy_stages = {
            "autocorr": ["airspy_input"],
            "crosscorr": ["airspy_inputA", "airspy_inputB"],
        }
        default_color_range = {"autocorr": [-20, 20], "crosscorr": [-30, 30]}
        airspy_stages = args.airspy_stages or default_airspy_stages.get(mode, [])

        # The exact data-clock spacing between emitted frames: N kotekan
        # integrations per frame x kotekan's per-integration period. This is
        # what the client's gap detector / time axis must key off. Reporting
        # the *requested* interval instead (rather than the realized
        # N x period) is what caused the every-other-row NaN striping in
        # crosscorr -- there the kotekan period (~105 ms) exceeds the 50 ms
        # request, so N clamps to 1 and the true spacing is ~105 ms, not 50.
        frame_period_s = float(kotekan.integrate_n) * float(kotekan.kotekan_period_s)
        nfreq, nvis, vis_labels = (
            int(kotekan.frame_nfreq),
            int(kotekan.frame_nvis),
            kotekan.vis_labels,
        )
        ui = {
            "color_range": default_color_range.get(mode, [-20, 20]),
            "freq_range_mhz": [
                float(kotekan.frame_freqs[0, 0] / 1e6),
                float(kotekan.frame_freqs[-1, 1] / 1e6),
            ],
            # Dataset-specific knobs the panels read instead of hardcoding:
            # baseline-fit line mask, auto-cal [calibrate, observe] freqs, tuner span.
            "line_mask_mhz": [1419.9, 1420.9],  # HI line, 1420.4 +/- 0.5
            "autocal_freqs_mhz": [1416.0, 1421.0],
            "tuning_range_mhz": [24.0, 1800.0],  # airspy R2 span
        }

    return {
        "version": VIEWER_PROTOCOL_VERSION,
        "mode": mode,
        "nfreq": nfreq,
        "nvis": nvis,
        "vis_labels": vis_labels,
        "frame_period_s": frame_period_s,
        "kotekan": {
            "rest_port": args.kotekan_rest_port,
            "airspy_stages": airspy_stages,
            "lag_align_stage": args.lag_align_stage,
        },
        "ui": ui,
        "optional_modules": {
            "airspy_controls": bool(airspy_stages) and not args.no_airspy_controls,
            "ccera_pointing": args.ccera_pointing,
            "galaxy_view": bool(args.galaxy_view_url),
            "galaxy_view_url": args.galaxy_view_url,
        },
        # ``has_site``: whether --lat/--lon are set. Without a site /gps_sky has no
        # positions; the table still lists the locked satellites and says "no site".
        "gps": {"has_site": args.lat is not None and args.lon is not None,},
    }


@implementer(IPushProducer)
class LiveBeamWSProtocol(WebSocketServerProtocol):
    """Per-client WebSocket protocol; sends viewer_config + FREQLIST on open,
    TIMESTEP each frame.

    Implements :class:`IPushProducer` so Twisted's transport can apply
    backpressure: when the TCP send buffer fills (slow client),
    :meth:`pauseProducing` flips ``_paused`` to True and :meth:`send_power_frame`
    starts dropping frames silently. Once the buffer drains,
    :meth:`resumeProducing` clears the flag. The client-side gap detector
    then renders the dropped span as a grey strip in the waterfall.
    """

    # IPushProducer plumbing -----------------------------------------------

    def pauseProducing(self):
        if not self._paused:
            log_.warning("WS %s: transport buffer full, dropping frames", self.peer)
            # One-shot watchdog so we don't sit paused forever if the client
            # stops draining. Cancelled when we resume; replaced fresh on
            # each pause edge so it always reflects the *current* pause's
            # age.
            self._stuck_watchdog = reactor.callLater(
                WS_STUCK_PAUSED_S, self._on_stuck_paused
            )
        self._paused = True

    def resumeProducing(self):
        if self._paused:
            log_.info("WS %s: transport drained, resuming frame stream", self.peer)
            self._cancel_stuck_watchdog()
        self._paused = False

    def stopProducing(self):
        # Transport is going away; just stop sending.
        self._paused = True
        self._cancel_stuck_watchdog()

    def _cancel_stuck_watchdog(self):
        wd = getattr(self, "_stuck_watchdog", None)
        if wd is not None and wd.active():
            wd.cancel()
        self._stuck_watchdog = None

    def _on_stuck_paused(self):
        log_.warning(
            "WS %s: stuck paused for %ds; aborting", self.peer, WS_STUCK_PAUSED_S
        )
        self._stuck_watchdog = None
        try:
            self.dropConnection(abort=True)
        except Exception as e:
            log_.warning("WS %s: dropConnection failed: %s", self.peer, e)

    # WebSocket lifecycle --------------------------------------------------

    def onConnect(self, request):
        log_.info("WS connecting: %s", request.peer)

    def onOpen(self):
        log_.info("WS open: %s", self.peer)
        self.factory.register(self)
        self._paused = False
        self._stuck_watchdog = None

        kotekan = self.factory.kotekan
        # GPS-only mode: there are no power frames to push, so send only the
        # viewer_config and skip the producer/backpressure plumbing and the FREQLIST.
        if kotekan is None:
            self.sendMessage(
                json.dumps(
                    {"nfreq": 0, "viewer_config": self.factory.viewer_config}
                ).encode("utf-8"),
                isBinary=False,
            )
            return

        # Tell the TCP transport to push us pause/resume callbacks when its
        # write buffer crosses the high/low water marks. ``streaming=True``
        # selects the IPushProducer (sync) protocol.
        self.transport.registerProducer(self, True)

        # Size the kernel + Twisted write buffers to ~WS_BUFFER_FRAMES
        # actual frames. A frame is the MSG_TIMESTEP header (1 byte type +
        # 8 byte f64 time) plus nvis*nfreq f32 spectrum bins. Sizing by
        # frame count keeps backpressure semantics consistent across modes
        # (a fixed byte cap was ~1 crosscorr frame, tripping pause/resume
        # on every single frame).
        frame_bytes = 9 + kotekan.frame_nvis * kotekan.frame_nfreq * 4
        buf_bytes = max(WS_BUFFER_MIN_BYTES, WS_BUFFER_FRAMES * frame_bytes)
        try:
            self.transport.getHandle().setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, buf_bytes
            )
        except (AttributeError, OSError) as e:
            log_.warning("WS %s: could not set SO_SNDBUF: %s", self.peer, e)
        self.transport.bufferSize = buf_bytes
        self.sendfreq = kotekan.frame_nfreq  # no downsampling for now

        # JSON config message. Top-level ``nfreq`` is the legacy field the
        # current waterfall.js still reads; ``viewer_config`` is the new
        # server-driven module/mode descriptor that the next commit teaches
        # the JS side to consume.
        self.sendMessage(
            json.dumps(
                {"nfreq": self.sendfreq, "viewer_config": self.factory.viewer_config}
            ).encode("utf-8"),
            isBinary=False,
        )

        # Binary FREQLIST.
        freqs_mhz = np.mean(
            (kotekan.frame_freqs / 1e6).reshape(self.sendfreq, -1), axis=1
        ).astype(np.float32)
        self.sendMessage(
            np.int8(MSG_FREQLIST).tobytes() + freqs_mhz.tobytes(), isBinary=True
        )

    def send_power_frame(self):
        kotekan = self.factory.kotekan
        if not kotekan.connected:
            self.transport.loseConnection()
            return

        # Skip the frame entirely if the transport is back-pressured -- the
        # client's own time-gap detector will paint the missed span grey.
        if self._paused:
            return
        t = np.float64(kotekan.sample_time())
        spectrum = kotekan.outbuf.astype(np.float32, copy=False)
        self.sendMessage(
            np.int8(MSG_TIMESTEP).tobytes() + t.tobytes() + spectrum.tobytes(),
            isBinary=True,
        )

    def onMessage(self, payload, isBinary):
        # The viewer is push-only; clients aren't expected to send anything.
        log_.info("Ignoring unexpected %d-byte message from client", len(payload))

    def onClose(self, wasClean, code, reason):
        log_.info("WS closed: %s", reason)
        self._cancel_stuck_watchdog()
        # unregisterProducer is safe to call even if registerProducer never
        # ran (eg the WS handshake failed before onOpen).
        try:
            self.transport.unregisterProducer()
        except Exception:
            pass
        self.factory.unregister(self)


class LiveBeamWSFactory(WebSocketServerFactory):
    """Holds the kotekan source, the viewer_config, and the client list."""

    protocol = LiveBeamWSProtocol

    def __init__(self, kotekan, viewer_config, url=None):
        WebSocketServerFactory.__init__(self, url)
        self.kotekan = kotekan
        self.viewer_config = viewer_config
        self.clients = []
        # autobahn-side keepalive: send a ping every WS_PING_INTERVAL_S
        # seconds; drop the connection if no pong comes back inside
        # WS_PING_TIMEOUT_S. The drop triggers the JS client's auto-
        # reconnect (see app/socket.js) so transient network blips heal
        # themselves instead of leaving a frozen browser tab.
        self.setProtocolOptions(
            autoPingInterval=WS_PING_INTERVAL_S, autoPingTimeout=WS_PING_TIMEOUT_S,
        )

    def register(self, client):
        if client not in self.clients:
            self.clients.append(client)

    def unregister(self, client):
        if client in self.clients:
            self.clients.remove(client)

    def broadcast(self):
        """Push the most recent kotekan frame to every connected client.

        Each client's send is isolated -- a transport-level error on one
        client must not propagate up to ``_tick``, which catches all
        ``Exception``s and tears down the *kotekan* TCP connection on the
        assumption it was a read failure. That misdiagnosis would mean a
        single misbehaving browser silently stops the whole pipeline until
        the server is restarted; the iter-over-copy + try/except below
        cleans up the broken client instead.
        """
        for c in list(self.clients):
            try:
                c.send_power_frame()
            except Exception as e:
                log_.warning(
                    "WS %s: send_power_frame raised %s; dropping client",
                    getattr(c, "peer", "?"),
                    e,
                )
                try:
                    c.transport.loseConnection()
                except Exception:
                    pass

    def close_all(self):
        for c in self.clients:
            c.sendClose()


class NoCacheFile(static.File):
    """A :class:`static.File` that asks the browser not to cache anything.

    Useful while iterating on the viewer: a hard-reload in Firefox in
    particular is sometimes happy to serve a stale ``waterfall.js`` from
    disk cache. ``Cache-Control: no-store`` is the big hammer; we don't
    need finer-grained caching for the in-tree static viewer.
    """

    def render_GET(self, request):
        request.setHeader(b"Cache-Control", b"no-store")
        request.setHeader(b"Pragma", b"no-cache")
        return static.File.render_GET(self, request)

    # ``File`` aliases ``render_HEAD = render_GET`` at class-definition time,
    # which binds ``render_HEAD`` to ``File.render_GET`` and skips our
    # override. Re-alias here so HEAD requests also get the no-store headers.
    render_HEAD = render_GET

    def createSimilarFile(self, path):
        f = self.__class__(
            path,
            defaultType=self.defaultType,
            ignoredExts=self.ignoredExts,
            registry=self.registry,
        )
        f.processors = self.processors
        f.indexNames = self.indexNames[:]
        f.childNotFound = self.childNotFound
        return f


class ModeResource(resource.Resource):
    """``GET /mode`` -> ``{"mode": "autocorr"|"crosscorr"|"unknown"}``.

    Lets the static front-end pick a per-mode bundle before the WebSocket
    opens. Synchronously available once kotekan has finished its handshake.
    """

    isLeaf = True

    def __init__(self, kotekan):
        resource.Resource.__init__(self)
        self.kotekan = kotekan

    def render_GET(self, request):
        request.responseHeaders.setRawHeaders("Content-Type", [b"application/json"])
        mode = self.kotekan.mode if self.kotekan is not None else "gps"
        return json.dumps({"mode": mode}).encode("utf-8")


# The signal inventory, one row per broker chain (discover_broker_chains gives the row shape).
# main() fills it on a --unified start, before _gps_signal_capability() and the GLONASS sky
# filter read it.
UNIFIED_SIGNALS = []

# Signal-id prefix -> constellation letter; main() backfills each row's "sys" with it.
_SYS_TAG = {"GPS": "G", "GAL": "E", "BDS": "C", "GLO": "R", "SBAS": "S"}

# The three column groups, which the page shows as High/Mid/Low. The keys are mirrored in
# app/panels/gps_table.js's BAND_LABEL and BAND_ORDER, so rename them together.
BAND_ORDER = ["high", "mid", "low"]


def _group_of(carrier_hz, who=""):
    """Column group for a signal, from its carrier -- the only input that cannot drift.

    A signal with no carrier cannot be classified. The table drops a group it does not know,
    while a guessed group only renders in the wrong column, so this guesses the last group
    and logs a warning.
    """
    f = float(carrier_hz or 0.0)
    if f > 0.0:
        return _band_of(f)
    log_.warning(
        "signal %s has no carrier_hz; column group guessed as %r",
        who or "?",
        BAND_ORDER[-1],
    )
    return BAND_ORDER[-1]


def _band_of(carrier_hz):
    """RF-band column group (high/mid/low) from the sky carrier.

    The low/mid boundary is 1.22 GHz. Galileo E5 is one AltBOC(15,10) signal centred at
    1191.795 MHz and BeiDou B2 is ACE-BOC on the same centre, so E5b/B2b (1207.14) are the
    upper sidebands of the same coherent transmission as E5a/B2a (1176.45) and belong in one
    group. Low = the E5/B2 complex and GLONASS L3OC (1202.03); mid = L2C (1227.60), GLONASS
    L2 (1246-1248), B3I (1268.52), E6 (1278.75); high >= 1.40 GHz.

    The keys are deliberately not signal names: the groups are multi-constellation and display
    as High/Mid/Low, while the broker's per-signal taxonomy ('L5', 'E5b', 'B3', 'E6', 'L2') is
    a different axis that uses those names.
    """
    if carrier_hz >= 1.40e9:
        return "high"
    if carrier_hz >= 1.22e9:
        return "mid"
    return "low"


def _connect_host(host):
    """The address to connect to for `host`: a bind-all address (0.0.0.0, '', ::) is not a
    destination, so it becomes the loopback."""
    if host in ("0.0.0.0", "", "::"):
        return "127.0.0.1"
    return host


def discover_broker_chains(host, port, timeout=3.0):
    """Ask the broker publisher which chains it is running (GET /get_chains).

    Returns the viewer's chain dicts, or None on any failure.

    Each chain's descriptor (constellation, carrier, record length, whether a search feeds
    it) comes from the broker's own arguments, its --signal descriptor
    (lib/stages/gnss/gnssSignal.hpp) where it has one, so the viewer draws what is being
    tracked rather than a static table.

    The stage names it returns are chain ids, which the publisher accepts as a path segment
    (/gps_l5/get_status), so one viewer instance polls every constellation on one port.
    """
    import json as _json
    import urllib.request as _u

    host = _connect_host(host)
    try:
        with _u.urlopen("http://%s:%d/get_chains" % (host, port), timeout=timeout) as r:
            raw = _json.load(r)
    except Exception as e:  # noqa: BLE001 -- any failure -> caller falls back
        log_.info(
            "broker chain discovery: %s:%d/get_chains unavailable (%s)", host, port, e
        )
        return None
    if not isinstance(raw, list) or not raw:
        return None
    out = []
    for c in raw:
        chain = c.get("chain")
        if not chain:
            continue
        out.append(
            {
                "tag": c.get("constellation") or "G",
                # The table's column group (high/mid/low), from the carrier alone: GPS L5
                # and Galileo E5a share one. The broker's rf_band names the band physically
                # ("E5b", "B3", "L3", "E6"), a different taxonomy, and the table drops a
                # group it does not know (see _group_of).
                "band": _group_of(c.get("carrier_hz"), chain),
                "col": c.get("short") or chain,
                "name": c.get("label") or chain,
                "sigid": c.get("sigid"),
                # Both point at the chain id: the publisher serves merged detections and
                # merged status on one port and filters on the path segment.
                "combiner": chain,
                "search": chain if c.get("has_search") else None,
                "t_rec": float(c.get("t_rec") or 1e-3),
            }
        )
    log_.info(
        "broker chain discovery: %d chain(s) from %s:%d -- %s",
        len(out),
        host,
        port,
        ", ".join(c["col"] for c in out),
    )
    return out or None


def _gps_signal_capability():
    """{sigid: [PRNs whose satellite block broadcasts it]} for every signal in UNIFIED_SIGNALS
    that has a sigid, plus "_prns": {constellation letter: every PRN the registry knows there}.

    Lets the table distinguish "this satellite does not transmit that signal" from "it does and
    we are not seeing it": a Block IIR satellite carries none of L1C, L2C and L5, so blanks
    there are correct, while the same blanks on a Block III satellite are a fault; likewise a
    GLONASS-M+ satellite blank on L2OC (K generations only) against a K1B blank there.

    The source is the IGS metadata SINEX via gnss_broker.prnmap, per constellation
    (prnmap.prn_blocks), the registry the broker's --signal-capability filters on, and
    prnmap.block_carries owns which block carries each signal (L3OC on GLO-K and GLO-M+, L2OC on
    GLO-K only, B2a/B2b on BDS-3 only), so viewer and broker cannot disagree. SVN -> PRN
    assignments carry a validity window, so a re-used PRN resolves to the satellite flying it
    today. Celestrak TLE titles are not used: they carry identity errors, BeiDou worst. A
    constellation the registry does not list gets no "_prns" entry and no lists, and the client
    then shows its blanks as "no detection".

    Returns {} on any failure, and the client then marks nothing as not transmitted: never
    claim a satellite is incapable when that could not be checked.
    """
    try:
        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnss")
        )
        from gnss_broker import prnmap

        prnmap.ensure_registry()
        # One block map per constellation the inventory names (sys is backfilled by main()).
        blocks = {}
        for s in UNIFIED_SIGNALS:
            sysc = s.get("sys") or _SYS_TAG.get((s.get("sigid") or "")[:3])
            if sysc and sysc not in blocks:
                blocks[sysc] = prnmap.prn_blocks(sysc)
        if not any(blocks.values()):
            log_.warning(
                "block map unavailable: the IGS registry "
                "(~/.cache/kotekan_gps/igs_satellite_metadata.snx) is missing or "
                "unparseable; not-transmitted marking is off (fail open)."
            )
            return {}
        caps = {}
        for s in UNIFIED_SIGNALS:
            sid = s.get("sigid")
            sysc = s.get("sys") or _SYS_TAG.get((sid or "")[:3])
            if not sid or not blocks.get(sysc):
                continue  # no sigid, or no registry for it: the client shows "no detection"
            caps[sid] = sorted(
                p for p, b in blocks[sysc].items() if prnmap.block_carries(sid, b)
            )
        # The PRNs the registry knows per constellation, so the client can tell "this block
        # does not carry the signal" from "this PRN is not in the registry" (unknown, shown as
        # no detection).
        caps["_prns"] = {c: sorted(b) for c, b in blocks.items() if b}
        return caps
    except Exception as e:
        log_.warning(
            "GPS signal capability unavailable (%s); "
            "table cannot distinguish not-transmitted from not-detected",
            e,
        )
        return {}


class WsPortResource(resource.Resource):
    """``GET /wsport`` -> ``{"ws_port": N}`` for this server, plus the signal inventory when
    --unified.

    The browser loads the page from one server's http_port but must open the WebSocket on the
    same server's ws_port, so the client fetches this (same-origin, so it hits the right
    server) and connects to the port it returns.
    """

    isLeaf = True

    def __init__(self, ws_port, unified=False, broker_chains=None):
        resource.Resource.__init__(self)
        self.ws_port = ws_port
        self.unified = unified
        if not unified:
            return
        # Unified: advertise the whole signal inventory. The client keys satellites by
        # tag+prn across every band, polls every combiner listed, and draws the table from
        # `signals` (a sub-table per constellation, a column per signal under its band group).
        # `chains` is the sky legend and colours, one entry per constellation.
        self.signals = [dict(c) for c in broker_chains or []]
        # The transmitting-PRN list per signal, so the table can say "not transmitted"
        # instead of "not detected". Computed once (one cached-file parse); {} when
        # unavailable, and the client then marks nothing as not transmitted.
        self.capability = _gps_signal_capability()
        _legend = [
            {"tag": "G", "name": "GPS", "color": "#4d9de0"},
            {"tag": "E", "name": "Galileo", "color": "#e8923c"},
            {"tag": "C", "name": "BeiDou", "color": "#d64550"},
            # GLONASS PRNs are orbital slot numbers (the L3OC code index is the slot), so
            # tag+prn keys GLONASS satellites the same way as the rest.
            {"tag": "R", "name": "GLONASS", "color": "#9b6fd6"},
            # SBAS GEOs (WAAS 131/133/135): fixed points in the sky, positions from the
            # broker's /get_sky (the BRDC S records).
            {"tag": "S", "name": "SBAS", "color": "#3fae8c"},
        ]
        # Only the constellations actually being tracked: an empty sub-table for a
        # constellation this instrument cannot see reads as "nothing detected" rather
        # than "not configured".
        if broker_chains:
            _live = {c["tag"] for c in broker_chains}
            _legend = [g for g in _legend if g["tag"] in _live]
        self.chains = _legend

    def render_GET(self, request):
        request.responseHeaders.setRawHeaders("Content-Type", [b"application/json"])
        reply = {"ws_port": self.ws_port}
        if self.unified:
            reply["chains"] = self.chains
            reply["unified"] = True
            reply["signals"] = self.signals
            # {sigid: [transmitting PRNs], "_prns": {sys: [registered PRNs]}}
            reply["capability"] = self.capability
        return json.dumps(reply).encode("utf-8")


class GpsSkyResource(resource.Resource):
    """``GET /gps_sky`` -> predicted GNSS sky positions for the viewer's sky plot.

    Reply::

        {"ok": bool, "computing": bool, "lat": .., "lon": .., "alt": ..,
         "sats": [{"prn": N, "const": tag, "az": deg, "el": deg}, ...]}

    plus ``"error"`` after a failed compute. Azimuth is degrees clockwise from true
    North, elevation degrees above the horizon; only the TLE path applies
    ``mask_deg``. The broker's REST endpoints supply the locks and metrics; this
    endpoint supplies where each satellite is: the broker's /get_sky for G/E/C and the
    SBAS GEOs, and Celestrak TLE + skyfield for GLONASS and for everything with a TLE
    group when the broker cannot answer.

    Positions are recomputed in a worker thread, on request, at most every
    :data:`SKY_REFRESH_S`; ``render_GET`` only serves the cached result, so the
    reactor never blocks on the broker, skyfield or a TLE fetch. With no observer
    site, or when the compute raises, the reply is ``ok=False`` and the panel falls
    back to the REST-only locked list; a failed TLE fetch drops only that
    constellation.
    """

    isLeaf = True

    def __init__(
        self, lat, lon, alt, mask_deg, glonass_sigids=None, broker=(None, None)
    ):
        resource.Resource.__init__(self)
        self.lat, self.lon, self.alt, self.mask_deg = lat, lon, alt, mask_deg
        self._broker = broker  # (host, port) of the broker's publisher
        # The GLONASS signals running (sigids, e.g. GLO_L2OF, GLO_L3OC_P), so the R sky layer
        # can filter to what any of them can track rather than to one signal's capability.
        # Empty -> no filter (show every GLONASS satellite).
        self.glonass_sigids = list(glonass_sigids or [])
        self._cache = {"ok": False, "sats": []}  # last good (or empty) result
        self._sats = {}  # {tag: {prn: EarthSatellite}} (TLE path)
        self._sats_t = {}  # tag -> monotonic s of the last TLE load attempt
        # No broadcast ephemeris is parsed or fetched here: the broker serves the sky
        # (_brdc_skypos).
        # GLONASS slots any running R signal can track (None = no filter), and the monotonic
        # time they were computed (they are kept 1 h).
        self._glo_capable = None
        self._glo_capable_t = None
        self._last_compute = None  # monotonic s of last attempt
        self._refreshing = False

    # Viewer tag -> the "const" letter of a broker /get_sky row (no R: GLONASS is TLE-only;
    # S, the SBAS GEOs, has no TLE group here, so it comes from the broker alone).
    _TAG_SYS = {"G": "G", "E": "E", "C": "C", "S": "S"}

    def _need_refresh(self):
        if self.lat is None or self.lon is None:
            return False
        if self._last_compute is None:
            return True
        return (time.monotonic() - self._last_compute) >= SKY_REFRESH_S

    # Constellations on the sky plot: tag and Celestrak TLE group URL. A group whose TLE
    # fetch fails is skipped; the others still render.
    CONSTELLATIONS = (
        ("G", None),  # None -> gps_beamtrack.DEFAULT_TLE_URL (gps-ops)
        ("E", "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle"),
        ("C", "https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle"),
        ("R", "https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=tle"),
    )

    # Constellations taken from the broker's /get_sky. 'R' always comes from TLE, even when
    # the broker's sky is healthy; _compute merges the two sources rather than choosing.
    _BRDC_TAGS = frozenset(("G", "E", "C", "S"))
    # Broker-only constellations: no TLE group above, so they are asked of /get_sky alone and
    # drop out of the TLE fallback (the SBAS GEOs, from the BRDC S records).
    _BRDC_ONLY = ("S",)

    def _compute(self):
        """Runs in a worker thread. G/E/C/S come from the broker's /get_sky, the
        broadcast-ephemeris geometry the tracker uses, keyed by PRN: the sky plot agrees with
        what is locked and needs no TLE name -> PRN mapping. GLONASS, and every constellation
        with a TLE group when the broker cannot answer, comes from Celestrak TLE + skyfield;
        the SBAS GEOs have none and are absent then.

        The two sources are merged, not chosen between: the broker path has no GLONASS, so
        taking its answer alone would drop GLONASS from the sky plot while it is being tracked.
        """
        want = [c[0] for c in self.CONSTELLATIONS]
        brdc_want = [t for t in want if t in self._BRDC_TAGS] + list(self._BRDC_ONLY)
        tle_want = [t for t in want if t not in self._BRDC_TAGS]
        sats = self._brdc_skypos(brdc_want) if brdc_want else None
        if sats is None:
            # Broker sky unavailable: every constellation comes from TLE.
            return {"ok": True, "sats": self._tle_skypos(want)}
        if tle_want:
            try:
                sats = sats + self._tle_skypos(tle_want)
            except Exception as e:
                log_.warning(
                    "gps_sky: %s TLE sky failed (%s); serving the broker sky alone",
                    ",".join(tle_want),
                    e,
                )
        return {"ok": True, "sats": sats}

    def _glonass_capable_set(self):
        """Cached (1 h) set of GLONASS slots trackable by any running GLONASS signal: the union
        of each running R signal's capability. It must be a union, not one signal's set,
        because the R sky layer draws every GLONASS signal on one marker per satellite: L3OC is
        on the K generations and GLO-M+ while L2OF is on every satellite, so filtering to L3OC
        alone would hide satellites that L2OF tracks.

        The union follows self.glonass_sigids (the GLONASS signals in the broker's chain
        inventory at startup): L3OC alone -> the K and M+ slots; L2OF present -> every slot
        with a known frequency channel k. None => no filter (no GLONASS signals, or any one
        signal's lookup came back empty or raised: a union missing one signal's set would hide
        the satellites only that signal tracks)."""
        if (
            self._glo_capable_t is not None
            and (time.monotonic() - self._glo_capable_t) < 3600.0
        ):
            return self._glo_capable
        union, failed = set(), False
        try:
            import gps_beamtrack as bt
            from gnss_broker import prnmap

            for sigid in self.glonass_sigids:
                # FDMA signals use glonass_fdma_trackable(), not signal_capable_prns: when the
                # frequency plan is unavailable the latter returns every slot, which cannot be
                # told from a real known-k set. glonass_fdma_trackable() returns empty on
                # failure and the K-marker lookup raises, so a failed lookup never adds every
                # slot to the union.
                if "OF" in sigid.upper():
                    fdma = bt.glonass_fdma_trackable()  # known-k, empty on failure
                    failed = failed or not fdma
                    union |= fdma
                    continue
                # CDMA: the IGS registry, the source the broker filters on (the K generations
                # and GLO-M+); the Celestrak K marker only when the registry proves nothing,
                # and it cannot see the M+ satellites.
                cap = prnmap.registry_capable(sigid)
                if cap is None:
                    cap = set(
                        bt.signal_capable_prns(sigid, bt.GLONASS_TLE_URL)
                    )  # K set
                    failed = failed or not cap
                union |= cap
        except Exception as e:
            log_.warning(
                "gps_sky: GLONASS capability set unavailable (%s); showing all GLONASS "
                "under R",
                e,
            )
            failed = True
        # A failed lookup, or no GLONASS signals: stand the filter down and show all, never
        # hide a real satellite behind a failed capability check.
        self._glo_capable = None if failed else (union or None)
        self._glo_capable_t = time.monotonic()
        return self._glo_capable

    # Broker sky rows older than this are dropped rather than drawn, so a broker whose
    # prediction has stopped advancing does not show satellites standing still. When every
    # row is stale, _brdc_skypos returns None and the sky comes from TLE.
    _SKY_MAX_AGE_S = 300.0

    def _brdc_skypos(self, want):
        """Sky positions for the tags in `want` from the broker's /get_sky, or None if it
        cannot supply them.

        The broker already computes every satellite's az/el from the ephemeris it tracks with;
        reading them keeps the sky plot consistent with the tracker. This process must not
        fetch broadcast ephemeris itself: gnss_brdc_supply.fetch_brdc() refreshes the shared
        nav cache, and a long-lived second writer with its own copy of the merge code would
        revert the broker's store.

        Returns None on any failure, which is the cue for the TLE fallback.
        """
        import urllib.request  # module-local, matching this file's existing style

        host, port = self._broker
        if not host or not port:
            return None
        try:
            url = "http://%s:%d/get_sky" % (_connect_host(host), port)
            with urllib.request.urlopen(url, timeout=3.0) as r:
                rows = json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            log_.warning("gps_sky: broker sky unavailable (%s); falling back to TLE", e)
            return None
        if not rows:
            return None
        fresh = [
            x
            for x in rows
            if x.get("age_s") is None or x["age_s"] <= self._SKY_MAX_AGE_S
        ]
        if not fresh:
            log_.warning(
                "gps_sky: every broker sky row is older than %.0f s -- the broker's "
                "prediction has stopped advancing; falling back to TLE",
                self._SKY_MAX_AGE_S,
            )
            return None
        by_sys = {}
        for x in fresh:
            by_sys.setdefault(x.get("const"), []).append(x)
        sats = []
        for tag in want:
            sysc = self._TAG_SYS.get(tag)
            if not sysc:
                continue
            for x in sorted(by_sys.get(sysc, []), key=lambda r: r["prn"]):
                p = int(x["prn"])
                sats.append(
                    {
                        "prn": p,
                        "const": tag,
                        "az": round(float(x["az"]), 2),
                        "el": round(float(x["el"]), 2),
                    }
                )
        return sats or None

    # A TLE group that failed to load (or loaded empty) is retried after _TLE_RETRY_S. A loaded
    # group is reloaded after _TLE_RELOAD_S, and load_gps_satellites re-downloads a cache file
    # older than that, keeping the old copy if the download fails.
    _TLE_RETRY_S = 600.0
    _TLE_RELOAD_S = 86400.0

    def _tle_skypos(self, want=None):
        """Sky positions from Celestrak TLE + skyfield propagation, for the tags in `want`
        (default: all configured). The whole-sky fallback when the broker's sky is unavailable,
        and the only source for GLONASS. Every group is loaded on the first call and kept
        fresh per _TLE_RETRY_S / _TLE_RELOAD_S."""
        from gps_beamtrack import DEFAULT_TLE_URL, load_gps_satellites, predict_skypos

        now = time.monotonic()
        for tag, url in self.CONSTELLATIONS:
            t = self._sats_t.get(tag)
            hold = self._TLE_RELOAD_S if self._sats.get(tag) else self._TLE_RETRY_S
            if t is not None and now - t < hold:
                continue
            self._sats_t[tag] = now
            try:
                self._sats[tag] = load_gps_satellites(
                    url or DEFAULT_TLE_URL, max_age_s=self._TLE_RELOAD_S
                )
            except Exception as e:
                log_.warning("gps_sky: %s TLEs unavailable: %s", tag, e)
        cap_R = self._glonass_capable_set()
        sats = []
        for tag, by_prn in self._sats.items():
            if want is not None and tag not in want:
                continue
            pos = predict_skypos(self.lat, self.lon, self.alt, _sats=by_prn)
            sats += [
                {"prn": p, "const": tag, "az": round(az, 2), "el": round(el, 2)}
                for p, (az, el) in sorted(pos.items())
                if el >= self.mask_deg
                and not (tag == "R" and cap_R is not None and p not in cap_R)
            ]
        return sats

    def _kick_refresh(self):
        if self._refreshing:
            return
        self._refreshing = True

        def done(res):
            self._cache = res
            self._last_compute = time.monotonic()
            self._refreshing = False

        def err(failure):
            self._cache = {"ok": False, "sats": [], "error": failure.getErrorMessage()}
            self._last_compute = time.monotonic()
            self._refreshing = False
            log_.warning("gps_sky compute failed: %s", failure.getErrorMessage())

        threads.deferToThread(self._compute).addCallbacks(done, err)

    def render_GET(self, request):
        request.responseHeaders.setRawHeaders("Content-Type", [b"application/json"])
        request.setHeader(b"Access-Control-Allow-Origin", b"*")
        if self._need_refresh():
            self._kick_refresh()
        out = {
            "ok": bool(self._cache.get("ok")),
            "computing": self._refreshing,
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "sats": self._cache.get("sats", []),
        }
        if self._cache.get("error"):
            out["error"] = self._cache["error"]
        return json.dumps(out).encode("utf-8")


def _pvt_measurements(globs, max_age_s, t_now, smooth_s=90.0):
    """Per-(sys, prn, chain) code observation across the obs logs, for the PVT self-survey:
    the freshest row's epoch, carrier-smoothed over the last `smooth_s` seconds of its own
    arc (gnss_pvt.hatch_smooth). One row per satellite-chain. Residuals are wrapped,
    clock-dominated ranges, de-wrapped per group to the group median so the common clock is
    a single free parameter and no satellite straddles a code-period edge. Returns a list of
    {group, sys, prn, az, el, resid_m}, plus n, sd_m and method where smoothing succeeded.

    The group is the obs row's band, i.e. the chain: one chain, one clock.
    """
    import glob as _glob
    import json as _json
    import gnss_pvt as _pvt

    C = 299792458.0

    t_floor = t_now - max(max_age_s, smooth_s) - 5.0
    hist = {}  # (sys, prn, band) -> rows inside the smoothing window
    latest = {}  # (sys, prn, band) -> (t, group, az, el, resid_m, L_m)
    for pat in globs:
        for path in _glob.glob(pat):
            try:
                # A file last written before the window holds no row inside it: the glob
                # matches every past day's log, and each tail-read costs up to 4 MB.
                if os.path.getmtime(path) < t_floor:
                    continue
                size = os.path.getsize(path)
                with open(path) as f:
                    # tail: the smoothing window of every satellite on this chain. Rows are
                    # ~1.2 kB and a chain logs ~13 rows/s, so 4 MB holds several minutes.
                    f.seek(max(0, size - 4 * 1024 * 1024))
                    f.readline()
                    for line in f:
                        # cheap epoch pre-filter before the JSON decode: rows start {"t":...
                        if line.startswith('{"t":'):
                            try:
                                if float(line[5 : line.index(",", 5)]) < t_floor:
                                    continue
                            except Exception:
                                pass
                        try:
                            d = _json.loads(line)
                        except Exception:
                            continue
                        t = d.get("t")
                        sysid, prn, fr = d.get("sys"), d.get("prn"), d.get("band")
                        res, az, el = d.get("code_resid_m"), d.get("az"), d.get("el")
                        if (
                            t is None
                            or sysid is None
                            or prn is None
                            or fr is None
                            or res is None
                            or az is None
                            or el is None
                        ):
                            continue
                        if t < t_floor:
                            continue
                        # a replica that is not on its signal (a noise probe, or not
                        # fleet_present) has a code phase but no range
                        if d.get("noise_probe") or d.get("fleet_present") is False:
                            continue
                        key = (sysid, prn, fr)
                        hist.setdefault(key, []).append(d)
                        if t_now - t > max_age_s:
                            continue
                        if key in latest and latest[key][0] >= t:
                            continue
                        cl, ch = d.get("code_len"), d.get("chip_rate_hz")
                        L = (cl / ch * C) if (cl and ch) else None
                        latest[key] = (t, fr, az, el, float(res), L)
            except Exception:
                continue
    # Carrier-smooth each satellite's code residual over its own arc (gnss_pvt.hatch_smooth):
    # the code observable is white at ~0.1-0.25 chips per record, while the carrier holds arcs
    # for hours.
    extra = {}
    for key, v in list(latest.items()):
        hs = _pvt.hatch_smooth(hist.get(key, []), smooth_s, wrap_m=v[5])
        if hs is None:
            continue
        r_s, n, sd, method = hs
        latest[key] = (v[0], v[1], v[2], v[3], float(r_s), v[5])
        extra[key] = {"n": n, "sd_m": sd, "method": method}

    # de-wrap each group's residuals to the group median (handles the clock straddling a
    # code-period boundary; L is the code period in metres).
    import statistics as _st

    by_group = {}
    for (sysid, prn, fr), v in latest.items():
        by_group.setdefault(v[1], []).append((sysid, prn, fr, v))
    out = []
    for group, rows in by_group.items():
        med = _st.median([r[3][4] for r in rows])
        for sysid, prn, fr, v in rows:
            _t0, g, az, el, res, L = v
            if L:
                res = res - L * round(
                    (res - med) / L
                )  # unwrap to within +-L/2 of the median
            m = {
                "group": g,
                "sys": sysid,
                "prn": prn,
                "az": az,
                "el": el,
                "resid_m": res,
            }
            m.update(extra.get((sysid, prn, fr), {}))
            out.append(m)
    return out


class DecodeHealthResource(resource.Resource):
    """``GET /decode_health`` -> merged nav-decode health from the brokers' JSON files
    plus a PVT self-survey computed from the obs-log code residuals (:mod:`gnss_pvt`). Stale
    broker files / obs rows are dropped, so a decoder or the position solve going dark shows up
    as the row/section disappearing, not a frozen last value. PVT is cached (recomputed every
    `pvt_ttl_s`) since it tail-reads several obs logs and solves."""

    isLeaf = True

    def __init__(self, dirpath, lat=None, lon=None, alt=None, obs_globs=None):
        resource.Resource.__init__(self)
        self.dirpath = dirpath
        self.max_age_s = 120.0
        self.lat, self.lon, self.alt = lat, lon, (alt or 0.0)
        self.obs_globs = obs_globs or []
        self.pvt_ttl_s = 15.0
        # Carrier-smoothing window (gnss_pvt.hatch_smooth) and the elevation mask. Low
        # satellites carry the multipath and the thickest ionosphere; both are slow, so no
        # amount of smoothing removes them -- a low satellite wanders by chips over minutes
        # while the high ones sit within a few hundredths.
        self.pvt_smooth_s = 90.0
        self.pvt_min_el_deg = 15.0
        # PVT wants near-simultaneous observations: the receiver clock (a free per-group param) is
        # assumed common across a group's sats, and a disciplined-oscillator drift of ~1e-9 over
        # 120 s is already ~36 m. 30 s keeps that error small while still catching ~1 row/sat.
        self.pvt_max_age_s = 30.0
        self._pvt_cache = (0.0, None)

    def _pvt(self, t_now):
        if self.lat is None or self.lon is None or not self.obs_globs:
            return None
        if t_now - self._pvt_cache[0] < self.pvt_ttl_s:
            return self._pvt_cache[1]
        try:
            import gnss_pvt as _pvt

            meas = _pvt_measurements(
                self.obs_globs, self.pvt_max_age_s, t_now, smooth_s=self.pvt_smooth_s
            )
            res = (
                _pvt.solve(
                    meas, self.lat, self.lon, self.alt, min_el_deg=self.pvt_min_el_deg
                )
                if meas
                else None
            )
            if res is not None:
                res["apriori"] = {"lat": self.lat, "lon": self.lon, "alt": self.alt}
                res["n_meas"] = len(meas)
                res["smooth_s"] = self.pvt_smooth_s
                res["min_el_deg"] = self.pvt_min_el_deg
                # the per-satellite inputs, so the panel can show what the fit ate
                res["meas"] = [
                    {
                        k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in m.items()
                    }
                    for m in meas
                ]
        except Exception as e:
            res = {"error": str(e)}
        self._pvt_cache = (t_now, res)
        return res

    def render_GET(self, request):
        request.responseHeaders.setRawHeaders("Content-Type", [b"application/json"])
        request.setHeader(b"Access-Control-Allow-Origin", b"*")
        try:
            import time as _t
            import decode_health as _dh

            now = _t.time()
            out = _dh.read_all(self.dirpath, max_age_s=self.max_age_s, t_now=now)
            out["pvt"] = self._pvt(now)
        except Exception as e:
            out = {"signals": {}, "chains": [], "error": str(e)}
        return json.dumps(out).encode("utf-8")


def main():
    ap = argparse.ArgumentParser(
        description="Bridge kotekan networkPowerStream to a browser viewer."
    )
    ap.add_argument(
        "--kotekan-host",
        default="0.0.0.0",
        help="TCP interface to bind for kotekan input (default 0.0.0.0)",
    )
    ap.add_argument(
        "--kotekan-port",
        default=23401,
        type=int,
        help="TCP port for kotekan input (default 23401)",
    )
    ap.add_argument(
        "--ws-port",
        default=8539,
        type=int,
        help="WebSocket port for browser clients (default 8539)",
    )
    ap.add_argument(
        "--http-port",
        default=8080,
        type=int,
        help="HTTP port serving the static viewer (default 8080)",
    )
    ap.add_argument(
        "-w",
        "--launch-browser",
        action="store_true",
        help="Open the viewer in a local browser on startup",
    )
    ap.add_argument(
        "--viewer-integration-ms",
        type=float,
        default=DEFAULT_VIEWER_INTEGRATION_MS,
        help="Browser-facing integration window in ms. kotekan "
        "integrations arriving within this window are "
        "averaged into one emitted frame (default "
        f"{DEFAULT_VIEWER_INTEGRATION_MS}). Larger = less "
        "data / smoother; smaller = finer time resolution.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")

    g = ap.add_argument_group("viewer modules")
    g.add_argument(
        "--kotekan-rest-port",
        default=12048,
        type=int,
        help="Port for kotekan's REST server (the browser fetches "
        "airspy gain/freq/adcstat via cross-origin requests)",
    )
    g.add_argument(
        "--airspy-stages",
        nargs="*",
        default=None,
        help="kotekan stage names of the airspy producers; "
        "default ['airspy_input'] for autocorr, "
        "['airspy_inputA','airspy_inputB'] for crosscorr",
    )
    g.add_argument(
        "--no-airspy-controls",
        action="store_true",
        help="Force-disable the airspy gain/freq panels even if "
        "airspy_stages is set",
    )
    g.add_argument(
        "--lag-align-stage",
        default=None,
        help="kotekan stage name for AirspyAlign; enables the Lagcorr "
        "control when set (crosscorr only)",
    )
    g.add_argument(
        "--ccera-pointing",
        action="store_true",
        help="Enable the CCERA telescope pointing panel "
        "(needs ccera_rest.py running on :3000)",
    )
    g.add_argument(
        "--galaxy-view-url",
        default=None,
        help="Background image URL for the all-sky galaxy view. "
        "Pass a URL to enable; leave unset to disable.",
    )

    gg = ap.add_argument_group("GPS live-status panel")
    gg.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Site latitude (deg, WGS84) for the sky positions and the a-priori of the "
        "PVT self-survey. Give the site the broker and the obs writers use: the obs "
        "logs' code residuals are relative to it. Without --lat and --lon there are no "
        "sky positions and no survey; the table still lists the locked satellites.",
    )
    gg.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Site longitude (deg east, WGS84); see --lat.",
    )
    gg.add_argument(
        "--alt",
        type=float,
        default=0.0,
        help="Site height (m above the WGS84 ellipsoid) for the sky positions and the "
        "PVT a-priori (default %(default)s).",
    )
    gg.add_argument(
        "--gps-mask-deg",
        type=float,
        default=5.0,
        help="Elevation mask (deg, default %(default)s) for the sky positions computed "
        "from TLE: GLONASS, and every constellation when the broker's /get_sky cannot "
        "answer. Positions from /get_sky are not masked.",
    )
    gg.add_argument(
        "--decode-health-dir",
        default=os.path.join(
            os.path.expanduser("~"), ".cache", "kotekan_gps", "decode_health"
        ),
        help="directory of the brokers' nav-decode health JSON files (each broker's "
        "--decode-health-file; only *.json is read), served merged at /decode_health "
        "for the viewer panel; a file older than 120 s is dropped. Default: %(default)s",
    )
    gg.add_argument(
        "--pvt-obs-globs",
        default="",
        help="comma-separated globs of obs-log jsonl (code_resid_m + az/el) the PVT "
        "self-survey reads; served under /decode_health's pvt field. Needs --lat and "
        "--lon. Unset = no survey.",
    )
    gg.add_argument(
        "--unified",
        action="store_true",
        help="Unified table: one row per satellite, every signal it carries shown "
        "side by side. /wsport carries the signal inventory the broker at "
        "--kotekan-host (0.0.0.0 = this host) and --kotekan-rest-port reports on "
        "/get_chains, read once at startup. The GPS panel polls only those signals, so "
        "without --unified its table is empty. Also serves the GPS panel's endpoints; "
        "the page shows the GPS panel only with --no-power-stream.",
    )
    gg.add_argument(
        "--no-power-stream",
        action="store_true",
        help="Run GPS-only: don't wait for / serve a kotekan power "
        "stream (no waterfall), so the GPS panel works without a "
        "full-rate diagnostic PFB. Implies the GPS panel.",
    )
    args = ap.parse_args()
    # The GPS panel (/gps_sky, /decode_health) runs on a GPS-only or unified page.
    gps_enabled = args.no_power_stream or args.unified

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.startLogging(sys.stdout, setStdout=False)

    # The signal list comes from the broker: the chains it is running, with constellation,
    # carrier, record length and whether a search feeds each. Reassigning the module global is
    # enough: _gps_signal_capability() reads it at call time. Done after logging is configured
    # so the discovery summary lands in the viewer log.
    _bchains = None
    if args.unified:
        _bchains = discover_broker_chains(args.kotekan_host, args.kotekan_rest_port)
        if _bchains:
            global UNIFIED_SIGNALS
            UNIFIED_SIGNALS = _bchains
            log_.info(
                "signal table: %d row(s) from the broker -- %s",
                len(_bchains),
                ", ".join("%s/%s" % (r.get("band"), r.get("col")) for r in _bchains),
            )
        # Backfill "sys" (the satellite's real constellation), derived from the signal id's
        # prefix where there is one, else the display tag.
        for _s in UNIFIED_SIGNALS:
            _s.setdefault(
                "sys",
                _SYS_TAG.get((_s.get("sigid") or "").split("_", 1)[0], _s.get("tag")),
            )

    static_dir = os.path.dirname(os.path.abspath(__file__))

    # GPS-only mode skips the kotekan power stream entirely (no blocking accept,
    # no waterfall) and serves only the GNSS panels (/gps_sky, /decode_health).
    # Otherwise stand up the power stream (blocks on accept until kotekan connects).
    if args.no_power_stream:
        kotekan = None
        log_.info("GPS-only mode: no power stream (waterfall disabled)")
    else:
        kotekan = KotekanPowerStream(
            host=args.kotekan_host,
            port=args.kotekan_port,
            emit_interval_s=args.viewer_integration_ms / 1000.0,
        )
        kotekan.start()  # blocks on accept until kotekan connects
        if kotekan.mode == "unknown":
            log_.warning(
                "Unrecognised kotekan nvis=%d; proceeding anyway.", kotekan.frame_nvis
            )

    viewer_config = build_viewer_config(kotekan, args)
    log_.info("viewer_config: %s", json.dumps(viewer_config, indent=2))

    ws_factory = LiveBeamWSFactory(kotekan, viewer_config)
    if kotekan is not None:
        kotekan.on_frame = ws_factory.broadcast

    reactor.listenTCP(args.ws_port, ws_factory)

    # HTTP root: static files + /mode endpoint. NoCacheFile saves us a lot of
    # "I edited the JS but the browser ignored me" trouble during development.
    root = NoCacheFile(static_dir)
    root.putChild(b"mode", ModeResource(kotekan))
    # One viewer, every constellation: the broker publisher names its own chains. Reuse
    # _bchains from the signal table above; asking again could return a different answer.
    root.putChild(
        b"wsport",
        WsPortResource(args.ws_port, unified=args.unified, broker_chains=_bchains),
    )

    # Sky positions and decode health for the GNSS panels. Only stood up when the panel is
    # enabled, so a plain radio-astronomy run does not pull in skyfield or fetch TLEs.
    if gps_enabled:
        # The GNSS helpers (gps_beamtrack, gnss_pvt, decode_health) are bare-name imports from
        # python/scripts/gnss. Without it on the path every /gps_sky compute fails, and the
        # sky plot is empty with the import error only in the reply and the log.
        _scripts = os.path.dirname(static_dir)
        sys.path.insert(0, _scripts)
        sys.path.insert(0, os.path.join(_scripts, "gnss"))
        # GLONASS sky filter is the union of the running R signals' capabilities (L3OC = K and
        # M+, L2OF = all known-k), so a satellite trackable by any of them shows. Derived from
        # the same discovered inventory the table uses, so a band swap needs no viewer edit.
        _glo_sigids = sorted(
            {
                s.get("sigid")
                for s in UNIFIED_SIGNALS
                if (s.get("sigid") or "").startswith("GLO_")
            }
            - {None}
        )
        root.putChild(
            b"gps_sky",
            GpsSkyResource(
                args.lat,
                args.lon,
                args.alt,
                args.gps_mask_deg,
                glonass_sigids=_glo_sigids,
                broker=(args.kotekan_host, args.kotekan_rest_port),
            ),
        )
        root.putChild(
            b"decode_health",
            DecodeHealthResource(
                args.decode_health_dir,
                lat=args.lat,
                lon=args.lon,
                alt=args.alt,
                obs_globs=[g for g in args.pvt_obs_globs.split(",") if g],
            ),
        )
        log_.info(
            "GPS panel enabled (site lat=%s lon=%s alt=%s); decode-health dir %s",
            args.lat,
            args.lon,
            args.alt,
            args.decode_health_dir,
        )

    reactor.listenTCP(args.http_port, server.Site(root))

    def _on_shutdown():
        if kotekan is not None:
            kotekan.close()
        ws_factory.close_all()

    reactor.addSystemEventTrigger("before", "shutdown", _on_shutdown)

    if args.launch_browser:
        import webbrowser

        webbrowser.open(f"http://localhost:{args.http_port}/")

    # Idempotent shutdown. When SpawnProcess wraps this script as a child of
    # kotekan, a single Ctrl-C delivers SIGINT to both kotekan and to us via
    # the foreground process group, and SpawnProcess *also* sends SIGINT on
    # its own shutdown -- so we'd get two stop attempts and the second one
    # raises ``ReactorNotRunning``.
    _stop_called = [False]

    def _safe_stop():
        if _stop_called[0]:
            return
        _stop_called[0] = True
        if reactor.running:
            reactor.stop()

    def _on_signal(sig, frame):
        reactor.callFromThread(_safe_stop)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    reactor.run(installSignalHandlers=False)


if __name__ == "__main__":
    main()
