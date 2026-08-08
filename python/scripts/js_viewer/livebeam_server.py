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
import math
import os
import re
import select
import signal
import socket
import struct
import sys
import time
from datetime import datetime, timezone

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


# How often the /gps_sky endpoint recomputes satellite az/el. GPS sats move
# <~0.1 deg/s, so a few seconds is plenty; the recompute runs in a worker
# thread (skyfield + the one-time TLE fetch) and render_GET only ever serves
# the cached result, so the Twisted reactor never blocks on orbit propagation.
SKY_REFRESH_S = 5.0

# /gps_sky reports geometry down to this elevation (deg) rather than masking at the display
# mask, so a BELOW-HORIZON tracked PRN (e.g. a --noise-probe, seeded deep-below to measure the
# noise floor) carries its REAL negative elevation instead of collapsing to null. That lets the
# viewer tell "below horizon / probe" (el < 0) apart from "geometry momentarily missing" (el ==
# null) unambiguously. -90 => every computable sat; consumers apply their own display threshold
# (the response still carries `mask_deg`, the display mask, separately).
SKY_REPORT_FLOOR_DEG = -90.0


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
    waterfall, so mode is "gps" and the client builds just the GPS panel.
    """
    if kotekan is None:
        # GPS-only: no power stream, so no waterfall/spectrum fields. The client
        # sees mode "gps" and builds only the GPS Sky panel (REST-polled).
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
        nfreq, nvis, vis_labels = (int(kotekan.frame_nfreq), int(kotekan.frame_nvis),
                                   kotekan.vis_labels)
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
            # GPS live-status panel (overhead-sky circle + locked-PRN list). It
            # polls the kotekan GNSS stages (search/combiner) for locks + |A| and
            # this server's /gps_sky endpoint for satellite az/el.
            "gps_sky": bool(getattr(args, "_gps_enabled", False)),
        },
        # Where the GPS panel finds its data. ``has_site`` is whether this server
        # knows the observer location (so it can serve /gps_sky); without it the
        # panel still shows the locked list, just no sky positions.
        "gps": {
            "search_stage": args.gps_search_stage,
            "combiner_stage": args.gps_combiner_stage,
            "airspy_stage": args.gps_airspy_stage,
            "mask_deg": args.gps_mask_deg,
            "has_site": args.lat is not None and args.lon is not None,
        },
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
        # GPS-only mode (no power stream): just ship the viewer_config and stop.
        # There are no power frames to push, so we skip the producer/backpressure
        # plumbing and the FREQLIST entirely. The client builds the GPS panel.
        if kotekan is None:
            self.sendMessage(
                json.dumps({"nfreq": 0,
                            "viewer_config": self.factory.viewer_config}).encode("utf-8"),
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


# Per-band constellation display defs (name shown in the sky legend + record period t_rec used
# for C/N0 = x / t_rec). tag/colour are constant across bands (G/E/C = blue/orange/red, matching
# the matplotlib composite-map palette); only the SIGNAL name and t_rec change. The client swaps
# its L1 defaults for these on startup (fetched via /wsport), so a non-L1 viewer's legend names
# the right signal instead of "GPS L1 C/A / Galileo E1C / BeiDou B1C".
BAND_CHAINS = {
    "l1":  [{"tag": "G", "name": "GPS L1 C/A",  "t_rec": 1e-3,  "color": "#4d9de0"},
            {"tag": "E", "name": "Galileo E1C", "t_rec": 4e-3,  "color": "#e8923c"},
            {"tag": "C", "name": "BeiDou B1C",  "t_rec": 10e-3, "color": "#d64550"},
            # GPS L1C-P (4th L1 chain): constellation G / PRNs 1-32 like C/A, so it needs a
            # DISTINCT viewer tag "L" (the series key is tag+prn) backed by explicit l1c_ stages
            # in WsPortResource.base -- reusing "G" would merge it into the C/A series.
            {"tag": "L", "name": "GPS L1C",     "t_rec": 10e-3, "color": "#6fbf73"}],
    # l2c dongle REPURPOSED to BeiDou B2b (2026-08-03): tag C, 1 ms records (was GPS L2C, tag G,
    # 20 ms). The mid-band swap (mid-band-gal-bds-receivers); the stage names stay l2c_gps_* (the
    # Shape-A primary convention), only the constellation/signal displayed change.
    "l2c": [{"tag": "C", "name": "BeiDou B2b",  "t_rec": 1e-3,  "color": "#d64550"}],
    "l5":  [{"tag": "G", "name": "GPS L5",      "t_rec": 1e-3,  "color": "#4d9de0"},
            {"tag": "E", "name": "Galileo E5a", "t_rec": 1e-3,  "color": "#e8923c"},
            {"tag": "C", "name": "BeiDou B2a",  "t_rec": 1e-3,  "color": "#d64550"}],
    # GLONASS L3OC (2026-08-04) -- the fourth constellation, tag R, 1 ms records. PRNs are
    # orbital SLOT numbers. Only the 7 GLONASS-K satellites transmit it; the sky layer filters
    # to those via _l3oc_capable_set, the same way "L" filters to GPS III.
    "l3oc": [{"tag": "R", "name": "GLONASS L3OC", "t_rec": 1e-3, "color": "#9b6fd6"}],
}

# UNIFIED VIEWER (2026-07-28): one page, one satellite per row, every signal that satellite
# carries shown side by side -- the shared-knowledge paradigm made visible (L1 C/A, L1C,
# L2C-CM, L2C-CL, L5 are ONE satellite). The merged kotekan already serves every band's
# stages on one REST port, so the client polls all of these; the split into three viewer
# ports was purely presentational. Each row here is ONE signal:
#   tag        constellation (G/E/C) -- the satellite key is tag+prn across ALL bands
#   band       frequency-band column group in the table (L1 / L2 / L5)
#   col        the short column label under that group ("CA","L1C","CM","CL","Q","E1C",...)
#   name       full signal name (tooltips / amp-history legend)
#   combiner   ABSOLUTE kotekan stage (merged instance -- already l1_/l2c_/l5_ prefixed)
#   search     detections stage, or None for a derived pilot (L2C-CL has no search)
#   t_rec      record period (s) -> incoherent C/N0 = x / t_rec
#   peel       whether this chain runs the voltage peel (peel-depth column applies)
# The table renders columns in THIS order; add a signal by adding a row.
#
# STATIC FALLBACK ONLY (2026-08-03): main() replaces this at startup with the list DISCOVERED
# from kotekan's running /config (see discover_signals() below), so a band/constellation swap in
# gnss_node.yaml -- L2C dongle -> BeiDou B2b today, Galileo E5b next, GLONASS later -- flows into
# the viewer with no edit here. This literal is used only when /config is unreachable at launch.
UNIFIED_SIGNALS = [
    {"tag": "G", "band": "L1", "col": "CA",  "name": "GPS L1 C/A", "sigid": "GPS_L1CA",
     "combiner": "l1_gps_combiner", "search": "l1_gps_search",  "t_rec": 1e-3,  "peel": True},
    {"tag": "G", "band": "L1", "col": "L1C", "name": "GPS L1C-P", "sigid": "GPS_L1C_P",
     "combiner": "l1_l1c_combiner", "search": "l1_l1c_search",  "t_rec": 10e-3, "peel": True},
    # L2/MID band: the l2c dongle was REPURPOSED from GPS L2C (CM data + CL pilot) to BeiDou B2b
    # (2026-08-03, mid-band-gal-bds-receivers). B2b is DATA-only (no pilot), tag C, 1 ms records;
    # the primary stage keeps its l2c_gps_* name (Shape-A convention), only the signal changes.
    {"tag": "C", "band": "L2", "col": "B2b", "name": "BeiDou B2b (data, B-CNAV3)", "sigid": "BDS_B2B_I",
     "combiner": "l2c_gps_combiner", "search": "l2c_gps_search", "t_rec": 1e-3,  "peel": False},
    {"tag": "G", "band": "L5", "col": "Q",   "name": "GPS L5-Q", "sigid": "GPS_L5_Q",
     "combiner": "l5_gps_combiner", "search": "l5_gps_search",  "t_rec": 1e-3,  "peel": True},
    # L5-I is the DATA component of the same signal L5-Q pilots (S4 phase 2). Like L2C-CL it
    # is DERIVED, not acquired -- seeded verbatim from its sibling's rows -- so search is None
    # and it will never show an SNR cell. Its combiner runs the composed NH10 + CNAV navwipe,
    # which is what lets it integrate past the 20 ms symbol boundary. Judge it against Q, its
    # equal-power sibling, not against an absolute: measured 6.4 dB below Q at the 1 s rung
    # (2026-07-29), down from 15.4 dB before the navwipe.
    {"tag": "G", "band": "L5", "col": "I",   "name": "GPS L5-I (data, CNAV)", "sigid": "GPS_L5_I",
     "combiner": "l5_i_combiner",   "search": None,             "t_rec": 1e-3,  "peel": False},
    {"tag": "E", "band": "L1", "col": "E1C", "name": "Galileo E1-C",
     "combiner": "l1_gal_combiner", "search": "l1_gal_search",  "t_rec": 4e-3,  "peel": True},
    # E1B / E5a-I / B1C-data / B2a-data: the DATA components (S5 D-components), each DERIVED off
    # its band-mate pilot (search None, seeded verbatim). Their combiners run the nav-bit wipe
    # (navwipe_bit_records) that lets a data channel integrate to 1 s, so they carry coherent
    # C/N0 like any chain; judge each against its pilot sibling, not an absolute. Unpeeled
    # (peel is GPS-only for now).
    {"tag": "E", "band": "L1", "col": "E1B", "name": "Galileo E1-B (data, I/NAV)",
     "combiner": "l1_e1b_combiner",  "search": None,            "t_rec": 4e-3,  "peel": False},
    {"tag": "E", "band": "L5", "col": "E5a", "name": "Galileo E5a-Q",
     "combiner": "l5_gal_combiner", "search": "l5_gal_search",  "t_rec": 1e-3,  "peel": True},
    {"tag": "E", "band": "L5", "col": "E5aI","name": "Galileo E5a-I (data, F/NAV)",
     "combiner": "l5_e5a_i_combiner","search": None,            "t_rec": 1e-3,  "peel": False},
    {"tag": "C", "band": "L1", "col": "B1C", "name": "BeiDou B1C",
     "combiner": "l1_bds_combiner", "search": "l1_bds_search",  "t_rec": 10e-3, "peel": True},
    {"tag": "C", "band": "L1", "col": "B1CD","name": "BeiDou B1C (data, B-CNAV1)",
     "combiner": "l1_b1c_d_combiner","search": None,            "t_rec": 10e-3, "peel": False},
    {"tag": "C", "band": "L5", "col": "B2a", "name": "BeiDou B2a",
     "combiner": "l5_bds_combiner", "search": "l5_bds_search",  "t_rec": 1e-3,  "peel": True},
    {"tag": "C", "band": "L5", "col": "B2aD","name": "BeiDou B2a (data, B-CNAV2)",
     "combiner": "l5_b2a_d_combiner","search": None,            "t_rec": 1e-3,  "peel": False},
]

# ---------------------------------------------------------------------------
# Signal-list discovery from kotekan's running /config (the swap-proof path).
#
# The hardcoded UNIFIED_SIGNALS above needed a hand-edit on every band swap. Instead we read the
# live merged config and reconstruct the table. Sources, all authoritative and per-chain:
#   * the 13 `cudaGnssTrack` commands (nested in the per-band `cudaProcess` stages) -- each carries
#     a seed_endpoint `/<chain>_track/set_seeds` (-> the combiner `<chain>_combiner`), the chain's
#     real `signal` (None on a primary chain -> inherits, so we fall back to its search stage), and
#     a per-chain `peel` flag.
#   * the `GnssChannelizedSearch` stages -- `<chain>_search`: the acquired chains' signal, and the
#     "is this chain acquired vs derived" flag (a derived data/pilot sibling has no search).
# band + t_rec come from the signal descriptor (parsed from gnssSignal.hpp, the C++ source of
# truth), so a signal newly added there flows through with no python edit. Only the display
# labels (tag/column/full-name) are genuine presentation -- SIG_DISPLAY, with a derived fallback.

# Presentation metadata per signal id: (viewer tag, short column label, full name). The tag is the
# satellite-series key (tag+prn) across every band -- GPS L1C is "L" not "G" so it stays a distinct
# series from C/A. Everything else about a signal is discovered; a signal absent here still renders
# via _display_for()'s derived fallback, so this table only prettifies -- it is not required.
SIG_DISPLAY = {
    "GPS_L1CA":   ("G", "CA",   "GPS L1 C/A"),
    "GPS_L1C_P":  ("L", "L1C",  "GPS L1C-P"),
    "GPS_L2C_CM": ("G", "CM",   "GPS L2C-CM"),
    "GPS_L2C_CL": ("G", "CL",   "GPS L2C-CL (pilot)"),
    "GPS_L5_Q":   ("G", "Q",    "GPS L5-Q"),
    "GPS_L5_I":   ("G", "I",    "GPS L5-I (data, CNAV)"),
    "GAL_E1C":    ("E", "E1C",  "Galileo E1-C"),
    "GAL_E1B":    ("E", "E1B",  "Galileo E1-B (data, I/NAV)"),
    "GAL_E5A_Q":  ("E", "E5a",  "Galileo E5a-Q"),
    "GAL_E5A_I":  ("E", "E5aI", "Galileo E5a-I (data, F/NAV)"),
    "GAL_E5B_Q":  ("E", "E5b",  "Galileo E5b-Q (pilot)"),
    "GAL_E5B_I":  ("E", "E5bI", "Galileo E5b-I (data, I/NAV)"),
    "BDS_B1C_P":  ("C", "B1C",  "BeiDou B1C"),
    "BDS_B1C_D":  ("C", "B1CD", "BeiDou B1C (data, B-CNAV1)"),
    "BDS_B2A_P":  ("C", "B2a",  "BeiDou B2a"),
    "BDS_B2A_D":  ("C", "B2aD", "BeiDou B2a (data, B-CNAV2)"),
    "BDS_B2B_I":  ("C", "B2b",  "BeiDou B2b (data, B-CNAV3)"),
    # Signals added after this table was last swept -- without a row each renders with its raw
    # id and a one-letter column ("D"), which reads as a bug in the table rather than an
    # omission in it.
    "GPS_L1C_D":  ("L", "L1CD", "GPS L1C-D (data, CNAV-2) -- carrier UNMODULATED, see notes"),
    "GAL_E6_C":   ("E", "E6C",  "Galileo E6-C (pilot)"),
    "GAL_E6_B":   ("E", "E6B",  "Galileo E6-B (data, HAS)"),
    "BDS_B1I":    ("C", "B1I",  "BeiDou B1I (legacy, D1)"),
    "BDS_B3I":    ("C", "B3I",  "BeiDou B3I (legacy, D1)"),
    "BDS_B2I":    ("C", "B2I",  "BeiDou B2I (legacy BDS-2, D1)"),
    "GLO_L3OC_P": ("R", "L3OC", "GLONASS L3OC-p (pilot)"),
    "GLO_L3OC_D": ("R", "L3OCd", "GLONASS L3OC-d (data)"),
}
_SYS_TAG = {"GPS": "G", "GAL": "E", "BDS": "C", "GLO": "R"}


def _display_for(sigid):
    """(tag, col, name) for a signal id -- SIG_DISPLAY if known, else derived from the id."""
    if sigid in SIG_DISPLAY:
        return SIG_DISPLAY[sigid]
    pre = sigid.split("_", 1)[0]
    tag = _SYS_TAG.get(pre, (pre[:1] or "?"))
    col = sigid.split("_")[-1]
    return (tag, col, sigid)


def _band_of(carrier_hz):
    """RF-band column group (L1/L2/L5) from the sky carrier. The mid band (~1200-1280 MHz:
    L2C, B2b/E5b, B3I, E6) groups under 'L2'; anything below 1.2 GHz (L5/E5a/B2a) under 'L5'."""
    if carrier_hz >= 1.40e9:
        return "L1"
    if carrier_hz >= 1.20e9:
        return "L2"
    return "L5"


def _signal_descriptors():
    """Parse gnssSignal.hpp -> {sigid: (carrier_hz, code_period_s)}.

    That C++ table is the single source of truth for every signal's sky carrier and primary-code
    period, so parsing it here means a signal added there (e.g. Galileo E5b) supplies its own
    band + record period with no edit in this file. Empty dict if the header can't be found ->
    the caller drops back to the static table.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.normpath(os.path.join(here, "../../../lib/stages/gnss/gnssSignal.hpp")),
        os.path.normpath(os.path.join(here, "../../../../lib/stages/gnss/gnssSignal.hpp")),
    ]
    text = None
    for p in cands:
        try:
            with open(p) as f:
                text = f.read()
            break
        except OSError:
            continue
    out = {}
    if not text:
        return out
    # Each descriptor initialiser begins:  "NAME", <carrier_hz>, <chip_rate>, <code_len>, <period>,
    num = r"[0-9][0-9.eE+-]*"
    pat = re.compile(r'"([A-Z0-9_]+)"\s*,\s*(%s)\s*,\s*%s\s*,\s*[0-9]+\s*,\s*(%s)' % (num, num, num))
    for m in pat.finditer(text):
        try:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        except ValueError:
            pass
    return out



def discover_broker_chains(host, port, timeout=3.0):
    """Ask a BROKER publisher what chains it is actually running (task #27 M6).

    Returns the viewer's chain dicts, or None on any failure so the caller keeps its
    existing behaviour.

    WHY THIS EXISTS. discover_signals() below reconstructs the signal table from KOTEKAN's
    /config, which a broker publisher does not serve -- so a viewer pointed at a broker port
    silently fell back to the static airspy table and polled stage names that have never
    existed on CHORD (gal_search, gal_combiner, airspy_in). The symptom reaching the user
    was a CORS error, which is three layers away from the cause.

    The broker knows every chain it runs, its constellation, its band and its record length,
    because those came from lib/stages/gnss/gnssSignal.hpp. Asking IT is both simpler and
    honest: what the viewer draws is then what is actually being tracked, not what a table
    written for a different instrument says should be.

    The stage names it returns are CHAIN IDS, which the publisher accepts as a path segment
    (/gps_l5/get_status) -- so one viewer instance polls every constellation on one port.
    """
    import json as _json
    import urllib.request as _u
    if host in ("0.0.0.0", "", "::"):
        host = "127.0.0.1"
    try:
        with _u.urlopen("http://%s:%d/get_chains" % (host, port), timeout=timeout) as r:
            raw = _json.load(r)
    except Exception as e:  # noqa: BLE001 -- any failure -> caller falls back
        log_.info("broker chain discovery: %s:%d/get_chains unavailable (%s)", host, port, e)
        return None
    if not isinstance(raw, list) or not raw:
        return None
    out = []
    for c in raw:
        chain = c.get("chain")
        if not chain:
            continue
        out.append({"tag": c.get("constellation") or "G",
                    # RF band (L1/L2/L5), NOT the signal name -- it is the table's
                    # column, and GPS L5-Q and Galileo E5a-Q belong in the SAME one.
                    "band": c.get("rf_band") or (c.get("band") or "").replace("MHz", ""),
                    "col": c.get("short") or chain,
                    "name": c.get("label") or chain,
                    "sigid": c.get("sigid"),
                    # BOTH point at the chain id: the publisher serves merged detections and
                    # merged status on one port and filters on the path segment.
                    "combiner": chain,
                    "search": chain if c.get("has_search") else None,
                    "t_rec": float(c.get("t_rec") or 1e-3),
                    "carrier_mhz": float(c.get("carrier_hz") or 0.0) / 1e6,
                    "peel": False})
    log_.info("broker chain discovery: %d chain(s) from %s:%d -- %s",
              len(out), host, port, ", ".join(c["col"] for c in out))
    return out or None

def discover_signals(host, rest_port, timeout=3.0):
    """Reconstruct the UNIFIED_SIGNALS table from kotekan's running /config. Returns the list, or
    None on any failure (fetch/parse/empty) so main() keeps the static UNIFIED_SIGNALS fallback."""
    import urllib.request
    # --kotekan-host defaults to 0.0.0.0 (a *bind* address); reuse it as a connect target only
    # after mapping the wildcard to loopback.
    if host in ("0.0.0.0", "", "::"):
        host = "127.0.0.1"
    url = "http://%s:%d/config" % (host, rest_port)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            cfg = json.load(r)
    except Exception as e:  # noqa: BLE001 -- any failure -> fallback, never fatal
        log_.warning("signal discovery: /config fetch failed (%s); using static table", e)
        return None

    searches = {}          # stage name -> signal (acquired chains only)
    tracks = []            # (chain, signal_or_None, peel_bool) in config order

    def walk(node):
        if isinstance(node, dict):
            for name, v in node.items():
                if isinstance(v, dict):
                    kind = v.get("kotekan_stage")
                    if kind == "GnssChannelizedSearch":
                        searches[name] = v.get("signal")
                    elif kind == "cudaProcess":
                        for cm in v.get("commands", []):
                            if isinstance(cm, dict) and cm.get("name") == "cudaGnssTrack":
                                m = re.match(r"/([A-Za-z0-9_]+)_track/set_seeds$",
                                             cm.get("seed_endpoint") or "")
                                if m:
                                    tracks.append((m.group(1), cm.get("signal"),
                                                   bool(cm.get("peel"))))
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    try:
        walk(cfg)
    except Exception as e:  # noqa: BLE001
        log_.warning("signal discovery: config walk failed (%s); using static table", e)
        return None
    if not tracks:
        log_.warning("signal discovery: no cudaGnssTrack commands in /config; using static table")
        return None

    desc = _signal_descriptors()
    band_rank = {"L1": 0, "L2": 1, "L5": 2}
    tag_rank = {"G": 0, "L": 1, "E": 2, "C": 3, "R": 4}
    rows = []
    for chain, sig, peel in tracks:
        search_stage = chain + "_search"
        sigid = sig or searches.get(search_stage)
        if not sigid:
            log_.warning("signal discovery: chain %s has no resolvable signal; skipping", chain)
            continue
        carrier, period = desc.get(sigid, (None, None))
        tag, col, name = _display_for(sigid)
        rows.append({
            "tag": tag,
            # ★ The CONSTELLATION this signal's satellite belongs to, which is NOT always the
            # display tag. "L" (GPS L1C) is a synthetic tag invented so L1C keeps a distinct
            # SIGNAL series from C/A on the same PRN -- but the bird is a GPS bird. The client
            # keys satellite ROWS by sys and signal COLUMNS by tag, so a Block III satellite
            # gets ONE row carrying both CA and L1C, which is the entire premise of the unified
            # viewer ("one satellite per row"). Keying rows by tag split it into G20 and L20.
            "sys": _SYS_TAG.get(sigid.split("_", 1)[0], tag),
            "band": _band_of(carrier) if carrier else "L1",
            "col": col,
            "name": name,
            "sigid": sigid,
            "combiner": chain + "_combiner",
            # A derived data/pilot sibling has no search stage -> no SNR cell in the table.
            "search": search_stage if search_stage in searches else None,
            "t_rec": period if period else 1e-3,
            "peel": peel,
        })

    # Column order: band (L1<L2<L5), then constellation (G<L<E<C<R), then acquired-before-derived.
    rows.sort(key=lambda s: (band_rank.get(s["band"], 9),
                             tag_rank.get(s["tag"], 9),
                             0 if s["search"] else 1))
    log_.info("signal discovery: %d chains from /config -> %s",
              len(rows), ", ".join("%s/%s" % (r["band"], r["sigid"]) for r in rows))
    return rows


def _gps_signal_capability():
    """{sigid: [PRNs that BROADCAST it]} for the GPS signals in UNIFIED_SIGNALS.

    Lets the table distinguish "this satellite does not transmit that signal" from "it does and
    we are not seeing it" -- the difference between a blank row and a fault. A Block IIR sat
    showing 238 sigma on L1 C/A and nothing on L1C/L2C/L5 is CORRECT (IIR predates all three);
    the same pattern on a Block III sat is a real problem, and the table could not tell them
    apart.

    Two deliberate choices:
      * The block -> signal rule is IMPORTED from gps_beamtrack (_signal_block_filter), not
        restated here. Which block first carried each civil signal is exactly the kind of
        convention that rots when it exists in two places.
      * The PRN -> block map is parsed from the CACHED Celestrak TLE names
        (~/.cache/kotekan_gps/tle_gps-ops.txt, written by the brokers), so the viewer needs no
        network and no skyfield. Block is METADATA, not orbital state: a stale TLE gives stale
        ORBITS but the block of a launched satellite never changes, so an old cache is fine
        here even though it would not be for visibility.
    Returns {} on any failure -- the client then falls back to today's behaviour (everything
    unknown reads as "no detection"), which is the safe direction: never claim a satellite is
    incapable when we could not check.
    """
    try:
        import re as _re
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "gnss"))
        from gps_beamtrack import _signal_block_filter
        tle = os.path.expanduser("~/.cache/kotekan_gps/tle_gps-ops.txt")
        blocks = {}
        with open(tle) as fh:
            for ln in fh:
                m = _re.match(r"\s*GPS\s+B([A-Z]+)-?\d+\s*\(PRN\s*(\d+)\)", ln)
                if m:
                    blocks[int(m.group(2))] = m.group(1)
        if not blocks:
            return {}
        caps = {}
        for s in UNIFIED_SIGNALS:
            sid = s.get("sigid")
            if not sid:
                continue          # non-GPS: no capability model here -> client shows "--"
            f = _signal_block_filter(sid)
            caps[sid] = sorted(blocks) if f is None else sorted(p for p, b in blocks.items()
                                                                if f(b))
        return caps
    except Exception as e:
        logger.warning("GPS signal capability unavailable (%s); "
                       "table cannot distinguish not-transmitted from not-detected", e)
        return {}


# The physical front ends, for the unified page's RF selector (spectrum + waterfall + airspy
# controls switch between them). ws_port = the per-band livebeam_server the browser opens for that
# band's power stream; airspy = the ADC stage on the merged kotekan.
#
# STATIC FALLBACK ONLY: main() replaces this with discover_rf_bands() from the running /config, so
# each front end is LABELLED BY ITS ACTUAL airspy tuning (not a hardcoded GPS band name -- the l2c
# dongle now tunes 1207.14 for BeiDou B2b / Galileo E5b, not GPS L2C's 1227.6), and a suspended
# band (absent airspy stage) drops out of the selector. Used only when /config is unreachable.
UNIFIED_RF_BANDS = [
    {"band": "l1",  "ws_port": 8539, "airspy": "l1_airspy_in",  "label": "L1 · 1575.42 MHz"},
    {"band": "l2c", "ws_port": 8639, "airspy": "l2c_airspy_in", "label": "L2 · 1207.14 MHz"},
    {"band": "l5",  "ws_port": 8739, "airspy": "l5_airspy_in",  "label": "L5 · 1176.45 MHz"},
]
_RF_WS_PORT = {"l1": 8539, "l2c": 8639, "l5": 8739}


def discover_rf_bands(host, rest_port, timeout=3.0):
    """Build the RF-band selector from the airspyInput stages in kotekan's /config: label each
    front end by its ACTUAL airspy freq, and include only the front ends present (a suspended band
    has no airspy stage -> drops out). Returns None on failure so main() keeps the static list."""
    import urllib.request
    if host in ("0.0.0.0", "", "::"):
        host = "127.0.0.1"
    try:
        with urllib.request.urlopen("http://%s:%d/config" % (host, rest_port), timeout=timeout) as r:
            cfg = json.load(r)
    except Exception as e:  # noqa: BLE001
        log_.warning("rf-band discovery: /config fetch failed (%s); using static table", e)
        return None
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for name, v in node.items():
                if isinstance(v, dict) and v.get("kotekan_stage") == "airspyInput":
                    m = re.match(r"([a-z0-9]+)_airspy", name)
                    if m and v.get("freq") is not None:
                        try:
                            found[m.group(1)] = (name, float(v["freq"]))
                        except (TypeError, ValueError):
                            pass
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(cfg)
    if not found:
        return None
    order = {"l1": 0, "l2c": 1, "l5": 2}
    rows = []
    for band, (name, freq_mhz) in sorted(found.items(), key=lambda kv: order.get(kv[0], 9)):
        region = _band_of(freq_mhz * 1e6)  # airspy freq is in MHz
        rows.append({"band": band, "ws_port": _RF_WS_PORT.get(band, 8539), "airspy": name,
                     "label": "%s · %.2f MHz" % (region, freq_mhz)})
    log_.info("rf-band discovery: %s", ", ".join(r["label"] for r in rows))
    return rows


class WsPortResource(resource.Resource):
    """``GET /wsport`` -> ``{"ws_port": N, "band": .., "chains": [..]}`` for THIS server.

    The browser loads the page from one server's http_port but must open the WebSocket on the
    SAME server's ws_port -- and http->ws is not a fixed offset across bands (L1 8080/8539,
    L2C 8081/8639). The client used to hardcode 8539, so every non-L1 viewer connected to L1's
    WebSocket and received L1's kotekan rest_port (12048) -> cross-origin failures against the
    wrong kotekan. The client now fetches this (same-origin, so it hits the right server) and
    connects to the port it returns. It ALSO carries the band's constellation display defs
    (:data:`BAND_CHAINS`, filtered to --gps-constellations) so the sky legend names the right
    signal for this band.
    """

    isLeaf = True

    def __init__(self, ws_port, band="l1", consts="G,E,C", stage_prefix="", unified=False,
                 broker_chains=None):
        resource.Resource.__init__(self)
        self.ws_port = ws_port
        self.band = band
        self.unified = unified
        # UNIFIED: advertise the whole signal inventory + the RF-band selector. The client
        # keys satellites by tag+prn across every band and polls every combiner here. `chains`
        # is still emitted (the sky legend + per-constellation colours read it), one entry per
        # CONSTELLATION rather than per band.
        if unified:
            # BROKER-DISCOVERED INVENTORY WINS. In unified mode the client draws from
            # `signals` (a sub-table per constellation, a column per RF band), not from
            # `chains` -- so pointing the chain discovery at `chains` alone left the panel
            # rendering the static AIRSPY inventory while claiming to be unified. Feed the
            # live chains in here too, and derive the band selector from the bands they
            # actually occupy rather than from a list of front ends CHORD does not have.
            if broker_chains:
                self.signals = [dict(c) for c in broker_chains]
                _bands, _seen = [], set()
                for c in broker_chains:
                    if c["band"] not in _seen:
                        _seen.add(c["band"])
                        _bands.append({"band": c["band"], "ws_port": ws_port,
                                       "airspy": None,
                                       "label": ("%s \u00b7 %.2f MHz"
                                                 % (c["band"], c.get("carrier_mhz") or 0.0))})
                self.rf_bands = _bands
            else:
                self.signals = [dict(s) for s in UNIFIED_SIGNALS]
                self.rf_bands = [dict(b) for b in UNIFIED_RF_BANDS]
            # Attach the transmitting-PRN list per signal so the table can say
            # "not transmitted" instead of "not detected". Computed once here (cheap: one
            # cached-file parse), empty when unavailable -> client keeps today's behaviour.
            self.capability = _gps_signal_capability()
            _legend = [{"tag": "G", "name": "GPS",     "color": "#4d9de0"},
                           {"tag": "E", "name": "Galileo", "color": "#e8923c"},
                           {"tag": "C", "name": "BeiDou",  "color": "#d64550"},
                           # GLONASS (2026-08-04): violet, distinct from the G/E/C palette and
                           # from the green the synthetic "L" (GPS L1C) tag uses. PRNs here are
                           # orbital SLOT numbers, which is what the L3OC code index turned out
                           # to be -- so tag+prn keys GLONASS satellites the same way as the rest.
                           {"tag": "R", "name": "GLONASS", "color": "#9b6fd6"}]
            # Only the constellations actually being tracked: an empty sub-table for a
            # constellation this instrument cannot see reads as "nothing detected" rather
            # than "not configured", which is the same confusion the static table caused.
            if broker_chains:
                _live = {c["tag"] for c in broker_chains}
                _legend = [g for g in _legend if g["tag"] in _live]
            self.chains = _legend
            return
        tags = [t.strip() for t in consts.split(",") if t.strip()]
        table = BAND_CHAINS.get(band, BAND_CHAINS["l1"])
        # EXPLICIT stage names per chain: a merged multi-band instance prefixes its stages
        # (l1_gps_combiner) and the browser must poll those. The client falls back to its old
        # hardcoded spellings when absent (old server / new client and vice versa both work).
        base = {"G": ("gps_search", "gps_combiner"), "E": ("gal_search", "gal_combiner"),
                "C": ("bds_search", "bds_combiner"),
                "L": ("l1c_search", "l1c_combiner"),  # GPS L1C-P: 4th L1 chain, synthetic tag
                # GLONASS L3OC is a Shape-A primary: its chain is labelled `constellation: gps`
                # in gnss_node.yaml so it keeps the un-prefixed gps_* stage names the primary
                # broker needs, even though the SIGNAL is GLONASS. Hence gps_*, not glo_*.
                "R": ("gps_search", "gps_combiner")}
        if broker_chains:
            # WHAT IS RUNNING BEATS WHAT THE TABLE SAYS. One entry per live broker chain,
            # every constellation, one viewer instance, one port.
            self.chains = [dict(c) for c in broker_chains]
            return
        self.chains = []
        for c in table:
            if c["tag"] not in tags:
                continue
            c = dict(c)
            se, co = base[c["tag"]]
            c["search"], c["combiner"] = stage_prefix + se, stage_prefix + co
            self.chains.append(c)

    def render_GET(self, request):
        request.responseHeaders.setRawHeaders("Content-Type", [b"application/json"])
        reply = {"ws_port": self.ws_port, "band": self.band, "chains": self.chains}
        if self.unified:
            reply["unified"] = True
            reply["signals"] = self.signals
            reply["rf_bands"] = self.rf_bands
            reply["capability"] = self.capability  # {sigid: [transmitting PRNs]}
        return json.dumps(reply).encode("utf-8")


class GpsSkyResource(resource.Resource):
    """``GET /gps_sky`` -> predicted GPS sky positions for the viewer's sky plot.

    Reply::

        {"ok": bool, "computing": bool, "lat": .., "lon": .., "alt": ..,
         "mask_deg": .., "sats": [{"prn": N, "az": deg, "el": deg}, ...]}

    Azimuth is degrees clockwise from true North, elevation degrees above the
    horizon; only sats above ``mask_deg`` are returned. The kotekan REST stages
    supply the locks / |A| / SNR; this endpoint supplies *where* each sat is,
    which needs skyfield orbit propagation (not available in the DSP).

    Positions are recomputed in a worker thread at most every
    :data:`SKY_REFRESH_S` and ``render_GET`` only serves the cached result, so
    the reactor never blocks on skyfield or the TLE fetch. Degrades to
    ``ok=False`` (no skyfield / no observer site / TLE network down) and the
    panel falls back to the REST-only locked list.
    """

    isLeaf = True

    def __init__(self, lat, lon, alt, mask_deg, consts="G,E,C", glonass_sigids=None):
        resource.Resource.__init__(self)
        self.lat, self.lon, self.alt, self.mask_deg = lat, lon, alt, mask_deg
        # Only serve the constellations this band actually tracks: L1 tri-band = G,E,C (all share
        # 1575.42); L2C / L5 are GPS-only, so showing Galileo/BeiDou there is misleading (they are
        # not even in this band). Filter the class table to the requested tags, order preserved.
        want = [t.strip() for t in consts.split(",") if t.strip()]
        self.CONSTELLATIONS = tuple(c for c in type(self).CONSTELLATIONS if c[0] in want)
        # The GLONASS signals actually running (sigids, e.g. GLO_L2OF / GLO_L3OC_P), so the R sky
        # layer can filter to what ANY of them can track rather than to one signal's capability.
        # None/empty -> no per-signal filter (show every GLONASS sat).
        self.glonass_sigids = list(glonass_sigids or [])
        self._cache = {"ok": False, "sats": []}  # last good (or empty) result
        self._sats = None                        # cached {prn: EarthSatellite} (TLE fallback)
        self._eph = None                         # cached parsed BRDC {(sys,prn):[recs]}
        self._eph_t = None                       # monotonic s of last BRDC parse
        self._l1c_capable = None                 # cached set of L1C-capable PRNs (GPS III)
        self._l1c_capable_t = None               # monotonic s of last capability fetch
        self._glo_capable = None                 # cached set of GLONASS slots any running R
        self._glo_capable_t = None               # signal can track (union); None = no filter
        self._last_compute = None                # monotonic s of last attempt
        self._refreshing = False

    # Viewer tag -> ephemeris system letter. 'L' (GPS L1C) is the same GPS birds as 'G',
    # so it reuses the GPS positions; there is no separate 'L' system in the ephemeris.
    _TAG_SYS = {"G": "G", "E": "E", "C": "C", "L": "G"}

    def _need_refresh(self):
        if self.lat is None or self.lon is None:
            return False
        if self._last_compute is None:
            return True
        return (time.monotonic() - self._last_compute) >= SKY_REFRESH_S

    # Constellations served on the sky plot: tag + Celestrak TLE group. GPS, Galileo and
    # BeiDou B1C all share the 1575.42 tune, so the tri-constellation configs track all
    # three; a group whose TLE fetch fails is skipped (the others still render).
    CONSTELLATIONS = (
        ("G", None),        # None -> gps_beamtrack.DEFAULT_TLE_URL (gps-ops)
        ("E", "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle"),
        ("C", "https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle"),
        # ⚠️ NO "L" LAYER. GPS L1C is a synthetic SIGNAL tag on GPS birds, not a constellation:
        # drawing it here put a second marker on top of every L1C-capable GPS satellite, at
        # identical az/el. (It was masked for a while because --unified hardcoded the sky to
        # "G,E,C"; deriving that list correctly is what exposed the duplicates.) The sky draws
        # one marker per SATELLITE; L1C's presence shows up as a column in the detections table.
        ("R", "https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=tle"),
    )

    # Constellations gnss_ephemeris can actually propagate. Its RINEX parser filters on
    # sysc in "GEC", and GLONASS broadcasts a position/velocity/acceleration state vector
    # integrated by Runge-Kutta rather than a Keplerian element set, so there is nothing for
    # sat_pos_clk to work with. 'R' therefore comes from TLE even when BRDC is healthy --
    # see _compute, which merges the two sources rather than choosing between them.
    _BRDC_TAGS = frozenset(("G", "E", "C", "L"))

    def _compute(self):
        """Runs in a worker thread. PRIMARY: broadcast-ephemeris (BRDC) geometry via predict_all
        -- the SAME source the tracker uses, so the sky plot agrees with what we actually lock,
        and it is PRN-intrinsic (no fragile TLE name->PRN mapping, which had mis-placed BeiDou
        MEOs C31/C39 below the horizon while we were tracking them at +17/+27 deg). FALLBACK:
        Celestrak TLE + skyfield, used only when the BRDC is unavailable (offline / parse fail).

        ★ The two sources are MERGED, not chosen between. GLONASS has no BRDC path at all (see
        _BRDC_TAGS), so the old "if BRDC returned anything, use it and stop" would have taken the
        BRDC result for GPS/Galileo/BeiDou and silently dropped GLONASS from the sky plot
        entirely -- satellites we are actively locking, absent from the picture, with nothing
        anywhere saying why. Same shape as the broker's zero-satellite almanac trap: a source
        succeeding for SOME constellations is not the same as it covering the ones you asked for.
        """
        want = [c[0] for c in self.CONSTELLATIONS]
        brdc_want = [t for t in want if t in self._BRDC_TAGS]
        tle_want = [t for t in want if t not in self._BRDC_TAGS]
        sats = self._brdc_skypos(brdc_want) if brdc_want else None
        if sats is None:
            # BRDC unavailable -> everything falls back to TLE (the original behaviour).
            return {"ok": True, "sats": self._tle_skypos(want), "src": "tle"}
        if tle_want:
            sats = sats + self._tle_skypos(tle_want)
            return {"ok": True, "sats": sats, "src": "brdc+tle"}
        return {"ok": True, "sats": sats, "src": "brdc"}

    def _glonass_capable_set(self):
        """Cached set of GLONASS SLOTS trackable by ANY GLONASS signal we are running -- the
        UNION of each running R signal's capability. This is the R analogue of _l1c_capable_set,
        but it MUST be a union, not one signal's set, because the R sky layer aggregates every
        GLONASS signal onto one marker per satellite.

        ★ WHY THE UNION MATTERS (the bug this fixes, 2026-08-04): L3OC is GLONASS-K only (7 sats)
        while L2OF is on EVERY satellite (~26 with a known channel). Filtering the whole R sky to
        L3OC capability hid every non-K satellite -- so with both signals running, only the K sats
        showed and L2OF's ~19 other trackable satellites vanished from the plot. A satellite that
        carries L2OF but not L3OC is not a silent bird; it is trackable, and belongs in the sky.

        The union conditions on what is ACTUALLY running (self.glonass_sigids from the live
        /config): L3OC alone -> K only; L2OF present -> all known-k slots. None => no filter (no
        GLONASS signals, or a capability lookup that covered ~everything and so carried no info)."""
        if (self._glo_capable_t is not None
                and (time.monotonic() - self._glo_capable_t) < 3600.0):
            return self._glo_capable
        union = set()
        try:
            import gps_beamtrack as bt
            for sigid in self.glonass_sigids:
                # ⚠️ Use the CLEAN primitive per signal, not signal_capable_prns: for an FDMA
                # signal that function falls back to the WHOLE slot set when the frequency plan
                # is unavailable (its block filter is None -> "all"), which is indistinguishable
                # by size from the legitimate ~26-slot known-k set. glonass_fdma_trackable() and
                # the K-marker path both return empty on failure instead, never a spurious full
                # set -- so the union stays trustworthy without a fragile size threshold.
                if "OF" in sigid.upper():
                    union |= bt.glonass_fdma_trackable()          # known-k, empty on failure
                else:
                    union |= set(bt.signal_capable_prns(sigid, bt.GLONASS_TLE_URL))  # K set
        except Exception as e:
            log_.warning("gps_sky: GLONASS capability set unavailable (%s); showing all GLONASS "
                         "under R", e)
            union = set()
        # Empty => the lookup failed (or no GLONASS signals): stand the filter down and show all,
        # never hide a real satellite behind a failed capability check.
        self._glo_capable = union or None
        self._glo_capable_t = time.monotonic()
        return self._glo_capable

    def _l1c_capable_set(self):
        """Cached set of PRNs whose GPS block actually broadcasts L1C (GPS III), from the live
        Celestrak block names -- so the 'L' (L1C) sky layer shows only birds that can transmit it,
        not every visible GPS sat (older blocks would sit silent, looking like failed L1C locks).
        None on failure / unknown => no filter (show all GPS under L, the prior behavior)."""
        if (self._l1c_capable_t is not None
                and (time.monotonic() - self._l1c_capable_t) < 3600.0):
            return self._l1c_capable
        try:
            import gps_beamtrack as bt
            caps = bt.signal_capable_prns("GPS_L1C_P")
            # signal_capable_prns returns ALL PRNs when the signal is unknown / on every sat;
            # a near-full set carries no information, so treat it as "don't filter".
            self._l1c_capable = set(caps) if caps and len(caps) < 32 else None
        except Exception as e:
            log_.warning("gps_sky: L1C capability set unavailable (%s); showing all GPS under L", e)
            self._l1c_capable = None
        self._l1c_capable_t = time.monotonic()
        return self._l1c_capable

    def _brdc_skypos(self, want):
        """Sky positions from the broadcast ephemeris, or None if the BRDC is unavailable.

        The parsed ephemeris is cached ~15 min (fetch_brdc is itself cache-gated), but positions
        still update every refresh because predict_all is re-evaluated at the current time."""
        try:
            import gnss_ephemeris as ge
            if (self._eph is None or self._eph_t is None
                    or (time.monotonic() - self._eph_t) > 900.0):
                raw = ge.parse_rinex_nav(ge.fetch_brdc(datetime.now(timezone.utc)))
                # Drop BeiDou GEO (i<5 deg): predict_all's plain Kepler does NOT apply the ICD
                # -5 deg GEO rotation, so their positions are untrustworthy (a mis-propagated GEO
                # could land above the horizon as a phantom). IGSO/MEO propagate correctly.
                self._eph = {k: r for k, r in raw.items()
                             if not (k[0] == "C" and r
                                     and math.degrees(r[-1]["i0"]) < 5.0)}
                self._eph_t = time.monotonic()
            pred = ge.predict_all(self._eph, self.lat, self.lon, self.alt,
                                  datetime.now(timezone.utc), mask_deg=SKY_REPORT_FLOOR_DEG)
        except Exception as e:
            log_.warning("gps_sky: BRDC geometry unavailable (%s); falling back to TLE", e)
            return None
        if not pred:
            return None
        cap_L = self._l1c_capable_set()
        sats = []
        for tag in want:
            sysc = self._TAG_SYS.get(tag)
            if not sysc:
                continue
            for (s, p), v in sorted(pred.items()):
                if s != sysc:
                    continue
                if tag == "L" and cap_L is not None and p not in cap_L:
                    continue  # 'L' = L1C: only GPS III birds actually transmit it
                sats.append({"prn": p, "const": tag,
                             "az": round(v["az"], 2), "el": round(v["el"], 2)})
        return sats

    def _tle_skypos(self, want=None):
        """Sky positions from Celestrak TLE + skyfield propagation, for the tags in `want`
        (default: all configured). Used as the whole-sky fallback when BRDC is unavailable, and
        as the ONLY source for constellations BRDC cannot propagate (GLONASS)."""
        from gps_beamtrack import (DEFAULT_TLE_URL, load_gps_satellites,
                                   predict_skypos)
        if self._sats is None:
            self._sats = {}
            for tag, url in self.CONSTELLATIONS:
                try:
                    self._sats[tag] = load_gps_satellites(url or DEFAULT_TLE_URL)
                except Exception as e:
                    log_.warning("gps_sky: %s TLEs unavailable: %s", tag, e)
        cap_L = self._l1c_capable_set()
        cap_R = self._glonass_capable_set()
        sats = []
        for tag, by_prn in self._sats.items():
            if want is not None and tag not in want:
                continue
            pos = predict_skypos(self.lat, self.lon, self.alt, _sats=by_prn)
            sats += [{"prn": p, "const": tag, "az": round(az, 2), "el": round(el, 2)}
                     for p, (az, el) in sorted(pos.items()) if el >= self.mask_deg
                     and not (tag == "L" and cap_L is not None and p not in cap_L)
                     and not (tag == "R" and cap_R is not None and p not in cap_R)]
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
            "mask_deg": self.mask_deg,
            "sats": self._cache.get("sats", []),
        }
        if self._cache.get("error"):
            out["error"] = self._cache["error"]
        return json.dumps(out).encode("utf-8")


def _pvt_measurements(globs, max_age_s, t_now):
    """Freshest per-(sys, prn, freq-band) code observation across the obs logs, for the PVT
    self-survey. One row per satellite-frequency (pilot/data of the same sat dedupe to one, same
    geometry). Residuals are wrapped clock-dominated ranges -> de-wrapped per group to the group
    median so the common clock is a single free parameter and no sat straddles a code-period edge.
    Returns a list of {group, az, el, resid_m}."""
    import glob as _glob
    import json as _json
    C = 299792458.0

    def _freq(band):
        b = str(band)
        if "L2" in b:
            return "L2"
        if "L5" in b or "E5" in b or "B2A" in b or "B2a" in b:
            return "L5"
        return "L1"

    latest = {}   # (sys, prn, freq) -> (t, group, az, el, resid_m, L_m)
    for pat in globs:
        for path in _glob.glob(pat):
            try:
                size = os.path.getsize(path)
                with open(path) as f:
                    f.seek(max(0, size - 512 * 1024))   # tail: freshest rows only
                    f.readline()
                    for line in f:
                        try:
                            d = _json.loads(line)
                        except Exception:
                            continue
                        t = d.get("t")
                        sysid, prn = d.get("sys"), d.get("prn")
                        res, az, el = d.get("code_resid_m"), d.get("az"), d.get("el")
                        if (t is None or sysid is None or prn is None or res is None
                                or az is None or el is None):
                            continue
                        if t_now - t > max_age_s:
                            continue
                        fr = _freq(d.get("band"))
                        key = (sysid, prn, fr)
                        if key in latest and latest[key][0] >= t:
                            continue
                        cl, ch = d.get("code_len"), d.get("chip_rate_hz")
                        L = (cl / ch * C) if (cl and ch) else None
                        latest[key] = (t, "%s-%s" % (sysid, fr), az, el, float(res), L)
            except Exception:
                continue

    # de-wrap each group's residuals to the group median (handles the clock straddling a
    # code-period boundary; L is the code period in metres). Keep the de-wrapped per-(sys,prn,freq)
    # residual so the dual-frequency iono-free combination can be formed from CLEAN ranges.
    import statistics as _st
    by_group = {}
    for (sysid, prn, fr), v in latest.items():
        by_group.setdefault(v[1], []).append((sysid, prn, fr, v))
    out = []
    dw = {}   # (sys, prn, freq) -> (az, el, resid_dewrapped)
    for group, rows in by_group.items():
        med = _st.median([r[3][4] for r in rows])
        for sysid, prn, fr, v in rows:
            _t0, g, az, el, res, L = v
            if L:
                res = res - L * round((res - med) / L)   # unwrap to within +-L/2 of the median
            out.append({"group": g, "az": az, "el": el, "resid_m": res})
            dw[(sysid, prn, fr)] = (az, el, res)

    # DUAL-FREQUENCY iono-free combination: for each satellite seen on two bands, remove the
    # first-order ionosphere. rho_IF = (f1^2 rho1 - f2^2 rho2)/(f1^2 - f2^2) -- the ~1/f^2 iono
    # cancels and any per-band clock/bias folds into a per-sat-independent constant the "-IF"
    # group clock absorbs. Prefer the widest split (L1+L5) for the strongest iono removal. The
    # solver puts these in their own combined_if solution (few-metre) beside the single-freq one.
    FREQ_HZ = {"L1": 1575.42e6, "L2": 1227.60e6, "L5": 1176.45e6}
    bysat = {}
    for (sysid, prn, fr), (az, el, res) in dw.items():
        bysat.setdefault((sysid, prn), {})[fr] = (az, el, res)
    for (sysid, prn), bands in bysat.items():
        pair = next((p for p in (("L1", "L5"), ("L1", "L2"), ("L2", "L5"))
                     if p[0] in bands and p[1] in bands), None)
        if pair is None:
            continue
        f1, f2 = FREQ_HZ[pair[0]], FREQ_HZ[pair[1]]
        (az1, el1, r1), (_a2, _e2, r2) = bands[pair[0]], bands[pair[1]]
        rif = (f1 * f1 * r1 - f2 * f2 * r2) / (f1 * f1 - f2 * f2)
        out.append({"group": "%s-IF" % sysid, "az": az1, "el": el1, "resid_m": rif})
    return out


class DecodeHealthResource(resource.Resource):
    """``GET /decode_health`` -> merged on-node nav-decode health (from the brokers' JSON files)
    PLUS a PVT self-survey computed from the obs-log code residuals (:mod:`gnss_pvt`). Stale
    broker files / obs rows are dropped, so a decoder or the position solve going dark shows up
    as the row/section disappearing, not a frozen last value. PVT is cached (recomputed every
    `pvt_ttl_s`) since it tail-reads several obs logs and solves."""

    isLeaf = True

    def __init__(self, dirpath, max_age_s=120.0, lat=None, lon=None, alt=None,
                 obs_globs=None, pvt_ttl_s=15.0, pvt_max_age_s=30.0):
        resource.Resource.__init__(self)
        self.dirpath = dirpath
        self.max_age_s = max_age_s
        self.lat, self.lon, self.alt = lat, lon, (alt or 0.0)
        self.obs_globs = obs_globs or []
        self.pvt_ttl_s = pvt_ttl_s
        # PVT wants NEAR-SIMULTANEOUS observations: the receiver clock (a free per-group param) is
        # assumed common across a group's sats, and a disciplined-oscillator drift of ~1e-9 over
        # 120 s is already ~36 m. 30 s keeps that error small while still catching ~1 row/sat.
        self.pvt_max_age_s = pvt_max_age_s
        self._pvt_cache = (0.0, None)

    def _pvt(self, t_now):
        if self.lat is None or self.lon is None or not self.obs_globs:
            return None
        if t_now - self._pvt_cache[0] < self.pvt_ttl_s:
            return self._pvt_cache[1]
        try:
            import gnss_pvt as _pvt
            meas = _pvt_measurements(self.obs_globs, self.pvt_max_age_s, t_now)
            res = _pvt.solve(meas, self.lat, self.lon, self.alt) if meas else None
            if res is not None:
                res["apriori"] = {"lat": self.lat, "lon": self.lon, "alt": self.alt}
                res["n_meas"] = len(meas)
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
        "--gps",
        action="store_true",
        help="Enable the GPS sky/locks panel. Auto-enabled when LAT/LON "
        "are known (CLI --lat/--lon or env LAT/LON).",
    )
    gg.add_argument("--lat", type=float, default=None,
                    help="Observer latitude (deg) for sky positions; "
                    "default env LAT.")
    gg.add_argument("--lon", type=float, default=None,
                    help="Observer longitude (deg) for sky positions; "
                    "default env LON.")
    gg.add_argument("--alt", type=float, default=None,
                    help="Observer altitude (m); default env ALT, else 0.")
    gg.add_argument("--gps-mask-deg", type=float, default=5.0,
                    help="Elevation mask (deg) for the sky plot (default 5).")
    gg.add_argument("--gps-search-stage", default="gps_search",
                    help="kotekan GnssChannelizedSearch stage name (get_detections); "
                    "default 'gps_search' (the browser alias-resolves it to the bare "
                    "'search' on configs that still use that spelling).")
    gg.add_argument("--gps-combiner-stage", default="gps_combiner",
                    help="kotekan GnssCoherentCombiner stage name (get_status); "
                    "default 'gps_combiner' (alias-resolved to 'combiner' as above).")
    gg.add_argument("--decode-health-dir",
                    default=os.path.join(os.path.expanduser("~"), ".cache",
                                         "kotekan_gps", "decode_health"),
                    help="dir the brokers write nav-decode health JSON to (matches run_live's "
                         "DECODE_DIR); served merged at /decode_health for the viewer panel")
    gg.add_argument("--pvt-obs-globs",
                    default="/tmp/gpswipe/obs_*.jsonl,/tmp/gps_l2c_gpu/obs_*.jsonl,"
                            "/tmp/gps_l5_gpu/obs_*.jsonl",
                    help="comma-separated globs of obs-log jsonl (code_resid_m + az/el) the PVT "
                         "self-survey reads; served under /decode_health's pvt field")
    gg.add_argument("--gps-constellations", default="G,E,C",
                    help="constellations to show on the sky plot (G=GPS, E=Galileo, C=BeiDou). "
                         "L1 tri-band = G,E,C; L2C = G; L5 = G,E,C (E5a/B2a).")
    gg.add_argument("--band", default="l1", choices=["l1", "l2c", "l5"],
                    help="which frequency band this viewer serves -> the per-constellation "
                    "legend names + record periods (t_rec for C/N0). Delivered to the client "
                    "via /wsport so the sky legend reads the RIGHT signal (e.g. L5 shows "
                    "'GPS L5 / Galileo E5a / BeiDou B2a', not the L1 defaults). Default l1.")
    gg.add_argument("--stage-prefix", default="",
                    help="Prefix on every kotekan GNSS stage name (merged multi-band instance: "
                         "'l1_'/'l2c_'/'l5_'; see gen_3band_config.py). Applied to the "
                         "search/combiner/airspy stage args AND carried per-chain in /wsport.")
    gg.add_argument("--unified", action="store_true",
                    help="UNIFIED viewer: one page, one row per satellite, every signal it "
                         "carries (L1 C/A, L1C, L2C-CM, L2C-CL, L5, E1C/E5a, B1C/B2a) shown "
                         "side by side, polling ALL band-prefixed combiners on the merged "
                         "kotekan. /wsport carries the full signal inventory (UNIFIED_SIGNALS) "
                         "and the RF-band selector list (UNIFIED_RF_BANDS). Without it, the "
                         "viewer serves its single --band as before.")
    gg.add_argument("--gps-airspy-stage", default="airspy_in",
                    help="kotekan airspy stage name for the ADC noise readout "
                    "(adcstat); default 'airspy_in'.")
    gg.add_argument("--no-power-stream", action="store_true",
                    help="Run GPS-only: don't wait for / serve a kotekan power "
                    "stream (no waterfall). The lean live config uses this so the "
                    "GPS panel works without a full-rate diagnostic PFB. Implies "
                    "--gps.")
    args = ap.parse_args()
    if args.stage_prefix:
        args.gps_search_stage = args.stage_prefix + args.gps_search_stage
        args.gps_combiner_stage = args.stage_prefix + args.gps_combiner_stage
        args.gps_airspy_stage = args.stage_prefix + args.gps_airspy_stage

    # Resolve the observer site from CLI, falling back to the env that
    # run_live.sh already exports for the broker's almanac assist. The site only
    # *positions* the sky plot; the panel is gated on the explicit --gps flag so
    # a non-GPS viewer launched from a shell that happens to have LAT/LON in its
    # environment never sprouts a stray GPS card. (live_l1.yaml passes --no-power-stream,
    # which implies --gps.)
    if args.lat is None and os.environ.get("LAT"):
        args.lat = float(os.environ["LAT"])
    if args.lon is None and os.environ.get("LON"):
        args.lon = float(os.environ["LON"])
    if args.alt is None:
        args.alt = float(os.environ.get("ALT", 0.0))
    # GPS-only mode is meaningless without the panel, so it implies --gps.
    # --unified is a GPS-first page across all constellations: force the panel on and the
    # sky to the full sky (the RF band selector owns which spectrum shows).
    if args.unified:
        args.gps = True
        # DERIVED, not a literal: "the full sky" is whatever constellations GpsSkyResource
        # knows how to place. This was hardcoded "G,E,C" and had already gone stale twice over
        # -- it silently dropped the synthetic "L" (GPS L1C) layer, and would have dropped
        # GLONASS the day it came up, on a page whose whole point is showing every signal at
        # once. Adding a constellation to CONSTELLATIONS is now enough.
        args.gps_constellations = ",".join(t for t, _ in GpsSkyResource.CONSTELLATIONS)
    args._gps_enabled = bool(args.gps or args.no_power_stream or args.unified)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.startLogging(sys.stdout, setStdout=False)

    # SWAP-PROOF SIGNAL LIST: rebuild UNIFIED_SIGNALS from kotekan's running /config so a
    # band/constellation swap in gnss_node.yaml needs no viewer edit. Falls back to the static
    # literal if /config is unreachable (kotekan not up yet, wrong port, ...). Reassigning the
    # module global is enough: _gps_signal_capability() and WsPortResource read it at call time.
    # Done after logging is configured so the discovery summary lands in the viewer log.
    if args.unified:
        discovered = discover_signals(args.kotekan_host, args.kotekan_rest_port)
        if discovered:
            global UNIFIED_SIGNALS
            UNIFIED_SIGNALS = discovered
        # Backfill "sys" (the satellite's real constellation) on whichever list we ended up
        # with, so the static fallback behaves identically to the discovered one. Derived from
        # the signal id's prefix where there is one, else the display tag.
        for _s in UNIFIED_SIGNALS:
            _s.setdefault("sys", _SYS_TAG.get((_s.get("sigid") or "").split("_", 1)[0],
                                              _s.get("tag")))
        rf = discover_rf_bands(args.kotekan_host, args.kotekan_rest_port)
        if rf:
            global UNIFIED_RF_BANDS
            UNIFIED_RF_BANDS = rf

    static_dir = os.path.dirname(os.path.abspath(__file__))

    # GPS-only mode skips the kotekan power stream entirely (no blocking accept,
    # no waterfall) and serves just the GPS panel + /gps_sky. Otherwise stand up
    # the power stream as usual (blocks on accept until kotekan connects).
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
    # ONE VIEWER, EVERY CONSTELLATION (task #27 M6). If the rest port is a BROKER publisher
    # it can name its own chains; if it is a kotekan node it cannot, and we fall through to
    # the band/constellation table exactly as before. Trying and falling back beats a flag,
    # because the failure it replaces was SILENT: a viewer pointed at a broker port used to
    # poll airspy-era stage names and report the result as a CORS error.
    _bchains = discover_broker_chains(args.kotekan_host, args.kotekan_rest_port)
    root.putChild(b"wsport", WsPortResource(args.ws_port, args.band, args.gps_constellations,
                                            args.stage_prefix, unified=args.unified,
                                            broker_chains=_bchains))

    # GPS sky positions (az/el) for the live-status panel. gps_beamtrack lives
    # one dir up (python/scripts); add it to the path so the worker thread can
    # import it. Only stood up when the panel is enabled, so a plain radio-
    # astronomy run doesn't pull in skyfield / fetch TLEs.
    if args._gps_enabled:
        # python/scripts (parent of js_viewer) AND python/scripts/gnss -- the GNSS helpers
        # (gps_beamtrack et al.) moved into the gnss/ subpackage on 2026-07-14 and this
        # bare-name import is how the sky panel finds them. Missing it makes /gps_sky return
        # {"sats": []} with an error nobody looks at, and the viewer just quietly loses the sky.
        _scripts = os.path.dirname(static_dir)
        sys.path.insert(0, _scripts)
        sys.path.insert(0, os.path.join(_scripts, "gnss"))
        # GLONASS sky filter is the UNION of the running R signals' capabilities (L3OC = K
        # only, L2OF = all known-k), so a satellite trackable by ANY of them shows. Derived from
        # the same discovered inventory the table uses, so a band swap needs no viewer edit.
        _glo_sigids = sorted({s.get("sigid") for s in UNIFIED_SIGNALS
                              if (s.get("sigid") or "").startswith("GLO_")} - {None})
        root.putChild(b"gps_sky",
                      GpsSkyResource(args.lat, args.lon, args.alt, args.gps_mask_deg,
                                     args.gps_constellations, glonass_sigids=_glo_sigids))
        root.putChild(b"decode_health",
                      DecodeHealthResource(args.decode_health_dir, lat=args.lat, lon=args.lon,
                                           alt=args.alt, obs_globs=args.pvt_obs_globs.split(",")))
        log_.info("GPS panel enabled (site lat=%s lon=%s alt=%s); decode-health dir %s",
                  args.lat, args.lon, args.alt, args.decode_health_dir)

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
