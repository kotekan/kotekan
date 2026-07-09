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

import numpy as np
from autobahn.twisted.websocket import WebSocketServerFactory, WebSocketServerProtocol
from twisted.internet import reactor
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


# Mode is decided by the number of visibility streams in the kotekan handshake.
# ARO's computeDualpolPower emits nvis=2 (XX/YY) -> "dualpol".
MODE_BY_NVIS = {1: "autocorr", 2: "dualpol", 4: "crosscorr"}
VIS_LABELS_BY_MODE = {
    "autocorr": ["I"],
    "crosscorr": ["AA", "BB", "Re{AB*}", "Im{AB*}"],
    # dualpol labels are derived from the handshake Stokes ids (see vis_labels).
}

# Stokes / polarisation id -> label, matching networkPowerStream's info-block
# convention (``-5 - e``): -5=XX, -6=YY, ... plus the positive Stokes I/Q/U/V.
STOKES_LOOKUP = {
    -8: "YX", -7: "XY", -6: "YY", -5: "XX",
    -4: "LR", -3: "RL", -2: "LL", -1: "RR",
    1: "I", 2: "Q", 3: "U", 4: "V",
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
        power_dtype="float32",
        sum_freq=1,
    ):
        self.host = host
        self.port = port
        self.on_frame = on_frame  # callable, fired once per emitted (averaged) frame
        self.connected = False
        # Frequency downsampling: sum this many adjacent channels into one
        # before sending to the browser (1 = no downsampling). Reduces the
        # per-frame WS payload and the number of pixels the client renders.
        self.sum_freq = max(1, int(sum_freq))
        # Browser-facing integration window; integrations drained inside one
        # window are averaged into a single emitted frame.
        self.emit_interval_s = emit_interval_s
        # Payload interpretation of the power spectrum bytes. The airspy path
        # (simpleAutocorr) sends float32; ARO's computeDualpolPower sends
        # uint32 *integer* power sums. The handshake's sample_type can't
        # distinguish them (both hardcode 4), so it's set explicitly. uint32
        # payloads are normalised by samples_summed to a per-sample mean power,
        # matching the historical ARO monitor's ``d/n`` (and giving a stable
        # dB scale independent of the integration length).
        self._np_dtype = np.uint32 if power_dtype == "uint32" else np.float32
        self._normalize_counts = power_dtype == "uint32"
        # Populated by start() once kotekan's handshake is parsed.
        self.frame_nvis = 0
        self.frame_nfreq = 0
        # True when the band is inverted (channel 0 on top, sample_bw < 0),
        # detected from a negative raw_cadence at handshake.
        self.band_inverted = False

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

        # ARO runs with sample_bw < 0 (channel 0 on top), which makes kotekan
        # report a *negative* raw_cadence. That's a frequency-ordering signal,
        # not a negative duration -- the actual band direction is already
        # carried by the descending info-block edges above. Use the magnitude
        # for every timing computation (period, integrate_n, sample_time);
        # left signed it would collapse integrate_n to 1 and run the waterfall
        # time axis backwards.
        self.band_inverted = self.frame_raw_cad < 0
        self.frame_raw_cad = abs(self.frame_raw_cad)

        # Display normalisation: what the browser receives is always
        # ascending in frequency and (optionally) downsampled. The instrument
        # may deliver either ordering -- ARO's PFB in the 2nd Nyquist zone
        # gives a descending band (channel 0 = top). We detect the ordering
        # from the actual bin edges and flip on the way out so every
        # front-end component (axis, slider, spectrum) only ever sees
        # ascending data. See _display_freqs_mhz / display_spectrum.
        self._flip_freq = float(self.frame_freqs[0].mean()) > float(
            self.frame_freqs[-1].mean()
        )
        # Largest channel count divisible by sum_freq (drop a partial tail).
        self._usable_nfreq = self.frame_nfreq - (self.frame_nfreq % self.sum_freq)
        self.send_nfreq = self._usable_nfreq // self.sum_freq
        # Per-output-bin centre frequencies (MHz), downsampled then flipped to
        # ascending -- the FREQLIST the browser plots against.
        centres = self.frame_freqs[: self._usable_nfreq].mean(axis=1) / 1e6
        centres = centres.reshape(self.send_nfreq, self.sum_freq).mean(axis=1)
        if self._flip_freq:
            centres = centres[::-1]
        self.display_freqs_mhz = np.ascontiguousarray(centres, dtype=np.float32)
        if self.sum_freq > 1 or self._flip_freq:
            log_.info(
                "Display: nfreq %d -> %d (sum_freq=%d), order=%s",
                self.frame_nfreq, self.send_nfreq, self.sum_freq,
                "flipped->ascending" if self._flip_freq else "ascending",
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
        """``'autocorr'`` nvis=1, ``'dualpol'`` nvis=2, ``'crosscorr'`` nvis=4."""
        return MODE_BY_NVIS.get(self.frame_nvis, "unknown")

    @property
    def vis_labels(self):
        """Per-vis labels for the configured mode.

        For dualpol the labels come from the handshake Stokes ids (XX/YY, or
        RR/LL, etc.) so the viewer names the pols correctly for whatever the
        instrument actually sent, rather than a hardcoded pair.
        """
        if self.mode == "dualpol":
            elems = getattr(self, "frame_elems", None)
            if elems is not None:
                return [STOKES_LOOKUP.get(int(e), f"pol{i}") for i, e in enumerate(elems)]
        return VIS_LABELS_BY_MODE.get(
            self.mode, [f"vis{i}" for i in range(self.frame_nvis)]
        )

    def display_spectrum(self):
        """The latest emitted frame as the browser should see it:
        ``(nvis, send_nfreq)`` float32, adjacent channels summed by
        ``sum_freq`` and flipped to ascending frequency if the instrument
        band was descending. This is the single place the display-order and
        downsampling conventions are applied."""
        s = self.outbuf[:, : self._usable_nfreq]
        if self.sum_freq > 1:
            s = s.reshape(self.frame_nvis, self.send_nfreq, self.sum_freq).sum(axis=2)
        if self._flip_freq:
            s = s[:, ::-1]
        return np.ascontiguousarray(s, dtype=np.float32)

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
            vals = np.frombuffer(d[self.subframe_hdr_len :], dtype=self._np_dtype)
            vals = vals.astype(np.float32)
            if self._normalize_counts and self.samples_summed:
                vals = vals / self.samples_summed
            self.databuf[elem_idx, :] = vals

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
    """
    mode = kotekan.mode

    # Pipeline-shaped defaults; the user can override the lists / values via CLI.
    default_airspy_stages = {
        "autocorr": ["airspy_input"],
        "crosscorr": ["airspy_inputA", "airspy_inputB"],
    }
    # dualpol shows raw per-sample mean power in dB (uint32 counts normalised
    # by samples_summed). Measured live on the ARO 46m iceboard stream the
    # band sits ~3-10 dB with RFI spiking to ~60 dB; [0, 20] shows the
    # bandpass with headroom and lets RFI saturate. Exact level depends on the
    # instrument's requantisation, so this is just a first-look bracket the
    # user re-centres with the Color slider (and baseline subtraction).
    default_color_range = {"autocorr": [-20, 20], "crosscorr": [-30, 30],
                           "dualpol": [0, 20]}
    airspy_stages = args.airspy_stages or default_airspy_stages.get(mode, [])

    # The exact data-clock spacing between emitted frames: N kotekan
    # integrations per frame x kotekan's per-integration period. This is
    # what the client's gap detector / time axis must key off. Reporting
    # the *requested* interval instead (rather than the realized
    # N x period) is what caused the every-other-row NaN striping in
    # crosscorr -- there the kotekan period (~105 ms) exceeds the 50 ms
    # request, so N clamps to 1 and the true spacing is ~105 ms, not 50.
    frame_period_s = float(kotekan.integrate_n) * float(kotekan.kotekan_period_s)

    return {
        "version": VIEWER_PROTOCOL_VERSION,
        "mode": mode,
        "nfreq": int(kotekan.send_nfreq),
        "nvis": int(kotekan.frame_nvis),
        "vis_labels": kotekan.vis_labels,
        "frame_period_s": frame_period_s,
        "kotekan": {
            "rest_port": args.kotekan_rest_port,
            "airspy_stages": airspy_stages,
            "lag_align_stage": args.lag_align_stage,
        },
        "ui": {
            "color_range": default_color_range.get(mode, [-20, 20]),
            # Always ascending, matching the flipped display order the browser
            # receives (the raw band may be descending; see start()).
            "freq_range_mhz": sorted([
                float(kotekan.frame_freqs[0, 0] / 1e6),
                float(kotekan.frame_freqs[-1, 1] / 1e6),
            ]),
            # Dataset-specific knobs the panels read instead of hardcoding:
            # baseline-fit line mask, auto-cal [calibrate, observe] freqs, tuner span.
            "line_mask_mhz": [1419.9, 1420.9],  # HI line, 1420.4 +/- 0.5
            "autocal_freqs_mhz": [1416.0, 1421.0],
            "tuning_range_mhz": [24.0, 1800.0],  # airspy R2 span
        },
        "optional_modules": {
            "airspy_controls": bool(airspy_stages) and not args.no_airspy_controls,
            "ccera_pointing": args.ccera_pointing,
            "galaxy_view": bool(args.galaxy_view_url),
            "galaxy_view_url": args.galaxy_view_url,
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
        # Tell the TCP transport to push us pause/resume callbacks when its
        # write buffer crosses the high/low water marks. ``streaming=True``
        # selects the IPushProducer (sync) protocol.
        self.transport.registerProducer(self, True)

        kotekan = self.factory.kotekan
        # Size the kernel + Twisted write buffers to ~WS_BUFFER_FRAMES
        # actual frames. A frame is the MSG_TIMESTEP header (1 byte type +
        # 8 byte f64 time) plus nvis*nfreq f32 spectrum bins. Sizing by
        # frame count keeps backpressure semantics consistent across modes
        # (a fixed byte cap was ~1 crosscorr frame, tripping pause/resume
        # on every single frame).
        frame_bytes = 9 + kotekan.frame_nvis * kotekan.send_nfreq * 4
        buf_bytes = max(WS_BUFFER_MIN_BYTES, WS_BUFFER_FRAMES * frame_bytes)
        try:
            self.transport.getHandle().setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, buf_bytes
            )
        except (AttributeError, OSError) as e:
            log_.warning("WS %s: could not set SO_SNDBUF: %s", self.peer, e)
        self.transport.bufferSize = buf_bytes
        self.sendfreq = kotekan.send_nfreq  # after freq summing (see sum_freq)

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

        # Binary FREQLIST -- downsampled, ascending centres (see start()).
        self.sendMessage(
            np.int8(MSG_FREQLIST).tobytes() + kotekan.display_freqs_mhz.tobytes(),
            isBinary=True,
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
        spectrum = kotekan.display_spectrum()  # (nvis, send_nfreq), ascending
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
        return json.dumps({"mode": self.kotekan.mode}).encode("utf-8")


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
        "--sum-freq",
        type=int,
        default=1,
        metavar="N",
        help="Sum N adjacent frequency channels into one before sending to "
        "the browser (default 1 = no downsampling). e.g. --sum-freq 4 turns "
        "a 1024-channel band into 256, lightening the viewer. A partial tail "
        "(nfreq %% N) is dropped.",
    )
    ap.add_argument(
        "--power-dtype",
        choices=["float32", "uint32"],
        default="float32",
        help="How to interpret the power-spectrum payload bytes. The airspy "
        "pipeline (simpleAutocorr) sends float32 (the default); ARO's "
        "computeDualpolPower sends uint32 integer power sums -- pass "
        "'uint32' for ARO, which also normalises by samples_summed to a "
        "per-sample mean power for a stable dB scale.",
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
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.startLogging(sys.stdout, setStdout=False)

    static_dir = os.path.dirname(os.path.abspath(__file__))

    kotekan = KotekanPowerStream(
        host=args.kotekan_host,
        port=args.kotekan_port,
        emit_interval_s=args.viewer_integration_ms / 1000.0,
        power_dtype=args.power_dtype,
        sum_freq=args.sum_freq,
    )
    kotekan.start()  # blocks on accept until kotekan connects

    if kotekan.mode == "unknown":
        log_.warning(
            "Unrecognised kotekan nvis=%d; proceeding anyway.", kotekan.frame_nvis
        )

    viewer_config = build_viewer_config(kotekan, args)
    log_.info("viewer_config: %s", json.dumps(viewer_config, indent=2))

    ws_factory = LiveBeamWSFactory(kotekan, viewer_config)
    kotekan.on_frame = ws_factory.broadcast

    reactor.listenTCP(args.ws_port, ws_factory)

    # HTTP root: static files + /mode endpoint. NoCacheFile saves us a lot of
    # "I edited the JS but the browser ignored me" trouble during development.
    root = NoCacheFile(static_dir)
    root.putChild(b"mode", ModeResource(kotekan))
    reactor.listenTCP(args.http_port, server.Site(root))

    def _on_shutdown():
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
