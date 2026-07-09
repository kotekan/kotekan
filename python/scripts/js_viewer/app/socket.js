// WebSocket client for the livebeam_server stream.
//
// Receives three kinds of messages:
//   * text JSON at WS-open containing nfreq + viewer_config,
//   * binary FREQLIST (msgtype 1),
//   * binary TIMESTEP (msgtype 2) per integration.
//
// Updates ``app.state`` directly and emits bus events for the panels.
//
// Robustness
// ----------
// * The server enables autobahn ping/pong; if the connection goes stale
//   the server drops it, which fires our onclose and triggers a reconnect.
// * Unintentional closes (server-side drop, network blip) auto-reconnect
//   with exponential backoff. Closes from user action (StartStopPanel)
//   are flagged so they don't kick off a reconnect loop.
// * The first message from the server carries a viewer_config with a
//   protocol ``version``; if it doesn't match CLIENT_PROTOCOL_VERSION we
//   warn the user (the layout shell shows a banner).

const MSG_FREQLIST = 1;
const MSG_TIMESTEP = 2;
const MSG_FOLD     = 3;

// Bump this when the WS protocol or viewer_config shape changes in a way
// the client can't handle. Server-side lives in livebeam_server.py as
// VIEWER_PROTOCOL_VERSION.
const CLIENT_PROTOCOL_VERSION = 1;

// Reconnect backoff: 500ms, 1s, 2s, ..., capped at 30s.
const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 30000;

export class Socket {
    constructor({app, url}) {
        this.app = app;
        this.url = url;
        this.ws = null;

        this._user_close = false;
        this._reconnect_delay = RECONNECT_MIN_MS;
        this._reconnect_timer = null;
    }

    connect() {
        // A fresh connect attempt -- whether user-initiated or from the
        // reconnect timer -- clears the user-close flag.
        this._user_close = false;
        if (this._reconnect_timer) {
            clearTimeout(this._reconnect_timer);
            this._reconnect_timer = null;
        }
        this._open_ws();
    }

    _open_ws() {
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = "arraybuffer";
        this.ws.onopen = () => {
            this._reconnect_delay = RECONNECT_MIN_MS;
            this.app.bus.emit("ws:open");
        };
        this.ws.onmessage = (e) => this._onmessage(e);
        this.ws.onclose = () => {
            // user_initiated lets listeners (eg WaterfallView's gap detector)
            // distinguish "user paused" from "server / network dropped us".
            this.app.bus.emit("ws:close", {user_initiated: this._user_close});
            if (!this._user_close) this._schedule_reconnect();
        };
        this.ws.onerror = (err) => console.error("WebSocket error:", err);
    }

    _schedule_reconnect() {
        const delay = this._reconnect_delay;
        console.warn(`WebSocket dropped; reconnecting in ${delay}ms`);
        this.app.bus.emit("ws:reconnect_scheduled", {delay_ms: delay});
        this._reconnect_timer = setTimeout(() => {
            this._reconnect_timer = null;
            this._open_ws();
        }, delay);
        // Exponential backoff, capped.
        this._reconnect_delay = Math.min(this._reconnect_delay * 2, RECONNECT_MAX_MS);
    }

    close() {
        // User-initiated close (StartStopPanel). Mark so we don't try to
        // reconnect; also cancel any pending reconnect timer.
        this._user_close = true;
        if (this._reconnect_timer) {
            clearTimeout(this._reconnect_timer);
            this._reconnect_timer = null;
        }
        if (this.ws) this.ws.close();
    }

    _onmessage(e) {
        const {app} = this;
        const {state, bus} = app;

        if (typeof e.data === "string") {
            const msg = JSON.parse(e.data);
            const cfg = msg.viewer_config;
            if (cfg) {
                const server_v = cfg.version;
                if (server_v != null && server_v !== CLIENT_PROTOCOL_VERSION) {
                    console.warn(`Viewer protocol mismatch: client=${CLIENT_PROTOCOL_VERSION}, server=${server_v}`);
                    bus.emit("ws:version_mismatch",
                             {client: CLIENT_PROTOCOL_VERSION, server: server_v});
                }
            }
            // Reallocate the baseline only on an actual bin-count change. The
            // server resends viewer_config on every reconnect, so resetting
            // unconditionally would wipe a captured baseline on any WS blip
            // (and, with subtraction on, divide by zeros -> +Inf).
            const new_num_freqs = (cfg && cfg.nfreq) || msg.nfreq;
            if (new_num_freqs !== state.num_freqs) {
                state.num_freqs = new_num_freqs;
                state.spectrum_baseline = new Array(state.num_freqs).fill(0);
                state.baseline_enabled = false; // captured baseline no longer valid
            }
            bus.emit("state:num_freqs_changed", {num_freqs: state.num_freqs});
            if (cfg) app.apply_viewer_config(cfg);
            bus.emit("state:redraw_requested");
            return;
        }

        const msgtype = new Int8Array(e.data.slice(0, 1))[0];
        if (msgtype === MSG_FREQLIST) {
            state.freq_list = new Float32Array(e.data.slice(1));
            bus.emit("state:freq_list_changed", {freq_list: state.freq_list});
            bus.emit("state:redraw_requested");
            return;
        }
        if (msgtype === MSG_TIMESTEP) {
            const timestamp = new Float64Array(e.data.slice(1, 9))[0];
            const data = new Float32Array(e.data.slice(9));
            bus.emit("state:frame_received", {timestamp, data});
            bus.emit("state:redraw_requested");
            return;
        }
        if (msgtype === MSG_FOLD) {
            // int32 nphase, then (nvis * nphase * nfreq) float32 folded power.
            const nphase = new Int32Array(e.data.slice(1, 5))[0];
            const data = new Float32Array(e.data.slice(5));
            state.fold = {nphase, data, nvis: state.nvis, nfreq: state.num_freqs};
            bus.emit("state:fold_received", state.fold);
            return;
        }
        console.log("Unknown binary msgtype:", msgtype);
    }

    // Send a JSON control message to the server (fold controls). No-op if the
    // socket isn't open.
    send(obj) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(obj));
        }
    }
}
