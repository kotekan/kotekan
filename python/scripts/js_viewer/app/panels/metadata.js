// Small metadata readout for the Control card: a connection light plus the
// key instrument parameters (mode, band, channel count + spectral resolution,
// frame cadence). Useful for any pipeline, not just ARO -- everything comes
// from the viewer_config the server sends at connect.

export class MetadataPanel {
    constructor({app, target, meta}) {
        this.app = app;
        this.bus = app.bus;
        this._target = target;
        this.meta = meta || {};
        this._build();
        // The panel is created *after* viewer_config arrives, i.e. after the
        // socket already fired ws:open -- so seed the light from the live
        // socket state instead of waiting for an event that already passed.
        const ws = this.app.socket && this.app.socket.ws;
        this._set_conn(!!ws && ws.readyState === 1 /* WebSocket.OPEN */);
        this.bus.on("ws:open", () => this._set_conn(true));
        this.bus.on("ws:close", () => this._set_conn(false));
        this.bus.on("ws:reconnect_scheduled", () => this._set_conn(false));
    }

    _build() {
        const m = this.meta;
        const wrap = $("<div/>")
            .css({margin: "6px 8px", "font-family": "sans-serif", "font-size": "12px",
                  color: "#333"})
            .appendTo("#" + this._target);

        // Connection light.
        const connrow = $("<div/>")
            .css({display: "flex", "align-items": "center", gap: "6px",
                  "margin-bottom": "4px"})
            .appendTo(wrap);
        this._light = $("<span/>")
            .css({width: "10px", height: "10px", "border-radius": "50%",
                  background: "#d62728", display: "inline-block",
                  "box-shadow": "0 0 3px rgba(0,0,0,0.4)"})
            .appendTo(connrow);
        this._connlabel = $("<span/>").text("connecting…")
            .css({"font-weight": "600"}).appendTo(connrow);

        // Parameter rows.
        const grid = $("<div/>")
            .css({display: "grid", "grid-template-columns": "auto 1fr",
                  "column-gap": "8px", "row-gap": "2px"})
            .appendTo(wrap);
        const row = (label, value) => {
            $("<span/>").text(label).css({color: "#888"}).appendTo(grid);
            $("<span/>").text(value).appendTo(grid);
        };

        if (m.mode) row("Mode", m.mode + (m.vis_labels ? ` (${m.vis_labels.join("/")})` : ""));
        if (m.band) row("Band", `${m.band[0].toFixed(1)}–${m.band[1].toFixed(1)} MHz`);
        if (m.nchan) {
            const res = m.res_mhz;
            const restxt = res == null ? "" :
                (res >= 1 ? ` (${res.toFixed(3)} MHz/ch)` : ` (${(res * 1e3).toFixed(1)} kHz/ch)`);
            row("Channels", `${m.nchan}${restxt}`);
        }
        if (m.cadence_ms) {
            const c = m.cadence_ms;
            row("Cadence", c >= 1000 ? `${(c / 1000).toFixed(3)} s/frame`
                                     : `${c.toFixed(1)} ms/frame`);
        }
    }

    _set_conn(ok) {
        this._light.css("background", ok ? "#2ca02c" : "#d62728");
        this._connlabel.text(ok ? "connected" : "disconnected");
    }
}
