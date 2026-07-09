// Pulse-fold controls for the dualpol viewer.
//
// Lets the user enable folding, pick a target from the B-pulsar catalogue
// (psrcat_b.json, served alongside the viewer) or type a period/DM by hand,
// toggle dedispersion, and clear the accumulator. Every change is pushed to
// the server as a JSON control message over the WebSocket (see
// livebeam_server.onMessage / PulseFolder).

export class FoldPanel {
    constructor({app, target, nphase}) {
        this.app = app;
        this.state = app.state;
        this.bus = app.bus;
        this._target = target;
        this.nphase = nphase || 128;
        this.cfg = {enabled: false, period: 1.0, dm: 0.0, dedisperse: false};
        this._catalog = {};   // name -> {period, dm}
        this._build();
        this._load_catalog();
    }

    _row(label) {
        const row = $("<div/>").css({
            display: "flex", "align-items": "center", gap: "6px",
            margin: "5px 8px", "font-family": "sans-serif", "font-size": "13px",
        }).appendTo("#" + this._target);
        if (label) $("<span/>").text(label).css({"min-width": "58px"}).appendTo(row);
        return row;
    }

    _build() {
        const self = this;

        // Enable folding.
        const r0 = this._row("");
        this._enable = $("<input/>", {type: "checkbox"}).appendTo(r0);
        this._enable.uniqueId();
        $("<label/>").attr("for", this._enable.attr("id")).text("Enable pulse fold")
            .css({cursor: "pointer", "font-weight": "600"}).appendTo(r0);
        this._enable.on("change", function() {
            self.cfg.enabled = this.checked; self._send();
        });

        // Target dropdown (psrcat).
        const r1 = this._row("Target");
        this._sel = $("<select/>").css({flex: "1 1 0", "min-width": "0"}).appendTo(r1);
        $("<option/>", {value: "", text: "— select pulsar —"}).appendTo(this._sel);
        this._sel.on("change", function() {
            const e = self._catalog[this.value];
            if (e) {
                self.cfg.period = e.period; self.cfg.dm = e.dm;
                self._period.val(e.period.toFixed(6));
                self._dm.val(e.dm != null ? e.dm.toFixed(3) : "0");
                self._send();
            }
        });

        // Manual period / DM.
        const r2 = this._row("Period s");
        this._period = $("<input/>", {type: "number", step: "any", value: "1.0"})
            .css({flex: "1 1 0", "min-width": "0"}).appendTo(r2);
        const r3 = this._row("DM");
        this._dm = $("<input/>", {type: "number", step: "any", value: "0"})
            .css({flex: "1 1 0", "min-width": "0"}).appendTo(r3);
        const push_manual = () => {
            const P = parseFloat(self._period.val());
            const dm = parseFloat(self._dm.val());
            if (P > 0) self.cfg.period = P;
            if (dm >= 0) self.cfg.dm = dm;
            self.cfg._manual = true;
            self._send();
        };
        this._period.on("change", push_manual);
        this._dm.on("change", push_manual);

        // Dedisperse toggle.
        const r4 = this._row("");
        this._dedisp = $("<input/>", {type: "checkbox"}).appendTo(r4);
        this._dedisp.uniqueId();
        $("<label/>").attr("for", this._dedisp.attr("id")).text("Dedisperse (use DM)")
            .css({cursor: "pointer"}).appendTo(r4);
        this._dedisp.on("change", function() {
            self.cfg.dedisperse = this.checked; self._send();
        });

        // Clear button.
        const r5 = this._row("");
        $("<button/>").text("Clear fold").appendTo(r5)
            .on("click", () => self.app.socket.send({cmd: "clear_fold"}));

        this._status = $("<div/>")
            .css({margin: "2px 8px 6px", "font-size": "11px", color: "#666",
                  "font-family": "sans-serif"})
            .appendTo("#" + this._target);
    }

    _load_catalog() {
        fetch("psrcat_b.json")
            .then((r) => r.ok ? r.json() : Promise.reject(r.status))
            .then((cat) => {
                const ps = cat.pulsars || {};
                const names = Object.keys(ps).filter(
                    (n) => ps[n] && ps[n].frequency > 0).sort();
                for (const n of names) {
                    const e = ps[n];
                    const period = 1.0 / e.frequency;
                    const dm = (e.dmeasure != null) ? e.dmeasure : 0;
                    this._catalog[n] = {period, dm};
                    const dmtxt = (e.dmeasure != null) ? e.dmeasure.toFixed(1) : "?";
                    $("<option/>", {
                        value: n,
                        text: `${n}   P=${period.toFixed(3)}s  DM=${dmtxt}`,
                    }).appendTo(this._sel);
                }
                this._status.text(`${names.length} pulsars loaded.`);
            })
            .catch((e) => this._status.text("Catalogue unavailable (" + e + "); use manual period/DM."));
    }

    _send() {
        this.app.socket.send({
            cmd: "set_fold",
            enabled: this.cfg.enabled,
            period: this.cfg.period,
            dm: this.cfg.dm,
            dedisperse: this.cfg.dedisperse,
            nphase: this.nphase,
        });
        this._status.text(
            `${this.cfg.enabled ? "folding" : "off"}: P=${this.cfg.period.toFixed(4)}s `
            + `DM=${this.cfg.dm} ${this.cfg.dedisperse ? "(dedispersed)" : "(raw)"}`);
    }
}
