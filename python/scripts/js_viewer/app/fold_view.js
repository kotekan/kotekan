// Pulse-fold view for the dualpol pipeline.
//
// Renders the server's folded profile (MSG_FOLD -> state.fold, emitted a
// couple of times a second): a phase-vs-power profile line per pol on top,
// and a phase-vs-frequency heatmap per pol below. With dedispersion off the
// heatmap shows the dispersion sweep as a diagonal; with it on (matching DM)
// the pulse stacks into a vertical stripe and the profile peak sharpens.
//
// state.fold = { nphase, nvis, nfreq, data: Float32Array(nvis*nphase*nfreq) }
// data layout: data[p*nphase*nfreq + ph*nfreq + f], linear power, NaN = empty.

const POL_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"];
const dB = (v) => 10 * Math.log10(v);

export class FoldView {
    constructor({app, target, labels}) {
        this.app = app;
        this.state = app.state;
        this.bus = app.bus;
        this.cb = app.state.cb;             // shared dB colormap (phase-freq heatmap)
        this.labels = labels && labels.length ? labels : ["XX", "YY"];

        const host = $("#" + target).css({
            display: "flex", "flex-direction": "column", height: "100%",
            "min-height": "0", padding: "4px", "box-sizing": "border-box",
            "font-family": "sans-serif",
        });

        // Profile canvas (top).
        const prof_cell = $("<div/>")
            .css({position: "relative", flex: "0 0 38%", "min-height": "0"})
            .appendTo(host);
        this.prof_canvas = $("<canvas/>")
            .css({width: "100%", height: "100%", display: "block"})
            .appendTo(prof_cell);

        // Heatmap row (bottom): one phase-freq canvas per pol.
        const hm_row = $("<div/>")
            .css({flex: "1 1 0", "min-height": "0", display: "grid",
                  "grid-template-columns": `repeat(${this.labels.length}, 1fr)`,
                  gap: "6px", "margin-top": "4px"})
            .appendTo(host);
        this.hm_canvases = this.labels.map((lab) => {
            const cell = $("<div/>")
                .css({position: "relative", "min-height": "0", "min-width": "0",
                      display: "flex", "flex-direction": "column",
                      border: "1px solid #ddd", "border-radius": "3px", overflow: "hidden"})
                .appendTo(hm_row);
            $("<div/>").text(lab + "  (phase ↑ vs freq →)")
                .css({"font-size": "11px", "font-weight": "600", padding: "2px 6px",
                      background: "#eef2f6", "border-bottom": "1px solid #d0d7de"})
                .appendTo(cell);
            const c = $("<canvas/>")
                .css({width: "100%", height: "100%", display: "block",
                      "image-rendering": "pixelated", flex: "1 1 0", "min-height": "0"})
                .appendTo(cell);
            return c;
        });

        this._placeholder(prof_cell);
        // Render on a new fold frame (a few seconds apart). We deliberately do
        // NOT render on the 40 Hz state:redraw_requested -- re-rendering the two
        // heatmaps + profile every frame starved the WS drain and dropped power
        // frames (waterfall gap-stripes).
        this.bus.on("state:fold_received", (f) => this.render(f));
        // But DO re-render on a display change (colormap, colour range, median
        // toggle) so the fold tracks the waterfall immediately instead of
        // waiting for the next fold frame. Throttled to ~10 Hz so dragging the
        // colour slider (which fires continuously) doesn't thrash -- it's a pure
        // client-side re-render of the last fold, so it's safe.
        this.bus.on("state:display_changed", () => this._redraw_display());
    }

    _redraw_display() {
        if (!this._last) return;
        const now = Date.now();
        if (this._last_disp && now - this._last_disp < 100) {
            if (!this._disp_timer) {
                this._disp_timer = setTimeout(() => {
                    this._disp_timer = null;
                    this._last_disp = Date.now();
                    if (this._last) this.render(this._last);
                }, 100);
            }
            return;
        }
        this._last_disp = now;
        this.render(this._last);
    }

    // Median of the first ``n`` entries of ``buf`` (sorts that slice in place).
    _median(buf, n) {
        const a = buf.subarray(0, n);
        a.sort();
        const m = n >> 1;
        return (n & 1) ? a[m] : 0.5 * (a[m - 1] + a[m]);
    }

    // Per-channel median over pulse phase, in dB -- the fold analog of the
    // waterfall's per-channel time median. Subtracting it removes the standing
    // bandpass from the fold so the pulse (which varies with phase) stands out.
    _pol_medians(row, nphase, nfreq) {
        const med = new Float32Array(nfreq);
        const col = new Float32Array(nphase);
        for (let f = 0; f < nfreq; f++) {
            let n = 0;
            for (let ph = 0; ph < nphase; ph++) {
                const v = row[ph * nfreq + f];
                if (v === v && v > 0) col[n++] = dB(v);
            }
            med[f] = n ? this._median(col, n) : 0;
        }
        return med;
    }

    _placeholder(cell) {
        this._ph = $("<div/>")
            .text("Fold off — pick a target and enable folding.")
            .css({position: "absolute", inset: 0, display: "flex",
                  "align-items": "center", "justify-content": "center",
                  color: "#888", "font-size": "13px"})
            .appendTo(cell);
    }

    // Split state.fold.data into an array of nphase*nfreq subarrays, one per pol.
    _split(fold) {
        const {nphase, nfreq, nvis, data} = fold;
        const out = [];
        for (let p = 0; p < nvis; p++) {
            out.push(data.subarray(p * nphase * nfreq, (p + 1) * nphase * nfreq));
        }
        return {nphase, nfreq, nvis, pols: out};
    }

    render(fold) {
        if (!fold || !fold.data || !fold.data.length) return;
        if (this._ph) { this._ph.remove(); this._ph = null; }
        this._last = fold;
        this._last_render = Date.now();
        // Per-pol median (if median-subtract on) computed once here and shared
        // by the profile and heatmap draws, rather than twice each.
        const {nphase, nfreq, nvis, pols} = this._split(fold);
        const meds = this.state.median_subtract
            ? pols.map((row) => this._pol_medians(row, nphase, nfreq)) : null;
        this._draw_profile(fold, meds);
        this._draw_heatmaps(fold, meds);
    }

    // Per-pol profile: nanmean over freq -> dB, drawn as a line with a light grid.
    _draw_profile(fold, meds) {
        const {nphase, nfreq, nvis, pols} = this._split(fold);
        const cv = this.prof_canvas[0];
        const W = cv.clientWidth || 400, H = cv.clientHeight || 150;
        cv.width = W; cv.height = H;
        const g = cv.getContext("2d");
        g.clearRect(0, 0, W, H);

        // profiles[p][ph] = mean over freq of the (optionally median-subtracted)
        // per-channel dB -- i.e. the mean of what the heatmap shows.
        const profiles = [];
        let lo = Infinity, hi = -Infinity;
        for (let p = 0; p < nvis; p++) {
            const row = pols[p];
            const med = meds ? meds[p] : null;
            const prof = new Float32Array(nphase);
            for (let ph = 0; ph < nphase; ph++) {
                let sum = 0, n = 0;
                const base = ph * nfreq;
                for (let f = 0; f < nfreq; f++) {
                    const v = row[base + f];
                    if (v === v && v > 0) { sum += dB(v) - (med ? med[f] : 0); n++; }
                }
                const d = n ? sum / n : NaN;
                prof[ph] = d;
                if (d === d) { if (d < lo) lo = d; if (d > hi) hi = d; }
            }
            profiles.push(prof);
        }
        if (!(hi > lo)) { hi = lo + 1; }
        const pad = (hi - lo) * 0.1 || 1;
        lo -= pad; hi += pad;

        const ml = 44, mb = 18, mt = 6, mr = 6;
        const x = (ph) => ml + (ph / (nphase - 1)) * (W - ml - mr);
        const y = (d) => mt + (1 - (d - lo) / (hi - lo)) * (H - mt - mb);

        // grid + axes
        g.strokeStyle = "#eee"; g.fillStyle = "#666"; g.font = "10px sans-serif";
        g.lineWidth = 1;
        for (let k = 0; k <= 4; k++) {
            const ph = k / 4 * (nphase - 1);
            g.beginPath(); g.moveTo(x(ph), mt); g.lineTo(x(ph), H - mb); g.stroke();
            g.fillText((k / 4).toFixed(2), x(ph) - 8, H - 6);
        }
        for (let k = 0; k <= 3; k++) {
            const d = lo + k / 3 * (hi - lo);
            g.beginPath(); g.moveTo(ml, y(d)); g.lineTo(W - mr, y(d)); g.stroke();
            g.fillText(d.toFixed(1), 4, y(d) + 3);
        }
        g.fillText("phase", (W - ml) / 2 + ml - 12, H - 6);

        // profile lines
        for (let p = 0; p < nvis; p++) {
            const prof = profiles[p];
            g.strokeStyle = POL_COLORS[p % POL_COLORS.length];
            g.lineWidth = 1.5;
            g.beginPath();
            let started = false;
            for (let ph = 0; ph < nphase; ph++) {
                const d = prof[ph];
                if (d !== d) { started = false; continue; }
                if (!started) { g.moveTo(x(ph), y(d)); started = true; }
                else g.lineTo(x(ph), y(d));
            }
            g.stroke();
            // legend swatch
            g.fillStyle = POL_COLORS[p % POL_COLORS.length];
            g.fillRect(W - mr - 60, mt + 2 + p * 12, 10, 3);
            g.fillStyle = "#333";
            g.fillText(this.labels[p] || ("pol" + p), W - mr - 46, mt + 6 + p * 12);
        }
    }

    // Per-pol phase-frequency heatmap: x = freq (ascending), y = phase (0 top).
    // Respects the median-subtract toggle (per-channel median over phase).
    _draw_heatmaps(fold, meds) {
        const {nphase, nfreq, nvis, pols} = this._split(fold);
        const cb = this.cb;
        for (let p = 0; p < nvis && p < this.hm_canvases.length; p++) {
            const cv = this.hm_canvases[p][0];
            cv.width = nfreq; cv.height = nphase;
            const g = cv.getContext("2d");
            g.imageSmoothingEnabled = false;
            const img = g.createImageData(nfreq, nphase);
            const row = pols[p];
            const med = meds ? meds[p] : null;
            for (let ph = 0; ph < nphase; ph++) {
                const base = ph * nfreq;
                for (let f = 0; f < nfreq; f++) {
                    const v = row[base + f];
                    if (v === v && v > 0) cb.setPixel(img, f, ph, dB(v) - (med ? med[f] : 0));
                    // else: leave transparent (empty phase bin)
                }
            }
            g.putImageData(img, 0, 0);
        }
    }
}
