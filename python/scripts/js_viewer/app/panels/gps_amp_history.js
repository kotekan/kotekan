// GPS deep-detection history: for a PRN picked from a dropdown, a live time series of EITHER
//   * Â  = deep_amplitude (the deep nav-wiped matched-filter amplitude), 1σ = Â/deep_snr, OR
//   * C/N₀ = 10·log10(deep_snr² / coherence_s) dB-Hz -- the gain-independent, un-compressed
//     signal-quality measure. Â ∝ √(C/N₀) and its absolute level tracks front-end gain, so it
//     looks flat across the (power-controlled, hemispherically-uniform) GPS sky; C/N₀ squares the
//     SNR back out so the real per-pass / per-satellite variation is visible. Dividing by
//     coherence_s (the winning deep window's coherent time) keeps it stable when the auto-coherence
//     ladder switches window length. The dB-Hz zero-point is approximate (nav-wipe processing loss),
//     but the VARIATION is right and comparable across days/receivers.
//
// Polls <combiner>/get_status on its own timer and BUFFERS the raw (time, Â, significance,
// coherence_s) per PRN in the browser only -- tiny volume, not served forward, fresh on reload.

const POLL_MS = 1500;
const MAX_PTS = 10000;      // per-PRN cap (~4 h at 1.5 s); drops oldest past this so the live
                            // re-render stays snappy. Data is tiny; this bounds RENDER cost, not RAM.
const LN10_10 = 10 / Math.LN10;   // 10/ln10 = 4.3429; d(dB)/dx = LN10_10 / x

export class GpsAmpHistoryPanel {
    constructor({app, target, combiner_stage}) {
        this.app = app;
        this.combiner_stage = combiner_stage || "combiner";
        this.hist = new Map();       // prn -> {t:[Date], a:[Â], sig:[σ significance], coh:[s]}
        this.selected = null;
        this.mode = "amp";           // "amp" (Â) | "cn0" (C/N₀ dB-Hz)
        this._inflight = false;
        this._dirty = false;         // new data for the selected PRN since last redraw
        this._plotted = false;
        this._maxDr = 0;             // largest deep_records seen = the converged (~1 s) window

        const root = $("#" + target);
        root.css({display: "flex", flexDirection: "column", width: "100%", height: "100%",
                  boxSizing: "border-box", fontFamily: "sans-serif", fontSize: "12px"});

        // Controls: PRN dropdown + quantity toggle + live readout + clear.
        const bar = $("<div/>").css({flex: "0 0 auto", padding: "2px 4px 4px",
            display: "flex", alignItems: "center", gap: "8px"}).appendTo(root);
        $("<span/>").text("PRN").css({color: "#444"}).appendTo(bar);
        this.sel = $("<select/>").css({fontSize: "12px"}).appendTo(bar);
        this.sel.on("change", () => {
            this.selected = this.sel.val(); // constellation-tagged key ("G12"/"E25"/"C19")
            this._plotted = false;   // force a fresh newPlot (axis rescale) on PRN switch
            this._redraw();
        });
        this.modeSel = $("<select/>").css({fontSize: "12px"}).appendTo(bar);
        this.modeSel.append($("<option/>").attr("value", "amp").text("Â"));
        this.modeSel.append($("<option/>").attr("value", "cn0").text("C/N₀"));
        this.modeSel.on("change", () => {
            this.mode = this.modeSel.val();
            this._plotted = false;   // Â and C/N₀ live on very different y-scales -> rescale
            this._redraw();
        });
        this._info = $("<span/>").css({color: "#666", marginLeft: "4px"}).appendTo(bar);
        $("<button/>").text("clear").css({fontSize: "11px", marginLeft: "auto"})
            .appendTo(bar).on("click", () => {
                if (this.selected != null && this.hist.has(this.selected))
                    this.hist.set(this.selected, {t: [], a: [], sig: [], coh: [], dr: []});
                this._plotted = false;
                this._redraw();
            });

        // Plot area.
        this.plot = $("<div/>").css({flex: "1 1 auto", minHeight: "0"}).appendTo(root)[0];

        // Re-fit Plotly to the card when GridStack resizes it.
        if (window.ResizeObserver) {
            this._ro = new ResizeObserver(() => {
                if (this._plotted && window.Plotly) window.Plotly.Plots.resize(this.plot);
            });
            this._ro.observe(this.plot);
        }

        this._tick();
        this._timer = setInterval(() => this._tick(), POLL_MS);
    }

    _tick() {
        if (this._inflight) return;
        this._inflight = true;
        const k = this.app.kotekan;
        // Tri-constellation: poll every combiner; missing stages 404 -> null -> skipped.
        // History keys are "G12"/"E25"/"C19".
        const chains = [["G", this.combiner_stage], ["E", "gal_combiner"], ["C", "bds_combiner"]];
        Promise.all(chains.map(([tag, stage]) =>
            k.stageGet(stage, "get_status")
                .then(r => r.ok ? r.json() : null).catch(() => null)
                .then(st => [tag, st])))
            .then(results => {
                this._inflight = false;
                const now = new Date();
                for (const [tag, status] of results) {
                if (!Array.isArray(status)) continue;
                for (const c of status) {
                    const prn = c.prn ? tag + c.prn : null;
                    if (!prn) continue;
                    const A = c.deep_amplitude || c.unbiased_amplitude || 0;
                    const coh = c.coherence_s || 0;
                    // deep_snr counts only when floor-cleared (coherence_s > 0): a floored
                    // deep (~7 sigma at the full window) is noise rectification, not signal.
                    const sig = coh > 0 ? Math.max(c.deep_snr || 0, c.amp_snr || 0)
                                        : (c.amp_snr || 0);
                    if (A === 0 && sig === 0) continue;   // no despread this emit -> a gap, not a 0
                    const dr = c.deep_records || 0;
                    if (dr > this._maxDr) this._maxDr = dr;   // the run's converged (~1 s) window
                    if (!this.hist.has(prn)) this.hist.set(prn, {t: [], a: [], sig: [], coh: [], dr: []});
                    const h = this.hist.get(prn);
                    const n = h.a.length;
                    // Dedup: consecutive polls can catch the SAME combiner emit -> identical
                    // floats. Skip those so the trace advances once per real emit.
                    if (n && h.a[n - 1] === A && h.sig[n - 1] === sig) continue;
                    h.t.push(now); h.a.push(A); h.sig.push(sig); h.coh.push(coh); h.dr.push(dr);
                    if (h.t.length > MAX_PTS) { h.t.shift(); h.a.shift(); h.sig.shift(); h.coh.shift(); h.dr.shift(); }
                    if (prn === this.selected) this._dirty = true;
                }
                }
                this._sync_dropdown();
                this._redraw();
            });
    }

    // Keep the dropdown in sync with the PRNs we've seen (sorted), preserving the selection.
    _sync_dropdown() {
        const prns = [...this.hist.keys()].sort((a, b) =>
            a[0] === b[0] ? parseInt(a.slice(1)) - parseInt(b.slice(1)) : a.localeCompare(b));
        const cur = this.sel.val();
        const have = new Set();
        this.sel.find("option").each(function () { have.add($(this).val()); });
        let changed = false;
        for (const p of prns) if (!have.has(String(p))) changed = true;
        if (changed) {
            this.sel.empty();
            for (const p of prns)
                this.sel.append($("<option/>").attr("value", p).text(p));
            // Preserve selection; else default to the PRN with the most history.
            if (this.selected != null && this.hist.has(this.selected))
                this.sel.val(String(this.selected));
            else if (prns.length) {
                const best = prns.reduce((m, p) =>
                    this.hist.get(p).a.length > this.hist.get(m).a.length ? p : m, prns[0]);
                this.selected = best;
                this.sel.val(String(best));
                this._plotted = false;
                this._dirty = true;
            }
        }
        if (cur && this.sel.val() !== cur && this.hist.has(cur))
            this.sel.val(cur);
    }

    // Map the buffered raw sample i to the plotted (y, 1σ) for the current mode.
    _yval(h, i) {
        const A = h.a[i], sig = h.sig[i], coh = h.coh[i];
        if (this.mode === "cn0") {
            // C/N₀ (dB-Hz) = 10 log10(deep_snr^2 / T_coh); dividing by the coherent time makes it a
            // density (stable across ladder window changes). 1σ from the significance's ~unit std:
            // d(dB)/d(sig) = (10/ln10)*(2/sig). coherence_s == 0 means the combiner certified NO
            // coherent detection (no rung beat its rectification floor) -> a gap, not a value.
            if (!(coh > 0)) return [null, 0];
            const y = sig > 0 ? 20 * Math.log10(sig) - 10 * Math.log10(coh) : null;
            const e = sig > 0 ? (2 * LN10_10) / sig : 0;
            return [y, e];
        }
        return [A, sig > 0 ? A / sig : 0];   // Â and its noise-scale error bar
    }

    _redraw() {
        if (this.selected == null || !window.Plotly) return;
        const h = this.hist.get(this.selected);
        if (!h) return;
        const cn0 = this.mode === "cn0";

        // Fresh arrays each render (Plotly.react diffs data BY REFERENCE, so reusing the in-place
        // buffers would make it skip the redraw -- the plot would only refresh on a full newPlot).
        // De-emphasize only LOW-SIGNIFICANCE short-window emits: near the ~7σ nav-wipe floor the
        // auto-coherence ladder's short windows are extreme-value-inflated (spurious up-spikes),
        // but a HIGH-significance short window is real signal -- post the 2026-07-11 code-loop fix
        // the ladder legitimately rides 125-500 ms windows at 50-160σ, and greying those hid the
        // true C/N₀ as "artifact". 20σ is safely above anything the floor's selection bias makes.
        const T = h.t.slice(), Yf = new Array(h.a.length), Ys = new Array(h.a.length),
              E = new Array(h.a.length);
        const thr = 0.6 * (this._maxDr || 0);
        const SIG_REAL = 20;   // significance above which any window length is trusted
        let lastFull = -1;
        for (let i = 0; i < h.a.length; i++) {
            const ye = this._yval(h, i); E[i] = ye[1];
            const isShort = this._maxDr > 0 && (h.dr[i] || 0) < thr
                            && (h.sig[i] || 0) < SIG_REAL;
            Yf[i] = isShort ? null : ye[0];
            Ys[i] = isShort ? ye[0] : null;
            if (!isShort) lastFull = i;
        }

        // Live readout: the latest FULL-window value (short-window emits are artifacts, not shown).
        const n = h.a.length;
        if (lastFull >= 0) {
            const sig = h.sig[lastFull], yv = this._yval(h, lastFull);
            const tag = (lastFull !== n - 1) ? ' · <span style="color:#aaa">short win now</span>' : '';
            if (cn0)
                this._info.html(`C/N₀ ≈ <b>${(yv[0] ?? 0).toFixed(1)}</b> ± ${yv[1].toFixed(1)} dB-Hz`
                    + `&nbsp;(${sig.toFixed(1)}σ)&nbsp;· ${n} pts` + tag);
            else
                this._info.html(`Â = <b>${h.a[lastFull].toFixed(3)}</b> ± ${yv[1].toFixed(3)}`
                    + `&nbsp;(${sig.toFixed(1)}σ)&nbsp;· ${n} pts` + tag);
        } else if (n) {
            this._info.html('<span style="color:#aaa">short-window emits only — integration converging…</span>');
        } else {
            this._info.text("(no samples yet)");
        }

        const unit = cn0 ? " dB-Hz" : "";
        const fmt = cn0 ? ".1f" : ".3f";
        const full = {
            x: T, y: Yf, customdata: E, type: "scatter", mode: "lines+markers",
            line: {color: "#1f77b4", width: 1}, marker: {size: 3, color: "#1f77b4"}, connectgaps: false,
            error_y: {type: "data", array: E, visible: true,
                      color: "rgba(31,119,180,0.35)", thickness: 1, width: 0},
            hovertemplate: `%{x|%H:%M:%S}<br>%{y:${fmt}} ± %{customdata:${fmt}}${unit}<extra></extra>`,
        };
        const shortWin = {
            x: T, y: Ys, type: "scatter", mode: "markers",
            marker: {size: 2.5, color: "rgba(150,150,150,0.45)"},
            hovertemplate: `%{x|%H:%M:%S}<br>%{y:${fmt}}${unit} · short window, low sig<extra></extra>`,
        };
        const layout = {
            margin: {l: 52, r: 10, t: 6, b: 28},
            xaxis: {type: "date", tickfont: {size: 10}, gridcolor: "#eee"},
            yaxis: {title: {text: cn0 ? "C/N₀ (dB-Hz, est.)" : "Â  (deep |A|)", font: {size: 11}},
                    rangemode: cn0 ? "normal" : "tozero", tickfont: {size: 10},
                    gridcolor: "#eee", zeroline: !cn0},
            showlegend: false, paper_bgcolor: "white", plot_bgcolor: "white",
        };
        if (!this._plotted) {
            window.Plotly.newPlot(this.plot, [full, shortWin], layout,
                {displayModeBar: false, responsive: true});
            this._plotted = true;
        } else if (this._dirty) {
            window.Plotly.react(this.plot, [full, shortWin], layout,
                {displayModeBar: false, responsive: true});
        }
        this._dirty = false;
    }
}
