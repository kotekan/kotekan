// GNSS per-satellite time series: pick a satellite and an observable, watch it evolve. One
// trace per signal the satellite carries.
//
// Observables (the dropdown, MODES), all read off the shared GpsFeed so this panel and the
// table cannot disagree:
//   * C/N₀ prompt -- cn0_prompt_db: per-record prompt power, presence- and on-peak-gated,
//                    probe-debiased.
//   * C/N₀ coh    -- cn0_kcoh_db: the ~1 s known-rate coherent fold.
//   * q           -- the code-loop quality (fleet discriminator); the lock metric.
//   * beam A/φ    -- the per-element complex-gain summary (gps_feed.beam_summary): median
//                    live-element amplitude (1 = uniform array) and the circular RMS of the
//                    element phases about the array mean.
//   * sig         -- the broker's significance (σ above noise), from the first source that
//                    applies (sig_src).
//   * coh         -- the auto-ladder's winning coherent window (s); 0 = no rung beat its
//                    rectification floor.
//   * dop         -- the published doppler_hz (Hz), the broker's commanded seed: a staircase
//                    that steps at each re-seed.
//   * snr         -- search detection significance (acquire grid), independent of tracking.
//
// Points, not lines: samples are irregular and gapped (a dropout is real information), so
// connecting them draws structure that is not in the data.
//
// History is buffered in the browser only (not served, fresh on reload).

const MAX_PTS = 10000;      // per-series cap (~4 h at the 1.5 s poll); bounds the render cost
const LN10_10 = 10 / Math.LN10;   // 10/ln10 = 4.3429; d(dB)/dx = LN10_10 / x

// A prompt C/N0 point below this lock duty is dropped from the trace rather than drawn: the
// estimator still publishes at low duty, and one unqualified point in a history trace looks
// like a real one. The coherent trace applies the eta qualifier the same way, and the table
// dims below 0.7 duty.
const CN0_MIN_DUTY = 0.25;

// key -> {label, axis, unit, fmt (d3 format), zero (y-axis rangemode tozero)}
const MODES = {
    cn0:      {label: "C/N₀ prompt", axis: "C/N₀ prompt, per-record (dB-Hz)",
               unit: " dB-Hz", fmt: ".1f"},
    cn0_kcoh: {label: "C/N₀ coh", axis: "C/N₀ coherent, known-rate fold (dB-Hz)",
               unit: " dB-Hz", fmt: ".1f"},
    beam_amp: {label: "beam A",  axis: "median element gain (×, 1 = uniform)", unit: "×",
               fmt: ".2f", zero: true},
    beam_ph:  {label: "beam φ",  axis: "element phase spread (deg RMS)", unit: "°",
               fmt: ".0f", zero: true},
    // q: the fleet discriminator's prompt-vs-shoulder ratio, "is the code loop on the peak".
    // Judge lock here, not on sig or C/N0: the deep fold re-searches and will certify a
    // satellite whose prompt tap is on noise. 2.2 is the lock bar (gps_feed's Q_LOCK).
    q:       {label: "q",        axis: "code-loop quality (fleet discriminator)", unit: "",
              fmt: ".2f", zero: true},
    sig:     {label: "sig",      axis: "significance (σ)", unit: "σ", fmt: ".1f", zero: true},
    coh_s:   {label: "coh",      axis: "coherent window (s)", unit: " s", fmt: ".3f", zero: true},
    dop:     {label: "dop",      axis: "tracked Doppler (Hz)", unit: " Hz", fmt: ".1f"},
    snr:     {label: "snr",      axis: "search significance (σ)", unit: "σ", fmt: ".1f", zero: true},
};
const MODE_ORDER = ["cn0", "cn0_kcoh", "q", "beam_amp", "beam_ph", "sig", "coh_s", "dop",
                    "snr"];

// A distinct-per-trace palette. Colours go by position among a satellite's signals, not by
// signal, so any constellation's 2-5 signals stay visually separable.
const SIG_PALETTE = ["#4d9de0", "#6fbf73", "#e8923c", "#c65d21", "#d64550", "#8e6fb0", "#39a0a0"];

export class GpsAmpHistoryPanel {
    constructor({target, feed}) {
        this.feed = feed;
        this.hist = new Map();   // key -> {t:[], cn0:[], sig:[], coh_s:[], dop:[], snr:[], ...}
        this.meta = new Map();   // key -> {col} (one entry per sat+signal)
        this.selected = null;
        this.mode = "cn0";
        this._dirty = false;
        this._plotted = false;
        // Signals the user has toggled off, by signal column ("E5a-Q", "E5b-Q", ...).
        // Kept by signal rather than by series key so the choice survives changing PRN:
        // "show me only E5a" is a viewing mode, not a per-satellite decision. Plotly's own
        // legend click cannot do this job here: _redraw calls Plotly.react on every
        // feed tick with freshly built traces, which resets legend visibility a second later.
        this._hidden = new Set();

        const root = $("#" + target);
        root.css({display: "flex", flexDirection: "column", width: "100%", height: "100%",
                  boxSizing: "border-box", fontFamily: "sans-serif", fontSize: "12px"});

        const bar = $("<div/>").css({flex: "0 0 auto", padding: "2px 4px 4px",
            display: "flex", alignItems: "center", gap: "8px"}).appendTo(root);
        $("<span/>").text("PRN").css({color: "#444"}).appendTo(bar);
        this.sel = $("<select/>").css({fontSize: "12px"}).appendTo(bar);
        this.sel.on("change", () => {
            this.selected = this.sel.val(); // constellation-tagged key ("G12"/"E25"/"C19")
            this._plotted = false;          // force a fresh newPlot (axis rescale)
            this._redraw();
        });
        this.modeSel = $("<select/>").css({fontSize: "12px"}).appendTo(bar);
        for (const k of MODE_ORDER)
            this.modeSel.append($("<option/>").attr("value", k).text(MODES[k].label));
        this.modeSel.val(this.mode);
        this.modeSel.on("change", () => {
            this.mode = this.modeSel.val();
            this._plotted = false;          // observables live on very different y-scales
            this._redraw();
        });
        this._info = $("<span/>").css({color: "#666", marginLeft: "4px"}).appendTo(bar);
        // Click a signal label to show/hide that trace. Delegated: _info's html is rebuilt
        // on every redraw, so binding the spans themselves would go stale immediately.
        this._info.on("click", "[data-sig]", ev => {
            const sig = $(ev.currentTarget).attr("data-sig");
            if (this._hidden.has(sig)) this._hidden.delete(sig);
            else this._hidden.add(sig);
            this._plotted = false;   // force newPlot so the y-axis rescales to what is left
            this._redraw();
        });
        $("<button/>").text("clear").css({fontSize: "11px", marginLeft: "auto"})
            .appendTo(bar).on("click", () => {
                if (this.selected != null && this.hist.has(this.selected))
                    this.hist.set(this.selected, this._blank());
                this._plotted = false;
                this._redraw();
            });

        this.plot = $("<div/>").css({flex: "1 1 auto", minHeight: "0"}).appendTo(root)[0];

        if (window.ResizeObserver) {
            this._ro = new ResizeObserver(() => {
                if (this._plotted && window.Plotly) window.Plotly.Plots.resize(this.plot);
            });
            this._ro.observe(this.plot);
        }

        this.feed.on(payload => this._on_feed(payload));
    }

    _blank() {
        return {t: [], cn0: [], cn0_kcoh: [], q: [], beam_amp: [], beam_ph: [], sig: [],
                coh_s: [], dop: [], snr: []};
    }

    // Append one per-signal metrics sample to a history series, with dedup + ring trim.
    _push(key, src, now, snr) {
        if (!this.hist.has(key)) this.hist.set(key, this._blank());
        const h = this.hist.get(key);
        const n = h.t.length;
        if (n && h.sig[n - 1] === src.sig && h.coh_s[n - 1] === src.coh_s
            && h.dop[n - 1] === src.dop && h.snr[n - 1] === snr) return;
        // Do not replot an estimate that has not been recomputed. The broker reruns its C/N0
        // estimators every estimator_every_s (the prompt C/N0 every cycle on a chain that reads
        // the gather's reduced taps) and serves the last values in between, so successive polls
        // repeat one estimate. Identical floats from a floating-point estimator mean the same
        // computation, so a sample whose non-null cn0 and whose cn0_kcoh both repeat the last
        // stored values is dropped, for every observable, and the C/N0 traces show one point
        // per estimate.
        if (n && h.cn0[n - 1] === src.cn0 && h.cn0_kcoh[n - 1] === src.cn0_kcoh
            && src.cn0 != null) return;
        h.t.push(now);
        // Qualified like the coherent trace below: a null cn0_duty (the fallback estimator) has
        // no duty to judge and passes; a noise probe or a duty below CN0_MIN_DUTY is a gap.
        h.cn0.push((src.cn0 != null && !src.noise_probe
                    && (src.cn0_duty == null || src.cn0_duty >= CN0_MIN_DUTY))
                   ? src.cn0 : null);
        h.sig.push(src.sig);
        // A coherent point is plotted only when the fold cohered (eta >= n_rec/4). Below that
        // the records did not add coherently and the value is the noise floor, not a C/N0.
        // The table greys the same case; a history plot has no room for a qualifier, so it is
        // a gap.
        h.cn0_kcoh.push((src.cn0_kcoh != null && src.kcoh_eta != null && src.kcoh_n
                         && src.kcoh_eta >= 0.25 * src.kcoh_n) ? src.cn0_kcoh : null);
        h.q.push(src.q != null ? src.q : null);
        h.beam_amp.push(src.beam_amp != null ? src.beam_amp : null);
        h.beam_ph.push(src.beam_ph != null ? src.beam_ph : null);
        h.coh_s.push(src.coh_s); h.dop.push(src.dop); h.snr.push(snr);
        if (h.t.length > MAX_PTS) for (const k of Object.keys(h)) h[k].shift();
        // Mark dirty when this series belongs to the selected satellite. Keys are
        // "G11|<signal>" while `selected` is the satellite "G11", so compare the satellite
        // part; without the dirty flag _redraw never calls Plotly.react after the newPlot.
        if (key.split("|")[0] === this.selected) this._dirty = true;
    }

    _on_feed({sats, signals}) {
        const now = new Date();
        for (const r of sats || []) {
            // One history series per (satellite, signal).
            for (const sg of signals || []) {
                if (sg.tag !== r.tag) continue;
                const m = r.sig_by && r.sig_by[sg.key];
                if (!m || !(m.sig > 0 || m.amp > 0 || m.cn0 != null)) continue;
                const key = r.id + "|" + sg.key;
                this.meta.set(key, {col: sg.col});
                // Search SNR is per signal (gps_feed keeps each detection's snr on
                // sig_by[key]). A signal with no search stage has none: a gap, not a zero.
                this._push(key, m, now, m.snr != null ? m.snr : null);
            }
        }
        this._sync_dropdown();
        this._redraw();
    }

    // Keep the dropdown in sync with the series we've seen (sorted), preserving the selection.
    // Keys are "G11|gps_l5"; sort by constellation then PRN.
    _sync_dropdown() {
        // The selector picks a satellite (G11); the plot overlays a trace per signal it
        // carries. So dedup the sat|signal history keys down to distinct sat ids.
        const sat_of = k => k.split("|")[0];
        const ids = [...new Set([...this.hist.keys()].map(sat_of))];
        const prns = ids.sort((a, b) =>
            a[0] !== b[0] ? a[0].localeCompare(b[0])
                          : (parseInt(a.slice(1)) || 0) - (parseInt(b.slice(1)) || 0));
        const label = k => k;
        const cur = this.sel.val();
        const have = new Set();
        this.sel.find("option").each(function () { have.add($(this).val()); });
        let changed = false;
        for (const p of prns) if (!have.has(String(p))) changed = true;
        if (changed) {
            this.sel.empty();
            for (const p of prns)
                this.sel.append($("<option/>").attr("value", p).text(label(p)));
            if (this.selected != null && this._has(this.selected))
                this.sel.val(String(this.selected));
            else if (prns.length) {
                const best = prns.reduce((m, p) => this._len(p) > this._len(m) ? p : m, prns[0]);
                this.selected = best;
                this.sel.val(String(best));
                this._plotted = false;
                this._dirty = true;
            }
        }
        if (cur && this.sel.val() !== cur && this._has(cur))
            this.sel.val(cur);
    }

    // A selection ("G11") maps to its history series: every signal key "G11|...".
    _series_keys(id) {
        return [...this.hist.keys()].filter(k => k.split("|")[0] === id);
    }
    _has(id) { return this._series_keys(id).length > 0; }
    _len(id) { return this._series_keys(id).reduce((n, k) => n + this.hist.get(k).t.length, 0); }

    // Overlay one trace per signal the selected satellite carries (CM vs CL vs L5... on one
    // axis). Colour and legend by signal; the observable dropdown applies to all of them.
    _redraw() {
        if (this.selected == null || !window.Plotly) return;
        const M = MODES[this.mode];
        const keys = this._series_keys(this.selected);
        const traces = [];
        const now_bits = [];
        keys.forEach((k, i) => {
            const h = this.hist.get(k); if (!h) return;
            const col = (this.meta.get(k) || {}).col || k;
            const color = SIG_PALETTE[i % SIG_PALETTE.length];
            const V = h[this.mode];
            const off = this._hidden.has(col);
            traces.push({
                x: h.t.slice(), y: V.map(v => (v == null ? null : v)),
                type: "scatter", mode: "markers", name: col,
                marker: {size: 4, color},
                // "legendonly" keeps the trace in the legend but out of the plot AND out of
                // the y-axis autoscale -- which is the point of hiding it.
                visible: off ? "legendonly" : true,
                hovertemplate: `${col} · %{x|%H:%M:%S}<br>%{y:${M.fmt}}${M.unit}<extra></extra>`,
            });
            // Latest non-null for the summary line. The label is the toggle: struck through
            // and greyed when hidden, so the control shows its own state.
            const style = off ? `color:#bbb;text-decoration:line-through` : `color:${color}`;
            const cell = v => `<span data-sig="${col}" title="click to show/hide ${col}" `
                            + `style="${style};cursor:pointer">${col}${v}</span>`;
            let shown = false;
            for (let j = V.length - 1; j >= 0; j--)
                if (V[j] != null) {
                    now_bits.push(cell(" " + V[j].toFixed(this.mode === "coh_s" ? 2 : 1)));
                    shown = true; break;
                }
            // A hidden signal still gets its label, or it could never be turned back on.
            if (!shown && off) now_bits.push(cell(""));
        });
        this._info.html(now_bits.length
            ? `${this.selected} · ${M.label}: ${now_bits.join(" · ")}${M.unit}`
            : `<span style="color:#aaa">${this.selected}: no ${M.label} yet…</span>`);
        const layout = {
            margin: {l: 56, r: 10, t: 6, b: 28},
            xaxis: {type: "date", tickfont: {size: 10}, gridcolor: "#eee"},
            yaxis: {title: {text: M.axis, font: {size: 11}},
                    rangemode: M.zero ? "tozero" : "normal", tickfont: {size: 10},
                    gridcolor: "#eee", zeroline: !!M.zero},
            showlegend: true, legend: {orientation: "h", y: 1.02, yanchor: "bottom",
                                       font: {size: 10}},
            paper_bgcolor: "white", plot_bgcolor: "white",
        };
        if (!traces.length) return;
        if (!this._plotted) {
            window.Plotly.newPlot(this.plot, traces, layout,
                {displayModeBar: false, responsive: true});
            this._plotted = true;
        } else if (this._dirty) {
            window.Plotly.react(this.plot, traces, layout,
                {displayModeBar: false, responsive: true});
        }
        this._dirty = false;
    }
}
