// GNSS per-PRN time series: pick a satellite and an observable, watch it evolve.
//
// Observables (the dropdown), all straight off the shared GpsFeed so this panel and the
// table can never disagree:
//   * C/N₀      -- cn0_prompt_db (task #57): per-record prompt power, q-gated on the noise
//                  probes' q population, probe-debiased. THE radiometry; supersedes the old
//                  coh/inc pair (inc was rectification-biased, coh rode the deep fold's
//                  per-integration rate re-search and its ~20 dB paired self-scatter).
//   * beam A/φ  -- the per-element complex-gain summary (#57 step 2): median live-element
//                  amplitude (1 = uniform array) and the circular RMS of the element
//                  phases about the array mean (the peel coefficient's dispersion). The
//                  full per-element vectors ride /get_elements and the NFS archive.
//   * sig       -- combined significance (σ above noise): deep when floor-certified, else
//                  the moment-debiased incoherent amp_snr. The lock metric.
//   * coh       -- the auto-ladder's winning coherent window (s). Its collapses ARE the
//                  carrier-coherence story; 0 = no rung beat its rectification floor.
//   * dop       -- tracked Doppler (Hz). Should be a smooth orbital curve; steps are seed
//                  re-anchors (staleness releases / escapes), the churn signature.
//   * snr       -- SEARCH detection significance (acquire grid), independent of tracking.
//
// POINTS, NOT LINES: emits are irregular and gapped (a dropout is real information), so
// connecting them draws structure that isn't in the data.
//
// History is buffered in-browser only (tiny, not served forward, fresh on reload).

const MAX_PTS = 10000;      // per-PRN cap (~4 h at 1.5 s); bounds RENDER cost, not RAM
const LN10_10 = 10 / Math.LN10;   // 10/ln10 = 4.3429; d(dB)/dx = LN10_10 / x

// key -> {label, unit, fmt, zero (y-axis rangemode tozero)}
// ONE C/N0 (task #57, 2026-08-15): the coh/inc pair is retired from display. `cn0` is now
// cn0_prompt_db off the feed -- per-record prompt power, q-gated, probe-debiased -- which
// supersedes both (the incoherent one was rectification-biased, the coherent one carried
// the deep fold's ~20 dB of paired self-scatter). The beam modes are #57 step 2: the
// per-element complex-gain summary (median live-element amplitude; circular RMS of the
// element phases about the array mean -- the peel coefficient's dispersion).
// A cn0_prompt point below this lock duty is dropped from the trace rather than drawn: the
// estimator still publishes at very low duty (measured 0.08-0.14 across four rows with the
// band completely dead on 2026-08-26 -- two of them NOISE PROBES), and one unqualified point
// in a history trace is indistinguishable from a real one. Mirrors the eta qualifier the
// coherent trace already applies below, and the 0.7 dimming the table already applies.
const CN0_MIN_DUTY = 0.25;

const MODES = {
    // ⚠️ NOT "inc" any more: this is cn0_prompt_db (per-record prompt power, q-gated,
    // probe-debiased), which RETIRED the incoherent/coherent pair. The old label outlived
    // the old estimator and had people reading a rectification-biased number that is no
    // longer computed.
    cn0:      {label: "C/N₀ prompt", axis: "C/N₀ prompt, per-record (dB-Hz)",
               unit: " dB-Hz", fmt: ".1f"},
    cn0_kcoh: {label: "C/N₀ coh", axis: "C/N₀ coherent, known-rate fold (dB-Hz)",
               unit: " dB-Hz", fmt: ".1f"},
    beam_amp: {label: "beam A",  axis: "median element gain (×, 1 = uniform)", unit: "×",
               fmt: ".2f", zero: true},
    beam_ph:  {label: "beam φ",  axis: "element phase spread (deg RMS)", unit: "°",
               fmt: ".0f", zero: true},
    // q -- THE CODE-LOOP QUALITY over time (2026-08-18). The fleet discriminator's
    // prompt-vs-shoulder ratio, i.e. "is the code loop on the peak". This is the series
    // that would have shown #87's stale-trim regression the moment it started: gal_e5b's
    // q fell 2.09 -> 1.09 and sat there for 3.5 h while every C/N0 trace looked normal.
    // Judge lock HERE, not on sig/C-N0 -- the deep fold re-searches and will certify a
    // satellite whose prompt tap is on noise (#47). ~2.2 is the working lock bar.
    q:       {label: "q",        axis: "code-loop quality (fleet discriminator)", unit: "",
              fmt: ".2f", zero: true},
    sig:     {label: "sig",      axis: "significance (σ)", unit: "σ", fmt: ".1f", zero: true},
    coh_s:   {label: "coh",      axis: "coherent window (s)", unit: " s", fmt: ".3f", zero: true},
    dop:     {label: "dop",      axis: "tracked Doppler (Hz)", unit: " Hz", fmt: ".1f"},
    snr:     {label: "snr",      axis: "search significance (σ)", unit: "σ", fmt: ".1f", zero: true},
};
const MODE_ORDER = ["cn0", "cn0_kcoh", "q", "beam_amp", "beam_ph", "sig", "coh_s", "dop",
                    "snr"];

// A distinct-per-trace palette. Colours are by POSITION among a sat's signals, not fixed per
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
        // Signals the user has toggled OFF, by signal column ("E5a-Q", "E5b-Q", ...).
        // Kept by SIGNAL rather than by series key so the choice survives changing PRN --
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
        // ⚠️ DO NOT REPLOT AN ESTIMATE THAT HAS NOT BEEN RECOMPUTED. The C/N0 estimators
        // run on their own cadence (--estimator-every-s) and the broker keeps serving the
        // last value in between, so at a 1.5 s poll one estimate becomes ~26 identical
        // points: the history drew flat plateaus with cliffs between them and looked like
        // a quantised instrument. Identical floats out of a floating-point estimator mean
        // "same computation", so dropping the repeats is safe and the plot then shows one
        // point per actual measurement.
        if (n && h.cn0[n - 1] === src.cn0 && h.cn0_kcoh[n - 1] === src.cn0_kcoh
            && src.cn0 != null) return;
        h.t.push(now);
        // ⚠️ QUALIFIED, for the same reason the coherent trace below is. A null cn0_duty is
        // a pre-#57 broker and passes (there is no duty to judge it by); anything else must
        // earn its point.
        h.cn0.push((src.cn0 != null && !src.noise_probe
                    && (src.cn0_duty == null || src.cn0_duty >= CN0_MIN_DUTY))
                   ? src.cn0 : null);
        h.sig.push(src.sig);
        // ⚠️ ONLY PLOT A COHERENT POINT THE FOLD ACTUALLY EARNED. eta well below n_rec
        // means the records did not add coherently, so the value is the noise floor
        // (~0 dB-Hz), not a C/N0 -- E19 through boresight drew a whole trace of those on
        // 2026-08-15 and it read as "the coherent estimator is broken" when what it was
        // reporting was "there is no coherence here". The table greys the same case; a
        // history plot has no room for a qualifier, so the honest rendering is a GAP.
        h.cn0_kcoh.push((src.cn0_kcoh != null && src.kcoh_eta != null && src.kcoh_n
                         && src.kcoh_eta >= 0.25 * src.kcoh_n) ? src.cn0_kcoh : null);
        h.q.push(src.q != null ? src.q : null);
        h.beam_amp.push(src.beam_amp != null ? src.beam_amp : null);
        h.beam_ph.push(src.beam_ph != null ? src.beam_ph : null);
        h.coh_s.push(src.coh_s); h.dop.push(src.dop); h.snr.push(snr);
        if (h.t.length > MAX_PTS) for (const k of Object.keys(h)) h[k].shift();
        // Mark dirty when this series belongs to the SELECTED satellite. Keys are
        // "G11|<signal>" while `selected` is the sat "G11", so compare the sat part -- else
        // the plot renders once (newPlot) and never react()s again (static after frame 1).
        if (key.split("|")[0] === this.selected) this._dirty = true;
    }

    _on_feed({sats, signals}) {
        const now = new Date();
        for (const r of sats || []) {
            // One history series per (satellite, signal) -- a sat carries several.
            for (const sg of signals || []) {
                if (sg.tag !== r.tag) continue;
                const m = r.sig_by && r.sig_by[sg.key];
                if (!m || !(m.sig > 0 || m.amp > 0 || m.cn0 != null)) continue;
                const key = r.id + "|" + sg.key;
                this.meta.set(key, {col: sg.col});
                // Search SNR IS per-signal: gps_feed stashes each detection's snr at
                // sig_by[key].snr and carries it onto the metrics object, precisely so
                // "strong in search, wrong downstream" stays visible. (undefined for a signal
                // with no blind acquisition at all, e.g. E5a/B2a, which is a genuine gap, not
                // a zero.)
                this._push(key, m, now, m.snr != null ? m.snr : null);
            }
        }
        this._sync_dropdown();
        this._redraw();
    }

    // Keep the dropdown in sync with the series we've seen (sorted), preserving the selection.
    // Keys are "G11|gps_l5"; sort by constellation then PRN.
    _sync_dropdown() {
        // The selector picks a SATELLITE (G11); the plot overlays a trace per signal it
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
    // axis) -- the shared-knowledge comparison. Colour + legend by signal; the "thing to plot"
    // dropdown picks the observable for all of them.
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
            // latest non-null for the summary line. The label is the toggle: struck through
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
