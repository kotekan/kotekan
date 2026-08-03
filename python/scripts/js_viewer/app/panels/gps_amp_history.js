// GNSS per-PRN time series: pick a satellite and an observable, watch it evolve.
//
// Observables (the dropdown), all straight off the shared GpsFeed so this panel and the
// table can never disagree:
//   * C/N₀ coh  -- 10 log10(deep_snr² / T_coh) dB-Hz: the DEEP estimator. Gain-independent
//                  and √T-deep, but defined only where the combiner certified the coherence
//                  (coherence_s > 0); a floored deep is nav-wipe rectification, not signal,
//                  so those emits are GAPS here rather than points.
//   * C/N₀ inc  -- 10 log10(x / t_rec) dB-Hz from the single-record power ratio
//                  x = u²/(a²-u²): needs no coherence at all, so it survives where the
//                  coherent estimator drops out. The pair together is the diagnostic --
//                  they should agree, and the map (gps_cn0_map.py) plots both for that reason.
//   * sig       -- combined significance (σ above noise): deep when floor-certified, else
//                  the moment-debiased incoherent amp_snr. The lock metric.
//   * coh       -- the auto-ladder's winning coherent window (s). Its collapses ARE the
//                  carrier-coherence story; 0 = no rung beat its rectification floor.
//   * dop       -- tracked Doppler (Hz). Should be a smooth orbital curve; steps are seed
//                  re-anchors (staleness releases / escapes), the churn signature.
//   * snr       -- SEARCH detection significance (acquire grid), independent of tracking.
//   * peel      -- fused voltage-peel depth (dB) = 20 log10(deep / peel_deep). OPEN ORANGE
//                  TRIANGLES are BOUNDS, not measurements: once the residual reaches the
//                  combiner's detection floor the ratio stops measuring the peel and starts
//                  measuring the floor, so those points read "≥ x dB" (an unbounded division
//                  reported 61 dB on 2026-07-24 -- every way a depth measurement fails makes
//                  the peel look BETTER). Filled blue = a real measurement above the floor.
//                  This series is the one to watch over TIME: the peel oscillates with the
//                  30 s GPS frame (constructed bits cover subframes 1-3 only), and a log line
//                  sampling at 9.984 s aliases that at 3.005x into a flat-looking collapse.
//
// POINTS, NOT LINES: emits are irregular and gapped (a dropout is real information), so
// connecting them draws structure that isn't in the data.
//
// History is buffered in-browser only (tiny, not served forward, fresh on reload).

const MAX_PTS = 10000;      // per-PRN cap (~4 h at 1.5 s); bounds RENDER cost, not RAM
const LN10_10 = 10 / Math.LN10;   // 10/ln10 = 4.3429; d(dB)/dx = LN10_10 / x

// key -> {label, unit, fmt, zero (y-axis rangemode tozero), errs (1σ bars)}
const MODES = {
    cn0_coh: {label: "C/N₀ coh", axis: "C/N₀ coherent (dB-Hz)", unit: " dB-Hz", fmt: ".1f",
              errs: true},
    cn0:     {label: "C/N₀ inc", axis: "C/N₀ incoherent (dB-Hz)", unit: " dB-Hz", fmt: ".1f"},
    sig:     {label: "sig",      axis: "significance (σ)", unit: "σ", fmt: ".1f", zero: true},
    coh_s:   {label: "coh",      axis: "coherent window (s)", unit: " s", fmt: ".3f", zero: true},
    dop:     {label: "dop",      axis: "tracked Doppler (Hz)", unit: " Hz", fmt: ".1f"},
    snr:     {label: "snr",      axis: "search significance (σ)", unit: "σ", fmt: ".1f", zero: true},
    peel:    {label: "peel",     axis: "voltage-peel depth (dB)", unit: " dB", fmt: ".1f",
              zero: true},
};
const MODE_ORDER = ["cn0_coh", "cn0", "sig", "coh_s", "dop", "snr", "peel"];

// ICD minimum received power per TRACKED COMPONENT, as a C/N₀ floor with
// N₀ = −204 dBW/Hz (290 K, lossless front end): C/N₀_min = P_min(dBW) + 204.
// Component powers: IS-GPS-200 (L1 C/A −158.5; L2C composite −160 → CM half −163),
// IS-GPS-705 (Q5 −157.9), Galileo OS SIS ICD (E1 total −157.25 → E1C pilot half
// −160.3; E5a total −155.25 → Q pilot half −158.3), BDS-3 ICDs, MEO values (B1C
// total −158.5 → pilot 3/4 −159.75; B2a total −155.5 → pilot half −158.5; IGSO
// sats spec 1.8 dB lower). A healthy strong sat should sit AT or ABOVE the line
// (specs are minima at 5° elevation); sitting 10+ dB under it at high elevation
// is a chain pathology, not geometry.
const CN0_BASELINE = {
    l1:  {G: 45.5, E: 43.7, C: 44.2},
    l2c: {G: 41.0},
    l5:  {G: 46.1, E: 45.7, C: 45.5},
};

// Unified overlay: signal draw order (frequency-ascending, pilot after its data sibling) and a
// distinct-per-trace palette. Colours are by POSITION among a sat's signals, not fixed per
// signal, so any constellation's 2-5 signals stay visually separable.
const SIG_ORDER = {CA: 0, L1C: 1, E1C: 2, B1C: 3, B2b: 4, Q: 6, E5a: 7, B2a: 8};
const SIG_PALETTE = ["#4d9de0", "#6fbf73", "#e8923c", "#c65d21", "#d64550", "#8e6fb0", "#39a0a0"];

export class GpsAmpHistoryPanel {
    constructor({app, target, feed}) {
        this.app = app;
        this.feed = feed;
        this.hist = new Map();   // key -> {t:[], cn0_coh:[], cn0:[], sig:[], coh_s:[], dop:[], snr:[], dr:[]}
        this.meta = new Map();   // key -> {label, band, tag} (unified: one entry per sat+signal)
        this.unified = false;
        this.selected = null;
        this.mode = "cn0_coh";
        this._dirty = false;
        this._plotted = false;
        this._maxDr = 0;         // largest deep_records seen = the converged (~1 s) window

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
        // `bound` parallels `peel`: true where the residual sat at the combiner's detection
        // floor, i.e. the point is a LOWER BOUND. Kept as its own array so the plot can never
        // render a bound as a measurement (see the header note).
        return {t: [], cn0_coh: [], cn0: [], sig: [], coh_s: [], dop: [], snr: [], dr: [],
                peel: [], bound: []};
    }

    // L1/L2/L5 (unified signal band) -> the CN0_BASELINE band key.
    static BAND_KEY = {L1: "l1", L2: "l2c", L5: "l5"};

    // Append one (metrics) sample to a history series, with dedup + ring trim. `src` is either
    // a flat row (r.snr present) or a per-signal metrics object (r.deep_snr, snr absent).
    _push(key, src, now, snr) {
        if (src.dr > this._maxDr) this._maxDr = src.dr;
        if (!this.hist.has(key)) this.hist.set(key, this._blank());
        const h = this.hist.get(key);
        const n = h.t.length;
        if (n && h.sig[n - 1] === src.sig && h.coh_s[n - 1] === src.coh_s
            && h.dop[n - 1] === src.dop && h.snr[n - 1] === snr) return;
        h.t.push(now);
        h.cn0_coh.push(src.cn0_coh); h.cn0.push(src.cn0); h.sig.push(src.sig);
        h.coh_s.push(src.coh_s); h.dop.push(src.dop); h.snr.push(snr);
        h.dr.push(src.dr || 0);
        h.peel.push(src.peel_db); h.bound.push(!!src.peel_bound);
        if (h.t.length > MAX_PTS) for (const k of Object.keys(h)) h[k].shift();
        // Mark dirty when this series belongs to the SELECTED satellite. Unified keys are
        // "G11|<signal>" while `selected` is the sat "G11", so compare the sat part -- else
        // the plot renders once (newPlot) and never react()s again (static after frame 1).
        if ((this.unified ? key.split("|")[0] : key) === this.selected) this._dirty = true;
    }

    _on_feed({sats, unified, signals}) {
        this.unified = !!unified;
        const now = new Date();
        for (const r of sats || []) {
            if (this.unified) {
                // One history series per (satellite, signal) -- a sat carries several.
                for (const sg of signals || []) {
                    if (sg.tag !== r.tag) continue;
                    const m = r.sig_by && r.sig_by[sg.key];
                    if (!m || !(m.sig > 0 || m.amp > 0 || m.cn0 != null)) continue;
                    const key = r.id + "|" + sg.key;
                    this.meta.set(key, {label: r.id + " · " + sg.col, col: sg.col,
                                        band: GpsAmpHistoryPanel.BAND_KEY[sg.band] || "l1",
                                        tag: r.tag});
                    this._push(key, m, now, null);   // per-signal search SNR n/a
                }
            } else {
                // No despread this emit -> a GAP, not a zero. (A tracked-but-silent sat
                // still reports sig 0 with a real dop; require SOME signal-side content.)
                if (!(r.sig > 0 || r.amp > 0 || r.cn0 != null || r.snr != null)) continue;
                this._push(r.id, r, now, r.snr);
            }
        }
        this._sync_dropdown();
        this._redraw();
    }

    // Keep the dropdown in sync with the series we've seen (sorted), preserving the selection.
    // Unified keys are "G11|l2c_cl_combiner"; sort by constellation, PRN, then signal so a
    // satellite's signals cluster (G11 · CA, G11 · CM, G11 · CL, ...).
    _sync_dropdown() {
        // Unified: the selector picks a SATELLITE (G11); the plot overlays a trace per signal
        // it carries. So dedup the sat|signal history keys down to distinct sat ids. Flat: the
        // keys ARE sat ids.
        const sat_of = k => k.split("|")[0];
        const ids = this.unified
            ? [...new Set([...this.hist.keys()].map(sat_of))]
            : [...this.hist.keys()];
        const prns = ids.sort((a, b) =>
            a[0] !== b[0] ? a[0].localeCompare(b[0])
                          : (parseInt(a.slice(1)) || 0) - (parseInt(b.slice(1)) || 0));
        const label = k => k;   // sat id (unified) or sat id (flat) -- both plain
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

    // A selection ("G11") maps to its history series: unified = every signal key "G11|...";
    // flat = the one key "G11".
    _series_keys(id) {
        if (!this.unified) return this.hist.has(id) ? [id] : [];
        return [...this.hist.keys()].filter(k => k.split("|")[0] === id);
    }
    _has(id) { return this._series_keys(id).length > 0; }
    _len(id) { return this._series_keys(id).reduce((n, k) => n + this.hist.get(k).t.length, 0); }

    // 1σ error bar for sample i, where the observable has one worth drawing.
    _err(h, i) {
        if (this.mode !== "cn0_coh") return 0;
        // significance has ~unit std -> d(dB)/d(sig) = (10/ln10)*(2/sig)
        const sig = h.sig[i];
        return sig > 0 ? (2 * LN10_10) / sig : 0;
    }

    // Unified: overlay one trace per signal the selected satellite carries (CM vs CL vs L5...
    // on one axis) -- the shared-knowledge comparison. Colour + legend by signal; the "thing
    // to plot" dropdown picks the observable for all of them.
    _redraw_unified() {
        const M = MODES[this.mode];
        const keys = this._series_keys(this.selected)
            .sort((a, b) => (SIG_ORDER[(this.meta.get(a) || {}).col] || 99)
                          - (SIG_ORDER[(this.meta.get(b) || {}).col] || 99));
        const traces = [];
        const now_bits = [];
        keys.forEach((k, i) => {
            const h = this.hist.get(k); if (!h) return;
            const col = (this.meta.get(k) || {}).col || k;
            const color = SIG_PALETTE[i % SIG_PALETTE.length];
            const V = h[this.mode];
            traces.push({
                x: h.t.slice(), y: V.map(v => (v == null ? null : v)),
                type: "scatter", mode: "markers", name: col,
                marker: {size: 4, color},
                hovertemplate: `${col} · %{x|%H:%M:%S}<br>%{y:${M.fmt}}${M.unit}<extra></extra>`,
            });
            // latest non-null for the summary line
            for (let j = V.length - 1; j >= 0; j--)
                if (V[j] != null) { now_bits.push(`<span style="color:${color}">${col} ${V[j].toFixed(this.mode === "coh_s" ? 2 : 1)}</span>`); break; }
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

    _redraw() {
        if (this.selected == null || !window.Plotly) return;
        if (this.unified) return this._redraw_unified();
        const h = this.hist.get(this.selected);
        if (!h) return;
        const M = MODES[this.mode];
        const V = h[this.mode];

        // Fresh arrays each render (Plotly.react diffs BY REFERENCE, so reusing the
        // in-place buffers would make it skip the redraw).
        //
        // De-emphasize only LOW-SIGNIFICANCE SHORT-window emits, and only for the
        // ladder-dependent observables: near the ~7σ nav-wipe floor the short rungs are
        // extreme-value-inflated (spurious up-spikes), but a HIGH-significance short
        // window is real signal -- the ladder legitimately rides 125-500 ms windows at
        // 50-160σ. The incoherent/search/Doppler observables don't come off the ladder
        // at all, so nothing is greyed there.
        // The SECOND trace is "this point is qualified, not plain": de-emphasized short-window
        // ladder emits for cn0_coh/sig, and floor-limited LOWER BOUNDS for peel. Same
        // mechanism, opposite emphasis -- a bound is not junk, it is a real statement about
        // the peel that simply cannot be read as an equality.
        const ladder = (this.mode === "cn0_coh" || this.mode === "sig");
        const isPeel = (this.mode === "peel");
        const thr = 0.6 * (this._maxDr || 0);
        const SIG_REAL = 20;
        const T = h.t.slice(), Yf = new Array(V.length), Ys = new Array(V.length),
              E = new Array(V.length);
        let lastFull = -1, lastAny = -1;
        for (let i = 0; i < V.length; i++) {
            E[i] = this._err(h, i);
            const v = (V[i] == null) ? null : V[i];
            const isAlt = isPeel
                ? !!h.bound[i]
                : (ladder && this._maxDr > 0 && (h.dr[i] || 0) < thr
                   && (h.sig[i] || 0) < SIG_REAL);
            Yf[i] = isAlt ? null : v;
            Ys[i] = isAlt ? v : null;
            if (!isAlt && v != null) lastFull = i;
            if (v != null) lastAny = i;
        }

        const n = V.length;
        // For peel, the newest point is worth reporting even when it is a BOUND -- "≥ 31 dB"
        // is the honest reading, and suppressing it as "no certified value" would hide the
        // deepest peels (the residual is at the floor precisely when the peel worked best).
        const iShow = (isPeel && lastAny > lastFull) ? lastAny : lastFull;
        if (iShow >= 0) {
            const v = V[iShow], e = E[iShow], sig = h.sig[iShow];
            const ge = (isPeel && h.bound[iShow]) ? "≥&nbsp;" : "";
            const tag = (iShow !== n - 1)
                ? (isPeel ? ' · <span style="color:#aaa">not peeling now</span>'
                          : ' · <span style="color:#aaa">short win / no value now</span>')
                : '';
            const err = M.errs && e ? ` ± ${e.toFixed(1)}` : "";
            this._info.html(`${M.label} = ${ge}<b>${v.toFixed(this.mode === "coh_s" ? 3 : 1)}</b>`
                + `${err}${M.unit}&nbsp;(${(sig || 0).toFixed(1)}σ)&nbsp;· ${n} pts` + tag);
        } else if (n) {
            this._info.html('<span style="color:#aaa">no certified value yet…</span>');
        } else {
            this._info.text("(no samples yet)");
        }

        // POINTS ONLY -- no connecting lines (see the header note).
        const full = {
            x: T, y: Yf, customdata: E, type: "scatter", mode: "markers",
            marker: {size: 4, color: "#1f77b4"},
            error_y: M.errs
                ? {type: "data", array: E, visible: true,
                   color: "rgba(31,119,180,0.35)", thickness: 1, width: 0}
                : {visible: false},
            hovertemplate: M.errs
                ? `%{x|%H:%M:%S}<br>%{y:${M.fmt}} ± %{customdata:${M.fmt}}${M.unit}<extra></extra>`
                : `%{x|%H:%M:%S}<br>%{y:${M.fmt}}${M.unit}<extra></extra>`,
        };
        // Bounds get their OWN glyph -- open orange triangles pointing up, the direction the
        // true value lies in. Greying them like a short-window emit would say "trust this
        // less"; the right message is "trust this, but only as an inequality".
        const shortWin = isPeel ? {
            x: T, y: Ys, type: "scatter", mode: "markers",
            marker: {size: 6, symbol: "triangle-up-open", color: "rgba(230,140,20,0.85)",
                     line: {width: 1}},
            hovertemplate: `%{x|%H:%M:%S}<br>≥ %{y:${M.fmt}}${M.unit} · residual at floor`
                + `<extra></extra>`,
        } : {
            x: T, y: Ys, type: "scatter", mode: "markers",
            marker: {size: 3, color: "rgba(150,150,150,0.45)"},
            hovertemplate: `%{x|%H:%M:%S}<br>%{y:${M.fmt}}${M.unit} · short window, low sig<extra></extra>`,
        };
        const layout = {
            margin: {l: 56, r: 10, t: 6, b: 28},
            xaxis: {type: "date", tickfont: {size: 10}, gridcolor: "#eee"},
            yaxis: {title: {text: M.axis, font: {size: 11}},
                    rangemode: M.zero ? "tozero" : "normal", tickfont: {size: 10},
                    gridcolor: "#eee", zeroline: !!M.zero},
            showlegend: false, paper_bgcolor: "white", plot_bgcolor: "white",
        };
        // Absolute reference: the ICD-minimum C/N₀ for this band + constellation
        // (dashed line, C/N₀ modes only). Baseline is per tracked component -- see
        // CN0_BASELINE for the power bookkeeping and sources.
        if (this.mode === "cn0_coh" || this.mode === "cn0") {
            // Per-signal band in unified mode (the selected key's meta), else the viewer's band.
            const band = this.unified
                ? ((this.meta.get(this.selected) || {}).band || "l1")
                : ((this.app && this.app.gps_band) || "l1");
            const sys = String(this.selected)[0];
            const base = (CN0_BASELINE[band] || {})[sys];
            if (base != null) {
                layout.shapes = [{type: "line", xref: "paper", x0: 0, x1: 1,
                                  yref: "y", y0: base, y1: base,
                                  line: {color: "rgba(200,60,60,0.55)", width: 1,
                                         dash: "dash"}}];
                layout.annotations = [{xref: "paper", x: 1, xanchor: "right",
                                       yref: "y", y: base, yanchor: "bottom",
                                       text: `ICD min ${base.toFixed(1)} dB-Hz (290 K)`,
                                       showarrow: false,
                                       font: {size: 9, color: "rgba(200,60,60,0.8)"}}];
            }
        }
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
