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

export class GpsAmpHistoryPanel {
    constructor({app, target, feed}) {
        this.app = app;
        this.feed = feed;
        this.hist = new Map();   // prn -> {t:[], cn0_coh:[], cn0:[], sig:[], coh_s:[], dop:[], snr:[], dr:[]}
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

    _on_feed({sats}) {
        const now = new Date();
        for (const r of sats || []) {
            // No despread this emit -> a GAP, not a zero. (A tracked-but-silent sat still
            // reports sig 0 with a real dop; require SOME signal-side content.)
            if (!(r.sig > 0 || r.amp > 0 || r.cn0 != null || r.snr != null)) continue;
            if (r.dr > this._maxDr) this._maxDr = r.dr;
            if (!this.hist.has(r.id)) this.hist.set(r.id, this._blank());
            const h = this.hist.get(r.id);
            const n = h.t.length;
            // Dedup: the poll runs faster than the combiner emits, so consecutive ticks
            // catch the SAME record -> identical floats. Advance once per real emit.
            if (n && h.sig[n - 1] === r.sig && h.coh_s[n - 1] === r.coh_s
                && h.dop[n - 1] === r.dop && h.snr[n - 1] === r.snr) continue;
            h.t.push(now);
            h.cn0_coh.push(r.cn0_coh);   // null where the coherence wasn't certified
            h.cn0.push(r.cn0);
            h.sig.push(r.sig);
            h.coh_s.push(r.coh_s);
            h.dop.push(r.dop);
            h.snr.push(r.snr);
            h.dr.push(r.dr || 0);
            h.peel.push(r.peel_db);      // null on chains that don't peel -> a GAP
            h.bound.push(!!r.peel_bound);
            if (h.t.length > MAX_PTS)
                for (const k of Object.keys(h)) h[k].shift();
            if (r.id === this.selected) this._dirty = true;
        }
        this._sync_dropdown();
        this._redraw();
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
            if (this.selected != null && this.hist.has(this.selected))
                this.sel.val(String(this.selected));
            else if (prns.length) {
                const best = prns.reduce((m, p) =>
                    this.hist.get(p).t.length > this.hist.get(m).t.length ? p : m, prns[0]);
                this.selected = best;
                this.sel.val(String(best));
                this._plotted = false;
                this._dirty = true;
            }
        }
        if (cur && this.sel.val() !== cur && this.hist.has(cur))
            this.sel.val(cur);
    }

    // 1σ error bar for sample i, where the observable has one worth drawing.
    _err(h, i) {
        if (this.mode !== "cn0_coh") return 0;
        // significance has ~unit std -> d(dB)/d(sig) = (10/ln10)*(2/sig)
        const sig = h.sig[i];
        return sig > 0 ? (2 * LN10_10) / sig : 0;
    }

    _redraw() {
        if (this.selected == null || !window.Plotly) return;
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
            const band = (this.app && this.app.gps_band) || "l1";
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
