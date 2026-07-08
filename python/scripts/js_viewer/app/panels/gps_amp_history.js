// GPS deep-amplitude history: Â(t) with error bars for a PRN picked from a dropdown.
//
// Polls <combiner>/get_status on its own timer and BUFFERS (time, Â, σ) per PRN in the
// browser only -- a tiny volume (one point per ~combiner emit) even over a long run, and it
// is not served forward (a page reload starts fresh). The plotted quantity is the deep
// nav-wiped amplitude Â = deep_amplitude, and its 1σ error bar is deep_amplitude/deep_snr
// (deep_snr = |Â| / its uncertainty, so Â/deep_snr IS that uncertainty). Where the deep isn't
// available it falls back to the moment-debiased unbiased_amplitude / amp_snr.

const POLL_MS = 1500;
const MAX_PTS = 10000;      // per-PRN cap (~4 h at 1.5 s); drops oldest past this so the live
                            // re-render stays snappy. Data is tiny; this bounds RENDER cost, not RAM.

export class GpsAmpHistoryPanel {
    constructor({app, target, combiner_stage}) {
        this.app = app;
        this.combiner_stage = combiner_stage || "combiner";
        this.hist = new Map();       // prn -> {t:[Date], a:[Â], s:[σ]}
        this.selected = null;
        this._inflight = false;
        this._dirty = false;         // new data for the selected PRN since last redraw
        this._plotted = false;

        const root = $("#" + target);
        root.css({display: "flex", flexDirection: "column", width: "100%", height: "100%",
                  boxSizing: "border-box", fontFamily: "sans-serif", fontSize: "12px"});

        // Controls: PRN dropdown + a live readout + clear.
        const bar = $("<div/>").css({flex: "0 0 auto", padding: "2px 4px 4px",
            display: "flex", alignItems: "center", gap: "8px"}).appendTo(root);
        $("<span/>").text("PRN").css({color: "#444"}).appendTo(bar);
        this.sel = $("<select/>").css({fontSize: "12px"}).appendTo(bar);
        this.sel.on("change", () => {
            this.selected = parseInt(this.sel.val(), 10);
            this._plotted = false;   // force a fresh newPlot (axis rescale) on PRN switch
            this._redraw();
        });
        this._info = $("<span/>").css({color: "#666", marginLeft: "4px"}).appendTo(bar);
        $("<button/>").text("clear").css({fontSize: "11px", marginLeft: "auto"})
            .appendTo(bar).on("click", () => {
                if (this.selected != null && this.hist.has(this.selected))
                    this.hist.set(this.selected, {t: [], a: [], s: []});
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
        k.stageGet(this.combiner_stage, "get_status")
            .then(r => r.ok ? r.json() : null).catch(() => null)
            .then(status => {
                this._inflight = false;
                if (!Array.isArray(status)) return;
                const now = new Date();
                for (const c of status) {
                    const prn = c.prn;
                    if (!prn) continue;
                    const A = c.deep_amplitude || c.unbiased_amplitude || 0;
                    const sig = Math.max(c.deep_snr || 0, c.amp_snr || 0);
                    if (A === 0 && sig === 0) continue;   // no despread this emit -> a gap, not a 0
                    const sigma = sig > 0 ? A / sig : 0;
                    if (!this.hist.has(prn)) this.hist.set(prn, {t: [], a: [], s: []});
                    const h = this.hist.get(prn);
                    const n = h.a.length;
                    // Dedup: consecutive polls can catch the SAME combiner emit -> identical
                    // floats. Skip those so the trace advances once per real emit.
                    if (n && h.a[n - 1] === A && h.s[n - 1] === sigma) continue;
                    h.t.push(now); h.a.push(A); h.s.push(sigma);
                    if (h.t.length > MAX_PTS) { h.t.shift(); h.a.shift(); h.s.shift(); }
                    if (prn === this.selected) this._dirty = true;
                }
                this._sync_dropdown();
                this._redraw();
            });
    }

    // Keep the dropdown in sync with the PRNs we've seen (sorted), preserving the selection.
    _sync_dropdown() {
        const prns = [...this.hist.keys()].sort((a, b) => a - b);
        const cur = this.sel.val();
        const have = new Set();
        this.sel.find("option").each(function () { have.add($(this).val()); });
        let changed = false;
        for (const p of prns) if (!have.has(String(p))) changed = true;
        if (changed) {
            this.sel.empty();
            for (const p of prns)
                this.sel.append($("<option/>").attr("value", p).text("PRN " + p));
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
        if (cur && this.sel.val() !== cur && this.hist.has(parseInt(cur, 10)))
            this.sel.val(cur);
    }

    _redraw() {
        if (this.selected == null || !window.Plotly) return;
        const h = this.hist.get(this.selected);
        if (!h) return;
        // Live readout: latest Â ± σ and its significance.
        const n = h.a.length;
        if (n) {
            const A = h.a[n - 1], s = h.s[n - 1];
            const sig = s > 0 ? A / s : 0;
            this._info.html(`Â = <b>${A.toFixed(3)}</b> ± ${s.toFixed(3)}`
                + `&nbsp;(${sig.toFixed(1)}σ)&nbsp;· ${n} pts`);
        } else {
            this._info.text("(no samples yet)");
        }
        // FRESH array references each render: Plotly.react diffs data arrays BY REFERENCE, so
        // handing it the same in-place-mutated h.* arrays makes it treat the trace as unchanged and
        // skip the redraw (the plot then only refreshes on a full newPlot, e.g. a PRN swap). Slicing
        // gives new references so the live append actually paints.
        const T = h.t.slice(), Y = h.a.slice(), S = h.s.slice();
        const trace = {
            x: T, y: Y, customdata: S, type: "scatter", mode: "lines+markers",
            line: {color: "#1f77b4", width: 1},
            marker: {size: 3, color: "#1f77b4"},
            error_y: {type: "data", array: S, visible: true,
                      color: "rgba(31,119,180,0.35)", thickness: 1, width: 0},
            hovertemplate: "%{x|%H:%M:%S}<br>Â %{y:.3f} ± %{customdata:.3f}<extra></extra>",
        };
        const layout = {
            margin: {l: 48, r: 10, t: 6, b: 28},
            xaxis: {type: "date", tickfont: {size: 10}, gridcolor: "#eee"},
            yaxis: {title: {text: "Â  (deep |A|)", font: {size: 11}},
                    rangemode: "tozero", tickfont: {size: 10}, gridcolor: "#eee", zeroline: true},
            showlegend: false, paper_bgcolor: "white", plot_bgcolor: "white",
        };
        if (!this._plotted) {
            window.Plotly.newPlot(this.plot, [trace], layout,
                {displayModeBar: false, responsive: true});
            this._plotted = true;
        } else if (this._dirty) {
            window.Plotly.react(this.plot, [trace], layout,
                {displayModeBar: false, responsive: true});
        }
        this._dirty = false;
    }
}
