// Stream-health strip: ADC rms / rail % / sample-drop rate from <airspy>/adcstat, plus the
// PIPELINE drop rate from this band's Valve counters -- both already fetched by the shared
// GpsFeed (~1.5 s), so this panel adds ZERO kotekan load.
//
// TWO INDEPENDENT PLACES DATA IS LOST, and they mean different things:
//   drop (ADC)   -- libairspy's USB FIFO overflowed. The host didn't drain the dongle in
//                   time: CPU starvation on the input core, or a marginal USB link.
//   drop (valve) -- the Valve's output buffer was full, so it threw the frame away. The
//                   GPU chain missed real time. THIS IS THE SILENT ONE: the frame never
//                   reaches the tracker, sample_seq jumps, the ring zero-fills the gap,
//                   and every coherent window touching it shreds. It looks exactly like
//                   signal that decohered -- which is why the 2026-07-27 L5 peel spent a
//                   day being diagnosed as broken add-back arithmetic. Any peel/feature
//                   that adds GPU load must be watched HERE first.
//
// Drop rate needs the 2026-07-18 airspyInput counters (samples_total = true index
// incl. accounted gaps, samples_dropped = cumulative libairspy FIFO drops,
// drop_events). The instantaneous rate is computed browser-side over a rolling
// ~15 s window of counter deltas; the cumulative fraction over the whole run is
// shown beside it. Counters absent (older kotekan) -> the drop cells show "n/a".
//
// Color language: green = healthy, amber = attention, red = data being lost.
//   rms: amber < 20 (weak input / dead antenna?)   rail%: amber > 0.1, red > 1
//   drop: amber > 0 in-window, red > 1e-4 cumulative -- same thresholds both sources.

const WINDOW_S = 15;

export class AirspyStatsPanel {
    constructor({target, feed}) {
        this.el = document.getElementById(target);
        this.feed = feed;
        this._hist = [];    // ring of {t, total, dropped}   (ADC / USB FIFO)
        this._vhist = [];   // ring of {t, total, dropped}   (Valve / pipeline)
        this._rf = new Map();   // unified: band -> {ah:[], vh:[]} per-band rings
        this._render(null, null);
        feed.on(p => this._update(p));
    }

    // Push one counter sample onto a rolling ~WINDOW_S window, restarting it if the
    // counters went backwards (kotekan restarted under us).
    _push(hist, now, total, dropped) {
        if (hist.length && total < hist[hist.length - 1].total) hist.length = 0;
        hist.push({t: now, total, dropped});
        while (hist.length > 2 && now - hist[0].t > WINDOW_S) hist.shift();
    }

    _update(payload) {
        // UNIFIED: one page, three physical front ends -> a compact row per band.
        if (payload && Array.isArray(payload.rf_health)) return this._update_rf(payload.rf_health);
        const adc = payload && payload.adc;
        const valve = (payload && payload.valve) || null;
        const now = performance.now() / 1000;
        if (adc && adc.samples_total != null)
            this._push(this._hist, now, adc.samples_total, adc.samples_dropped || 0);
        if (valve) {
            // passed is null on older kotekan -> carry dropped alone; the window then
            // yields a drop RATE (frames/s) and the fraction cells read "n/a".
            const tot = valve.passed != null ? valve.passed + valve.dropped : null;
            this._push(this._vhist, now, tot != null ? tot : valve.dropped, valve.dropped);
        }
        this._render(adc || null, valve);
    }

    // Drop fraction over the rolling window: delta(dropped) / delta(dropped + passed).
    _window_frac(hist) {
        const h = hist;
        if (h.length < 2) return null;
        const dTot = h[h.length - 1].total - h[0].total;
        const dDrop = h[h.length - 1].dropped - h[0].dropped;
        return dTot > 0 ? dDrop / dTot : null;
    }

    // Frames lost per second over the window -- the fallback reading when there is no
    // denominator, and the number that actually moves the moment a peel is engaged.
    _window_rate(hist) {
        const h = hist;
        if (h.length < 2) return null;
        const dt = h[h.length - 1].t - h[0].t;
        return dt > 0 ? (h[h.length - 1].dropped - h[0].dropped) / dt : null;
    }

    _drop_window() {
        return this._window_frac(this._hist);
    }

    _render(adc, valve) {
        if (!this.el) return;
        const cell = (label, value, color, tip) =>
            `<div style="flex:1;min-width:7em;padding:.35em .6em;" title="${tip}">
               <div style="font-size:.72em;opacity:.65;">${label}</div>
               <div style="font-size:1.25em;font-weight:600;color:${color};">${value}</div>
             </div>`;
        const fmtFrac = f => f == null ? "n/a"
            : f === 0 ? "0"
            : f < 1e-6 ? "<1e-6" : (f * 100).toPrecision(2) + "%";
        let cells;
        if (!adc) {
            cells = cell("ADC", "—", "#8a8f98", "no /adcstat yet");
        } else {
            const rms = adc.rms != null ? adc.rms : NaN;
            const rmsCol = rms >= 20 ? "#3fb26f" : "#e8a13c";
            const rail = (adc.railfrac || 0) * 100;
            const railCol = rail > 1 ? "#d64550" : rail > 0.1 ? "#e8a13c" : "#3fb26f";
            const have = adc.samples_total != null;
            const cumFrac = have && adc.samples_total > 0
                ? (adc.samples_dropped || 0) / adc.samples_total : null;
            const winFrac = this._drop_window();
            const dropCol = !have ? "#8a8f98"
                : (cumFrac > 1e-4 ? "#d64550" : (winFrac > 0 ? "#e8a13c" : "#3fb26f"));
            cells =
                cell("ADC rms", isNaN(rms) ? "—" : rms.toFixed(0), rmsCol,
                     "raw ADC rms (counts); healthy input sits well above ~20") +
                cell("rail", rail.toFixed(rail > 0 ? 2 : 0) + "%", railCol,
                     "fraction of samples at the ADC rails -- must stay ~0") +
                cell("drop (15 s)", fmtFrac(winFrac), dropCol,
                     "libairspy FIFO drop fraction over the last ~15 s "
                     + "(browser-side counter delta)") +
                cell("drop (run)", fmtFrac(cumFrac), dropCol,
                     "cumulative drop fraction since kotekan start") +
                cell("events", have ? String(adc.drop_events || 0) : "n/a", dropCol,
                     "distinct drop episodes since start");
        }

        // ---- Valve: frames the PIPELINE threw away (GPU chain behind real time) ----
        if (!valve) {
            cells += cell("valve", "—", "#8a8f98",
                          "no kotekan_valve_dropped_frames_total for this band yet");
        } else {
            const vWinFrac = this._window_frac(this._vhist);
            const vRate = this._window_rate(this._vhist);
            const haveDen = valve.passed != null;
            const cumFrac = haveDen && valve.passed + valve.dropped > 0
                ? valve.dropped / (valve.passed + valve.dropped) : null;
            // Any nonzero drop in the window is worth attention: unlike the ADC's USB
            // FIFO, nothing upstream is bursty here -- a full output buffer means the
            // consumer is genuinely behind.
            const vCol = (vRate > 0 || (vWinFrac || 0) > 0) ? "#d64550"
                : (valve.dropped > 0 ? "#e8a13c" : "#3fb26f");
            const now = haveDen ? fmtFrac(vWinFrac)
                : (vRate == null ? "n/a" : vRate.toFixed(vRate >= 1 ? 0 : 2) + "/s");
            cells +=
                cell("valve drop (15 s)", now, vCol,
                     `frames dropped by ${valve.stage} over the last ~15 s -- SILENT data `
                     + "loss: the tracker never sees them and zero-fills the gap. Nonzero "
                     + "means the GPU chain is missing real time RIGHT NOW.") +
                cell("valve lost (run)", String(valve.dropped)
                     + (cumFrac != null ? ` (${fmtFrac(cumFrac)})` : ""), vCol,
                     "frames lost since kotekan start"
                     + (cumFrac != null ? ", and as a fraction of all frames offered"
                                        : " (no passed-frame counter on this build)"));
        }

        this.el.innerHTML =
            `<div style="display:flex;flex-wrap:wrap;align-items:stretch;">${cells}</div>`;
    }

    // Unified: push each band's ADC + valve counters onto its own rings, then render one
    // compact row per band -- the three front ends at a glance, valve drop (the silent loss)
    // called out per band since it is the number a new peel/feature moves first.
    _update_rf(rf) {
        const now = performance.now() / 1000;
        for (const b of rf) {
            const r = this._rf.get(b.band) || {ah: [], vh: []};
            if (b.adc && b.adc.samples_total != null)
                this._push(r.ah, now, b.adc.samples_total, b.adc.samples_dropped || 0);
            if (b.valve) {
                const tot = b.valve.passed != null ? b.valve.passed + b.valve.dropped : null;
                this._push(r.vh, now, tot != null ? tot : b.valve.dropped, b.valve.dropped);
            }
            this._rf.set(b.band, r);
        }
        this._render_rf(rf);
    }

    _render_rf(rf) {
        if (!this.el) return;
        const fmt = f => f == null ? "n/a" : f === 0 ? "0"
            : f < 1e-6 ? "<1e-6" : (f * 100).toPrecision(2) + "%";
        const cell = (v, color, tip) =>
            `<div style="flex:0 0 5.5em;padding:.15em .5em;" title="${tip}">
               <span style="font-size:1.05em;font-weight:600;color:${color};">${v}</span></div>`;
        const rows = rf.map(b => {
            const r = this._rf.get(b.band) || {ah: [], vh: []};
            const adc = b.adc || {};
            const rms = adc.rms != null ? adc.rms : NaN;
            const rmsCol = rms >= 20 ? "#3fb26f" : "#e8a13c";
            const aCum = adc.samples_total > 0 ? (adc.samples_dropped || 0) / adc.samples_total : null;
            const aWin = this._window_frac(r.ah);
            const aCol = aCum > 1e-4 ? "#d64550" : (aWin > 0 ? "#e8a13c" : "#3fb26f");
            const v = b.valve;
            const vWin = this._window_frac(r.vh), vRate = this._window_rate(r.vh);
            const vCol = v && (vRate > 0 || (vWin || 0) > 0) ? "#d64550"
                : (v && v.dropped > 0 ? "#e8a13c" : "#3fb26f");
            const vNow = !v ? "—" : (v.passed != null ? fmt(vWin)
                : (vRate == null ? "n/a" : vRate.toFixed(vRate >= 1 ? 0 : 2) + "/s"));
            const vLost = v ? String(v.dropped) : "—";
            return `<div style="display:flex;align-items:center;border-top:1px solid #f0f0f0;">
                <div style="flex:0 0 9em;font-weight:600;color:#5a6472;padding:.15em .5em;">${b.label}</div>`
                + cell(isNaN(rms) ? "—" : rms.toFixed(0), rmsCol, "ADC rms (counts)")
                + cell(fmt(aWin), aCol, "USB FIFO drop, last ~15 s")
                + cell(vNow, vCol, "VALVE drop (silent pipeline loss), last ~15 s")
                + cell(vLost, vCol, "valve frames lost since start")
                + "</div>";
        }).join("");
        const hdr = `<div style="display:flex;align-items:center;font-size:.72em;opacity:.6;">
            <div style="flex:0 0 9em;padding:.1em .5em;">front end</div>
            <div style="flex:0 0 5.5em;padding:.1em .5em;">rms</div>
            <div style="flex:0 0 5.5em;padding:.1em .5em;">ADC drop</div>
            <div style="flex:0 0 5.5em;padding:.1em .5em;">valve 15s</div>
            <div style="flex:0 0 5.5em;padding:.1em .5em;">valve lost</div></div>`;
        this.el.innerHTML = hdr + rows;
    }
}
