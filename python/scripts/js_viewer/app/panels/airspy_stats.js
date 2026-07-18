// Airspy stream-health strip: ADC rms / rail % / sample-drop rate, from the same
// <airspy>/adcstat the shared GpsFeed already polls (~1.5 s) -- ZERO extra kotekan
// load; this panel only renders what the feed already fetches.
//
// Drop rate needs the 2026-07-18 airspyInput counters (samples_total = true index
// incl. accounted gaps, samples_dropped = cumulative libairspy FIFO drops,
// drop_events). The instantaneous rate is computed browser-side over a rolling
// ~15 s window of counter deltas; the cumulative fraction over the whole run is
// shown beside it. Counters absent (older kotekan) -> the drop cells show "n/a".
//
// Color language: green = healthy, amber = attention, red = data being lost.
//   rms: amber < 20 (weak input / dead antenna?)   rail%: amber > 0.1, red > 1
//   drop: amber > 0 in-window, red > 1e-4 cumulative

const WINDOW_S = 15;

export class AirspyStatsPanel {
    constructor({target, feed}) {
        this.el = document.getElementById(target);
        this.feed = feed;
        this._hist = [];   // ring of {t, total, dropped}
        this._render(null);
        feed.on(p => this._update(p));
    }

    _update(payload) {
        const adc = payload && payload.adc;
        if (!adc) return this._render(null);
        const now = performance.now() / 1000;
        if (adc.samples_total != null) {
            const h = this._hist;
            // counter reset (kotekan restart) -> restart the window
            if (h.length && adc.samples_total < h[h.length - 1].total) h.length = 0;
            h.push({t: now, total: adc.samples_total, dropped: adc.samples_dropped || 0});
            while (h.length > 2 && now - h[0].t > WINDOW_S) h.shift();
        }
        this._render(adc);
    }

    _drop_window() {
        const h = this._hist;
        if (h.length < 2) return null;
        const dTot = h[h.length - 1].total - h[0].total;
        const dDrop = h[h.length - 1].dropped - h[0].dropped;
        return dTot > 0 ? dDrop / dTot : null;
    }

    _render(adc) {
        if (!this.el) return;
        const cell = (label, value, color, tip) =>
            `<div style="flex:1;min-width:7em;padding:.35em .6em;" title="${tip}">
               <div style="font-size:.72em;opacity:.65;">${label}</div>
               <div style="font-size:1.25em;font-weight:600;color:${color};">${value}</div>
             </div>`;
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
            const fmtFrac = f => f == null ? "n/a"
                : f === 0 ? "0"
                : f < 1e-6 ? "<1e-6" : (f * 100).toPrecision(2) + "%";
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
        this.el.innerHTML =
            `<div style="display:flex;flex-wrap:wrap;align-items:stretch;">${cells}</div>`;
    }
}
