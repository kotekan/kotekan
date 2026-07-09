// Per-pol mean spectrum for the dualpol viewer: the mean over the visible
// waterfall buffer of each channel, one trace per pol (XX/YY). This is the raw
// bandpass and is deliberately independent of the waterfall's median-subtract
// toggle -- subtracting each channel's per-time median from a mean-over-time
// would just collapse the trace to ~0.
//
// Computed in dB space to match the waterfall (10*log10 is applied per sample
// before averaging), and rate-limited so the Plotly redraw doesn't fight the
// waterfall's animation.

const MIN_UPDATE_INTERVAL_MS = 150;
const POL_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"];

export class DualpolSpectrumView {
    constructor({app, target, labels}) {
        this.app = app;
        this.bus = app.bus;
        this.state = app.state;
        this.labels = labels && labels.length ? labels : ["XX", "YY"];
        this._last_update = 0;
        this._pending = false;

        $("#" + target).css({
            height: "100%", "min-height": "0", display: "flex",
            "flex-direction": "column",
        });
        const wrapper = $("<div/>").uniqueId()
            .css({width: "100%", flex: "1 1 0", "min-height": "120px"})
            .appendTo($("#" + target));
        this.plot = wrapper.attr("id");

        const data = this.labels.map((lab, i) => ({
            x: [], y: [], type: "scatter", mode: "lines", name: lab,
            line: {color: POL_COLORS[i % POL_COLORS.length], width: 1},
        }));
        const layout = {
            xaxis: {title: {text: "Frequency (MHz)"}, linecolor: "black", zeroline: false},
            yaxis: {title: {text: "Power (dB)"}, linecolor: "black", zeroline: false},
            margin: {t: 8, l: 50, r: 10, b: 40},
            legend: {xanchor: "right", x: 1.0, y: 1.0},
            autosize: true,
        };
        Plotly.newPlot(this.plot, data, layout, {staticPlot: true, responsive: true});
        new ResizeObserver(() => Plotly.Plots.resize(this.plot)).observe(wrapper[0]);

        this.bus.on("state:redraw_requested", () => this._request());
    }

    _request() {
        const now = Date.now();
        if (now - this._last_update >= MIN_UPDATE_INTERVAL_MS) {
            this._last_update = now;
            this._update();
        } else if (!this._pending) {
            this._pending = true;
            setTimeout(() => {
                this._pending = false; this._last_update = Date.now(); this._update();
            }, MIN_UPDATE_INTERVAL_MS - (now - this._last_update));
        }
    }

    _update() {
        const s = this.state;
        if (!s.freq_list || s.freq_list.length < 2 || !s.scroll_data.length) return;
        const nfreq = s.num_freqs;
        const nvis = s.nvis || this.labels.length;
        const scd = s.scroll_data;
        const ntimes = Math.min(scd.length, s.waterfall_buffer_display_length);
        const start = Math.max(0, scd.length - ntimes);
        const freqs = Array.from(s.freq_list);

        const ys = [];
        for (let p = 0; p < nvis; p++) {
            const off = p * nfreq;
            const y = new Array(nfreq);
            for (let f = 0; f < nfreq; f++) {
                let sum = 0, n = 0;
                for (let j = start; j < scd.length; j++) {
                    const v = scd[j][off + f];
                    if (v === v && v > 0) { sum += 10 * Math.log10(v); n++; }
                }
                y[f] = n ? sum / n : null;
            }
            ys.push(y);
        }
        Plotly.update(this.plot,
            {x: ys.map(() => freqs), y: ys}, {},
            ys.map((_, i) => i));
    }
}
