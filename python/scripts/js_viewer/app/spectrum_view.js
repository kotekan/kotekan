// Plotly-based spectrum panel (the "Spectral Power" plot under the waterfall)
// plus the optional "Excess Spectral Power" plot. Both originally lived in
// ``add_spectrum`` and ``add_spectrum_excess`` in the monolithic waterfall.js.
//
// Updates are throttled (see ``MIN_UPDATE_INTERVAL_MS``): the bus emits
// ``state:redraw_requested`` on every incoming frame (40 Hz), but a
// Plotly.update at that rate dominates CPU and starves the waterfall's
// requestAnimationFrame.

const MIN_UPDATE_INTERVAL_MS = 100;

export class SpectrumView {
    constructor({app, target}) {
        this.app = app;
        this.bus = app.bus;
        this.state = app.state;
        this._mounted = false;
        this._target = target;
        this._last_update = 0;
        this._update_pending = false;

        // Make the host card's mount-point fill its panel-body (height
        // 100%) and lay out its children as a flex column. The two plot
        // wrappers below each grab ``flex: 1`` so they split the available
        // height equally and grow with the GridStack card. ``min-height:
        // 0`` keeps flex children from being clamped at their intrinsic
        // size.
        $("#" + target).css({
            height: "100%",
            "min-height": "0",
            display: "flex",
            "flex-direction": "column",
        });

        const wrapper = $("<div/>").uniqueId()
            .css({width: "100%", flex: "1 1 0", "min-height": "150px"})
            .appendTo($("#" + target));
        this.spectrum_plot = wrapper.attr("id");

        const data = [
            {x: [], y: [], type: "scatter", name: "Latest", mode: "markers", marker: {size: 3}},
            {x: [], y: [], type: "scatter", name: "Baseline"},
            {x: [], y: [], type: "scatter", name: "Mean"},
        ];
        const layout = {
            title: {text: "Spectral Power"},
            xaxis: {title: {text: "Frequency (MHz)"}, linecolor: "black", zeroline: false},
            yaxis: {title: {text: "Power (dB bits^2)"}, linecolor: "black", zeroline: false},
            margin: {t: 30, l: 50, r: 10, b: 40},
            legend: {xanchor: "right", x: 1.0, y: 0.},
            autosize: true,
        };
        Plotly.newPlot(this.spectrum_plot, data, layout,
                       {staticPlot: true, responsive: true});
        new ResizeObserver(() => Plotly.Plots.resize(this.spectrum_plot))
            .observe(wrapper[0]);

        // Refresh on every redraw request so the spectrum follows the
        // waterfall + color-range sliders without depending on a
        // freshly-drawn waterfall. ``_request_update`` rate-limits.
        this.bus.on("state:redraw_requested", () => this._request_update());
    }

    // Coalesce bursts of redraw events into at most one Plotly.update per
    // MIN_UPDATE_INTERVAL_MS. If a request arrives during the cooldown,
    // we schedule a single trailing update so the final state still lands.
    _request_update() {
        const now = Date.now();
        const since = now - this._last_update;
        if (since >= MIN_UPDATE_INTERVAL_MS) {
            this._last_update = now;
            this._update_spectrum();
            return;
        }
        if (this._update_pending) return;
        this._update_pending = true;
        setTimeout(() => {
            this._update_pending = false;
            this._last_update = Date.now();
            this._update_spectrum();
        }, MIN_UPDATE_INTERVAL_MS - since);
    }

    add_excess({target}) {
        const wrapper = $("<div/>").uniqueId()
            .css({width: "100%", flex: "1 1 0", "min-height": "120px"})
            .appendTo($("#" + target));
        this.spectrum_excess_plot = wrapper.attr("id");

        const data = [
            {x: [], y: [], type: "scatter", name: "Latest", mode: "markers", marker: {size: 3}},
            {x: [], y: [], type: "scatter", name: "Mean"},
        ];
        const layout = {
            title: {text: "Excess Spectral Power"},
            xaxis: {title: {text: "Frequency (MHz)"}, linecolor: "black", zeroline: false},
            yaxis: {title: {text: "Excess Power (dB bits^2)"},
                    linecolor: "black", zeroline: false, range: [-1, 4]},
            margin: {t: 30, l: 50, r: 10, b: 40},
            legend: {xanchor: "right", x: 1.0, y: 0.},
            autosize: true,
        };
        Plotly.newPlot(this.spectrum_excess_plot, data, layout,
                       {staticPlot: true, responsive: true});
        new ResizeObserver(() => Plotly.Plots.resize(this.spectrum_excess_plot))
            .observe(wrapper[0]);
        this._excess_enabled = true;
    }

    _update_spectrum() {
        const s = this.state;
        if (!s.freq_list || !s.spectrum) return;
        const scd = s.scroll_data;
        if (scd.length === 0) return;

        // Slice the current display freq window. Clamp because ``disp_freq``
        // can sit slightly outside ``fl[0]..fl[last]`` (the slider range is
        // bin-edges, ``fl`` is bin-centres).
        const fl = s.freq_list;
        const freq_sc = fl[fl.length - 1] - fl[0];
        const freq_lo = Math.max(0,           Math.floor(s.num_freqs * (s.disp_freq[0] - fl[0]) / freq_sc));
        const freq_hi = Math.min(s.num_freqs, Math.ceil (s.num_freqs * (s.disp_freq[1] - fl[0]) / freq_sc));

        const x          = Array.from(fl).slice(freq_lo, freq_hi);
        const latest     = Array.from(scd[scd.length - 1]).slice(freq_lo, freq_hi);
        const mean       = Array.from(s.spectrum).slice(freq_lo, freq_hi);
        const baseline   = s.spectrum_baseline.slice(freq_lo, freq_hi);
        const dB         = (v) => 10 * Math.log10(v);

        let y_latest, y_mean, y_baseline;
        if (s.baseline_enabled) {
            y_latest   = latest.map((v, i) => dB(v / baseline[i]));
            y_mean     = mean  .map((v, i) => dB(v / baseline[i]));
            y_baseline = baseline.map(dB);
        } else {
            y_latest   = latest.map(dB);
            y_mean     = mean.map(dB);
            y_baseline = baseline.map(dB);
        }
        // ``Plotly.update`` sets both the trace data and the layout in one
        // shot. The layout half tracks the color-bar slider's [min, max]
        // onto the spectrum plot's Y axis, and pins the X axis range to
        // the actual data range so the leftmost point sits flush against
        // the Y axis (Plotly otherwise auto-pads, leaving an empty gap).
        const x_range = [x[0], x[x.length - 1]];
        Plotly.update(this.spectrum_plot,
            {x: [x, x, x], y: [y_latest, y_baseline, y_mean]},
            {"yaxis.range": [s.cb.min, s.cb.max],
             "xaxis.range": x_range});

        if (this._excess_enabled) {
            const ratio = (a, b) => dB(a / b);
            Plotly.update(this.spectrum_excess_plot,
                {x: [x, x],
                 y: [latest.map((v, i) => ratio(v, baseline[i])),
                     mean  .map((v, i) => ratio(v, baseline[i]))]},
                {"xaxis.range": x_range});
        }
    }
}
