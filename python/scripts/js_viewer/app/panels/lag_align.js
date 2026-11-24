// Crosscorr lag-alignment controls (ported from the legacy
// waterfall_crosscorr.js ``addLagcorr``).
//
// The AirspyAlign stage cross-correlates the two dongles' streams to find
// the integer-sample delay between them; applying that delay (as an
// ``add_lag`` config on the lagging airspyInput) brings them into phase
// alignment. Endpoints used, all via the shared KotekanRest adapter:
//
//   GET  <lag_align>/cal_lag          -> {lag}
//   GET  <lag_align>/get_correlation  -> {corr_pos[], corr_neg[], lag}
//   POST <airspy>/set_config {add_lag} -> shift that input
//
// Sign convention: lag > 0 means stream A is ahead, so we delay A
// (airspy_stages[0]); lag < 0 means B is ahead, delay B (airspy_stages[1]).

export class LagAlignPanel {
    constructor({app, target, lag_align_stage, airspy_stages}) {
        this.app = app;
        this.state = app.state;
        this.rest = app.kotekan;
        this._stage = lag_align_stage;
        this._airspy = airspy_stages || [];
        this._lag = 0;

        const host = $("#" + target).css({
            height: "100%", "min-height": "0",
            display: "flex", "flex-direction": "column",
        });

        // Button row -- flex-wrap so it reflows as the card narrows.
        const bar = $("<div/>")
            .css({display: "flex", "flex-wrap": "wrap", gap: "6px",
                  padding: "6px 8px", "box-sizing": "border-box"})
            .appendTo(host);

        const mkbtn = (label, icon, click) =>
            $("<button/>").appendTo(bar)
                .button({label, icons: icon ? {primary: icon} : {}})
                .click(click);

        mkbtn("Calculate Lag", "ui-icon-calculator", () => this._calc_lag());
        mkbtn("Show Lag Corr (slow)", "ui-icon-image", () => this._show_corr());
        mkbtn("Apply Lag", "ui-icon-check", () => this._apply_lag());
        // One-sample manual nudges. Nudging A delays A relative to B; nudging
        // B delays B relative to A -- between them they cover both directions.
        mkbtn("Nudge A +1", "ui-icon-plus", () => this._nudge(this._airspy[0]));
        mkbtn("Nudge B +1", "ui-icon-plus", () => this._nudge(this._airspy[1]));

        // Status line: last computed lag.
        this._status = $("<div/>")
            .css({"font-family": "sans-serif", "font-size": "12px",
                  padding: "0 8px 4px"})
            .text("Lag: (not yet calculated)")
            .appendTo(host);

        // Correlation plot fills the rest of the card.
        const plotwrap = $("<div/>").uniqueId()
            .css({flex: "1 1 0", "min-height": "120px", width: "100%"})
            .appendTo(host);
        this._plot = plotwrap.attr("id");
        const data = [
            {x: [], y: [], type: "scatter", name: "Positive lag (A ahead)"},
            {x: [], y: [], type: "scatter", name: "Negative lag (B ahead)"},
        ];
        const layout = {
            title: {text: "Airspy Lag Correlation"},
            xaxis: {title: {text: "Lag (samples)"}, linecolor: "black", zeroline: false},
            yaxis: {title: {text: "Correlation (arb)"}, linecolor: "black", zeroline: false},
            margin: {t: 30, l: 50, r: 10, b: 40},
            legend: {xanchor: "right", x: 1.0, y: 0.2},
            autosize: true,
        };
        Plotly.newPlot(this._plot, data, layout,
                       {staticPlot: true, responsive: true});
        new ResizeObserver(() => Plotly.Plots.resize(this._plot))
            .observe(plotwrap[0]);
    }

    _set_lag(lag) {
        this._lag = lag;
        const who = lag === 0 ? "aligned"
                  : lag > 0   ? `A ahead by ${lag}`
                  :             `B ahead by ${-lag}`;
        this._status.text(`Lag: ${lag} (${who})`);
    }

    _calc_lag() {
        this.rest.stageGet(this._stage, "cal_lag")
            .then(r => r.json())
            .then(d => this._set_lag(d.lag))
            .catch(e => console.warn("cal_lag failed:", e));
    }

    _show_corr() {
        this.rest.stageGet(this._stage, "get_correlation")
            .then(r => r.json())
            .then(d => {
                Plotly.restyle(this._plot, {
                    x: [_.range(0, d.corr_pos.length),
                        _.range(0, -d.corr_neg.length, -1)],
                    y: [d.corr_pos, d.corr_neg],
                });
                this._set_lag(d.lag);
            })
            .catch(e => console.warn("get_correlation failed:", e));
    }

    _apply_lag() {
        if (this._lag === 0) { this._set_lag(0); return; }
        // Delay whichever input is ahead.
        const stage = this._lag > 0 ? this._airspy[0] : this._airspy[1];
        this.rest.stagePost(stage, "set_config", {add_lag: Math.abs(this._lag)});
    }

    /// Fire-and-forget single-sample lag bump on @c stage. The last-calculated
    /// lag in @c this._lag becomes stale after this -- hit "Calculate Lag" to
    /// see the new residual.
    _nudge(stage) {
        if (!stage) return;
        this.rest.stagePost(stage, "set_config", {add_lag: 1})
            .catch(e => console.warn("nudge failed:", e));
    }
}
