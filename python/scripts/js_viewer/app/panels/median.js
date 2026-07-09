// Median-subtract toggle for the dual-pol waterfall.
//
// When on, each pol's waterfall subtracts its per-channel median over the
// visible time window (the old pyPeek "medsub"): the standing bandpass drops
// out and drifting sources / transients stand out. The actual subtraction is
// done per view in WaterfallView._dodraw (so XX and YY each remove their own
// bandpass); this panel just flips ``state.median_subtract`` and asks the
// colour bar to re-fit to the now near-zero-centred data.

export class MedianSubtractPanel {
    constructor({app, target}) {
        this.app = app;
        this.state = app.state;
        this.bus = app.bus;
        this._target = target;
        this._add_checkbox();
    }

    _add_checkbox() {
        const self = this;
        const wrapper = $("<div/>")
            .css({margin: "8px", display: "flex", "align-items": "center", gap: "6px"})
            .appendTo("#" + this._target);

        const cb = $("<input/>", {type: "checkbox"})
            .prop("checked", !!this.state.median_subtract)
            .appendTo(wrapper);
        cb.uniqueId();

        $("<label/>")
            .attr("for", cb.attr("id"))
            .text("Subtract per-channel median")
            .css({"font-family": "sans-serif", "font-size": "13px", cursor: "pointer"})
            .appendTo(wrapper);

        cb.on("change", function() {
            self.state.median_subtract = this.checked;
            // Re-fit the colour bar: median-subtracted data is centred near 0
            // with a much smaller spread than the raw bandpass.
            self.state.request_color_fit = true;
            self.bus.emit("state:redraw_requested");
        });

        $("<div/>")
            .css({margin: "0 8px 8px", "font-family": "sans-serif",
                  "font-size": "11px", color: "#666"})
            .text("Removes the standing bandpass (median over the visible time "
                + "window, per channel) so drifting sources and transients pop.")
            .appendTo("#" + this._target);
    }
}
