// Two-handled slider that picks the displayed frequency window. The window
// lives on ``app.state.disp_freq``; WaterfallView and SpectrumView read it
// directly to crop their displays.
//
// The slider's MHz range is rebuilt live via ``set_range`` when the band is
// retuned (App.apply_band), so the window tracks whatever LO the airspy is
// actually on.
//
// Label positioning is HTML-only: handle value labels are children of the
// jQuery-UI handle nodes, so they track the handles through resizes and
// slides without any pixel math.

export class FreqRangePanel {
    constructor({app, target, range}) {
        this.app = app;
        this.bus = app.bus;
        this.state = app.state;
        this._marg = 15;

        this._wrapper = $("<div/>")
            .css({position: "relative", width: "100%",
                  padding: `10px ${this._marg}px 6px`, "box-sizing": "border-box"})
            .appendTo($("#" + target));

        this._build(range);
    }

    /// Rebuild the slider for a new MHz range (e.g. after a band retune).
    /// ``disp_freq`` optionally sets the displayed window; otherwise the
    /// current window is clamped into the new range.
    set_range(range, disp_freq) {
        if (disp_freq) this.state.disp_freq = disp_freq.slice();
        this._build(range);
    }

    _build(range) {
        this._range = range;
        const inrange = [-100, 100];
        const scale = (inrange[1] - inrange[0]) / (range[1] - range[0]);
        const self = this;

        // Clamp the displayed window into the (possibly new) band.
        let d0 = Math.min(Math.max(this.state.disp_freq[0], range[0]), range[1]);
        let d1 = Math.min(Math.max(this.state.disp_freq[1], range[0]), range[1]);
        this.state.disp_freq = [d0, d1];

        this._wrapper.empty();
        const slider = $("<div/>").uniqueId().css({width: "100%"}).appendTo(this._wrapper);

        let handle_label_0, handle_label_1;
        slider.slider({
            min: inrange[0], max: inrange[1], range: true,
            values: [(d0 - range[0]) * scale + inrange[0],
                     (d1 - range[0]) * scale + inrange[0]],
            slide: function(event, ui) {
                self.state.disp_freq[0] = (ui.values[0] - inrange[0]) / scale + range[0];
                self.state.disp_freq[1] = (ui.values[1] - inrange[0]) / scale + range[0];
                handle_label_0.text(self.state.disp_freq[0].toFixed(2));
                handle_label_1.text(self.state.disp_freq[1].toFixed(2));
                self.bus.emit("state:redraw_requested");
            },
        });

        // Value labels are children of the handle DOM nodes (see color.js).
        const handles = slider.find(".ui-slider-handle");
        const label_css = {position: "absolute", top: "100%", left: "50%",
                           transform: "translateX(-50%)", "white-space": "nowrap",
                           "font-size": "12px", "font-family": "sans-serif",
                           "margin-top": "2px"};
        handle_label_0 = $("<span/>").css(label_css).text(d0.toFixed(2)).appendTo(handles[0]);
        handle_label_1 = $("<span/>").css(label_css).text(d1.toFixed(2)).appendTo(handles[1]);

        $("<div/>")
            .css({width: "100%", "text-align": "center", "font-size": "14px",
                  "font-family": "sans-serif", "margin-top": "24px"})
            .text("Freq Range [MHz]")
            .appendTo(this._wrapper);
    }
}
