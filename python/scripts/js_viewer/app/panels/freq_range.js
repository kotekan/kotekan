// Two-handled slider that picks the displayed frequency window. The window
// lives on ``app.state.disp_freq``; WaterfallView and SpectrumView read it
// directly to crop their displays.
//
// Label positioning is HTML-only: handle value labels are children of the
// jQuery-UI handle nodes, so they track the handles through resizes and
// slides without any pixel math.

export class FreqRangePanel {
    constructor({app, target, range}) {
        this.app = app;
        this.bus = app.bus;
        this.state = app.state;

        const inrange = [-100, 100];
        const scale = (inrange[1] - inrange[0]) / (range[1] - range[0]);
        const marg = 15;

        const wrapper = $("<div/>")
            .css({position: "relative", width: "100%",
                  padding: `10px ${marg}px 6px`, "box-sizing": "border-box"})
            .appendTo($("#" + target));
        const slider = $("<div/>").uniqueId()
            .css({width: "100%"})
            .appendTo(wrapper);

        const self = this;
        let handle_label_0, handle_label_1;

        slider.slider({
            min: inrange[0], max: inrange[1], range: true,
            values: [(this.state.disp_freq[0] - range[0]) * scale + inrange[0],
                     (this.state.disp_freq[1] - range[0]) * scale + inrange[0]],
            slide: function(event, ui) {
                self.state.disp_freq[0] = (ui.values[0] - inrange[0]) / scale + range[0];
                self.state.disp_freq[1] = (ui.values[1] - inrange[0]) / scale + range[0];
                handle_label_0.text(self.state.disp_freq[0].toFixed(2));
                handle_label_1.text(self.state.disp_freq[1].toFixed(2));
                self.bus.emit("state:redraw_requested");
            },
        });

        // Attach value labels directly to the handle DOM nodes -- see
        // color.js for the rationale (handles already do the pixel-space
        // positioning we want).
        const handles = slider.find(".ui-slider-handle");
        const label_css = {position: "absolute", top: "100%", left: "50%",
                           transform: "translateX(-50%)", "white-space": "nowrap",
                           "font-size": "12px", "font-family": "sans-serif",
                           "margin-top": "2px"};
        handle_label_0 = $("<span/>").css(label_css)
            .text(this.state.disp_freq[0].toFixed(2)).appendTo(handles[0]);
        handle_label_1 = $("<span/>").css(label_css)
            .text(this.state.disp_freq[1].toFixed(2)).appendTo(handles[1]);

        $("<div/>")
            .css({width: "100%", "text-align": "center", "font-size": "14px",
                  "font-family": "sans-serif", "margin-top": "24px"})
            .text("Freq Range [MHz]")
            .appendTo(wrapper);
    }
}
