// Two sliders that control how many integrations are kept in the ring
// buffer and how many of those are actually painted in the waterfall.

// Common slider+input+optional-button row. ``buttons`` is an array of
// ``{label, icon, click}`` descriptors rendered after the input.
function _slider_row({target, app, label, min, max, value, on_change, buttons}) {
    const self = {};
    const marg = 10;
    const wrapper = $("<div/>").uniqueId()
        .css({width: "100%", margin: marg + "px 0",
              padding: `0 ${marg}px`, "box-sizing": "border-box",
              display: "flex", "align-items": "center", gap: "8px",
              "flex-wrap": "wrap"})
        .appendTo($("#" + target));

    $("<span/>").css({"font-family": "sans-serif", "white-space": "nowrap"})
        .text(label).appendTo(wrapper);

    const slider = $("<div/>").uniqueId()
        .css({flex: "1 1 80px", "min-width": "80px"})
        .appendTo(wrapper);

    const bins_text = $("<input type='number'/>")
        .attr({min, max})
        .css({width: "5em", "font-size": "16pt"})
        .val(value);
    bins_text.appendTo(wrapper);

    slider.slider({
        min, max, value,
        slide: function(event, ui) {
            on_change(ui.value);
            bins_text.val(ui.value);
        },
    });
    bins_text.change(function() {
        let v = parseInt(this.value);
        if (v < min) v = min;
        if (v > max) v = max;
        this.value = v;
        slider.slider("value", v);
        on_change(v);
    });
    bins_text.numeric();

    for (const b of (buttons || [])) {
        $("<button/>").uniqueId().appendTo(wrapper)
            .button({label: b.label, icons: {primary: b.icon}})
            .click(b.click);
    }
    return self;
}

export class BufferControlPanel {
    constructor({app, target}) {
        this.app = app;
        this.bus = app.bus;
        this.state = app.state;

        const self = this;
        _slider_row({
            target, app,
            label: "Buffer length:",
            min: 100, max: this.state.waterfall_buffer_max_length,
            value: this.state.waterfall_buffer_length,
            on_change: (v) => {
                self.state.waterfall_buffer_length = v;
                self.bus.emit("state:buffer_length_changed");
                self.bus.emit("state:redraw_requested");
            },
            buttons: [{
                label: "Clear", icon: "ui-icon-close",
                click: () => {
                    self.state.scroll_data = [];
                    self.state.timearr = [];
                    self.bus.emit("state:redraw_requested");
                },
            }],
        });
    }
}

export class WaterfallControlPanel {
    constructor({app, target}) {
        this.app = app;
        this.bus = app.bus;
        this.state = app.state;

        const self = this;
        _slider_row({
            target, app,
            label: "Display length:",
            min: 100, max: this.state.waterfall_buffer_length,
            value: this.state.waterfall_buffer_display_length,
            on_change: (v) => {
                self.state.waterfall_buffer_display_length = v;
                self.bus.emit("state:redraw_requested");
            },
        });
    }
}
