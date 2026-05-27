// Colormap controls: a select for the palette, a horizontal gradient bar
// showing the current mapping, and a slider that sets the min/max values
// (in dB) which the colormap is stretched across.
//
// All three pieces share state via ``app.state.cb`` (an ``imgPlotter``
// instance) which is also what WaterfallView reads.
//
// Pure HTML/CSS: the gradient is a CSS ``linear-gradient`` background,
// labels are positioned with percentages, and slider value labels are
// children of the jQuery-UI handle nodes so they track pixel-perfectly.

export class ColorPanel {
    constructor({app, target, color_range}) {
        this.app = app;
        this.bus = app.bus;
        this.state = app.state;
        this._target = target;
        this._color_range = color_range || [-20, 20];

        this._cb_tags = [];
        this._cb_bar = null;   // DOM element whose background is the gradient

        // Built in this order on purpose: select first, bar second, slider third.
        this._add_select(["viridis","inferno","magma","plasma","jet","gray","heat"]);
        this._add_bar();
        this._add_slider(this._color_range);
    }

    _add_select(palettes) {
        const cb = this.state.cb;
        const marg = 15;
        const cp = $("<select/>").appendTo(
            $("<div/>").appendTo("#" + this._target)
                .css({margin: marg + "px", display: "block"}));
        for (const name of palettes) {
            if (name in cb.colormaps) cp.append("<option>" + name + "</option>");
            else console.log(name + " not a known colormap!");
        }
        cp.selectmenu({
            change: (event, data) => this._change_palette(data.item.label),
            width: false,   // let the rendered menu size to its container
        })
        .data("ui-selectmenu")
        ._renderItem = function(ul, item) {
            // Mini gradient preview to the right of each palette name. The
            // <li> must be position:relative so the absolutely-positioned
            // swatch resolves against its own row -- without it every swatch
            // stacked at one spot (only the last visible) and stretched to
            // the full menu height.
            const li = $("<li>", {text: item.label})
                .css({position: "relative", "min-height": "1.8em",
                      "padding-right": "45%"});
            $("<span/>").appendTo(li)
                .css({position: "absolute", right: "6px", width: "40%",
                      top: "4px", bottom: "4px",
                      border: "1px solid #ccc",
                      background: cb.cssGradString(cb.colormaps[item.label])});
            return li.appendTo(ul);
        };
        // selectmenu wraps the <select> in two divs; both need to flex.
        cp.selectmenu("widget").css({width: "100%"});
    }

    _add_bar() {
        const cb = this.state.cb;
        const marg = 15;
        const BAR_HEIGHT = 30;

        const wrapper = $("<div/>")
            .css({width: "100%", padding: `0 ${marg}px`, "box-sizing": "border-box"})
            .appendTo($("#" + this._target));

        // Gradient bar: a div with a CSS linear-gradient background. Scales
        // perfectly with the card -- no viewBox, no SVG, no Raphael.
        this._cb_bar = $("<div/>")
            .css({width: "100%", height: BAR_HEIGHT + "px",
                  background: cb.cssGradString(cb.colormaps_current || _current_colors(cb))})
            .appendTo(wrapper);

        // Tag label strip below the bar. Labels distribute across the actual
        // rendered bar width via ``left: %``; edge labels anchor to their
        // corner so they stay on-canvas at narrow widths.
        const labeldiv = $("<div/>")
            .css({position: "relative", width: "100%", height: "18px",
                  "font-family": "sans-serif", "font-size": "12px"})
            .appendTo(wrapper);
        const ntags = 5;
        for (let i = 0; i < ntags; i++) {
            const pct = i / (ntags - 1) * 100;
            const transform = (i === 0)             ? "translateX(0)"
                            : (i === ntags - 1)     ? "translateX(-100%)"
                            :                         "translateX(-50%)";
            this._cb_tags.push(
                $("<span/>")
                    .css({position: "absolute", top: "2px", left: pct + "%",
                          transform, "white-space": "nowrap"})
                    .text((i / (ntags - 1) * (cb.max - cb.min) + cb.min).toFixed(2))
                    .appendTo(labeldiv));
        }
    }

    _add_slider(range) {
        const cb = this.state.cb;
        const self = this;
        const marg = 15;
        const inrange = [-1000, 1000];
        const scale = (inrange[1] - inrange[0]) / (range[1] - range[0]);

        const wrapper = $("<div/>")
            .css({position: "relative", width: "100%",
                  padding: `10px ${marg}px 6px`, "box-sizing": "border-box"})
            .appendTo($("#" + this._target));

        const cbslider = $("<div/>").uniqueId()
            .css({width: "100%"})
            .appendTo(wrapper);

        let handle_label_0, handle_label_1;

        cbslider.slider({
            min: inrange[0], max: inrange[1], range: true,
            values: [(cb.min - range[0]) * scale + inrange[0],
                     (cb.max - range[0]) * scale + inrange[0]],
            slide: function(event, ui) {
                cb.min = (ui.values[0] - inrange[0]) / scale + range[0];
                cb.max = (ui.values[1] - inrange[0]) / scale + range[0];
                handle_label_0.text(cb.min.toFixed(2));
                handle_label_1.text(cb.max.toFixed(2));
                for (let i = 0; i < self._cb_tags.length; i++) {
                    self._cb_tags[i].text(
                        (i / (self._cb_tags.length - 1) * (cb.max - cb.min) + cb.min).toFixed(2));
                }
                self.bus.emit("state:redraw_requested");
            },
        });

        // Attach value labels directly to the handle DOM nodes. jQuery-UI
        // positions handles with ``left: %`` of the track, so a child with
        // ``left: 50%; transform: translateX(-50%)`` sits centered under
        // the handle and tracks it perfectly through resizes and slides.
        const handles = cbslider.find(".ui-slider-handle");
        const label_css = {position: "absolute", top: "100%", left: "50%",
                           transform: "translateX(-50%)", "white-space": "nowrap",
                           "font-size": "12px", "font-family": "sans-serif",
                           "margin-top": "2px"};
        handle_label_0 = $("<span/>").css(label_css).text(cb.min.toFixed(2))
            .appendTo(handles[0]);
        handle_label_1 = $("<span/>").css(label_css).text(cb.max.toFixed(2))
            .appendTo(handles[1]);

        // Centered title below the slider. ``margin-top`` leaves room for
        // the value labels hanging off the handles.
        $("<div/>")
            .css({width: "100%", "text-align": "center", "font-size": "14px",
                  "font-family": "sans-serif", "margin-top": "24px"})
            .text("Color Bar Range [dB]")
            .appendTo(wrapper);
    }

    _change_palette(name) {
        const cb = this.state.cb;
        cb.gradientScale(cb.colormaps[name]);
        if (this._cb_bar) {
            this._cb_bar.css("background", cb.cssGradString(cb.colormaps[name]));
        }
        this.bus.emit("state:redraw_requested");
    }
}

// imgPlotter doesn't expose the current colors list directly; fall back to
// viridis on initial render. After the user changes palette, ``_change_palette``
// always sets the gradient explicitly from ``cb.colormaps[name]``.
function _current_colors(cb) {
    return cb.colormaps.viridis;
}
