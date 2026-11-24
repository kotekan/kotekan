// Center-frequency (LO) control. Retunes the airspy device(s) live via the
// REST set_config endpoint and re-derives the viewer's frequency axis from
// the new LO -- so you can hop bands (e.g. down to ~90 MHz FM) from the
// browser without restarting kotekan.
//
// For crosscorr both dongles must share an LO, so we POST the new freq to
// every airspy stage. The actual axis recompute lives in App.retune /
// App.apply_band; this panel is just the input + button.

export class TuningPanel {
    constructor({app, target, initial_mhz, tuning_range_mhz}) {
        this.app = app;
        const [lo_mhz, hi_mhz] = tuning_range_mhz || [24, 1800]; // default: airspy R2 span

        const marg = 10;
        const wrapper = $("<div/>")
            .css({width: "100%", padding: `4px ${marg}px`, "box-sizing": "border-box",
                  display: "flex", "flex-wrap": "nowrap", "align-items": "center",
                  gap: "8px"})
            .appendTo($("#" + target));

        $("<span/>").css({"font-family": "sans-serif", "white-space": "nowrap"})
            .text("Center (LO) [MHz]:").appendTo(wrapper);

        this._input = $("<input type='number'/>")
            .attr({min: lo_mhz, max: hi_mhz, step: "any"})
            .css({flex: "1 1 0", "min-width": "0", "font-size": "14pt"})
            .val(initial_mhz != null ? initial_mhz.toFixed(3) : "")
            .appendTo(wrapper);

        const apply = () => {
            const mhz = parseFloat(this._input.val());
            if (!isFinite(mhz)) return;
            this.app.retune(mhz);
        };

        $("<button/>").appendTo(wrapper)
            .css({"flex-shrink": "0"})
            .button({label: "Set", icons: {primary: "ui-icon-signal"}})
            .click(apply);

        // Enter in the field also applies.
        this._input.on("keydown", (e) => { if (e.key === "Enter") apply(); });
    }

    /// Reflect the actual LO once it's read back from the device.
    set_value(mhz) {
        if (mhz != null && isFinite(mhz)) this._input.val(mhz.toFixed(3));
    }
}
