// Baseline tools: take a snapshot of the mean spectrum as a baseline, fit
// a polynomial through the off-line bins, subtract on display. Also the
// auto-cal routine that briefly retunes airspyInput to a calibration
// frequency and captures a clean bandpass.

export class BaselinePanel {
    constructor({app, target, autocal_stage, line_mask_mhz, autocal_freqs_mhz}) {
        this.app = app;
        this.state = app.state;
        this.bus = app.bus;

        this._target = target;
        this._autocal_stage = autocal_stage || null;
        // From viewer_config (defaults: HI line / airspy): the [lo, hi] band the
        // poly fit excludes, and the [calibrate, observe] auto-cal retune freqs.
        this._line_mask = line_mask_mhz || [1419.9, 1420.9];
        this._autocal_freqs = autocal_freqs_mhz || [1416, 1421];

        this._add_take_baseline();
        this._add_poly_fitter();
        this._add_subtract_checkbox();
        if (this._autocal_stage) this._add_autocal();

        // Drive the bandpass / skip-mode state machine off the WS frame stream.
        this.bus.on("state:frame_received", (d) => this._on_frame(d));
    }

    _add_take_baseline() {
        const self = this;
        const wrapper = $("<div/>").uniqueId().appendTo($("#" + this._target))
            .css({position: "relative", display: "block", margin: "4px 8px"});
        $("<button/>").appendTo(wrapper)
            .button({label: "Take a Spectral Baseline", icons: {primary: "ui-icon-play"}})
            .click(function() {
                // Mean of the current scroll_data, bin by bin.
                self.state.spectrum_baseline = _.map(_.unzip(self.state.scroll_data), _mean);
                self._reveal_checkbox();
            });
    }

    _add_poly_fitter() {
        const self = this;
        const wrapper = $("<div/>").uniqueId().appendTo($("#" + this._target))
            .css({position: "relative", display: "block", margin: "4px 8px"});
        $("<button/>").appendTo(wrapper)
            .button({label: "Fit Polynomial Baseline", icons: {primary: "ui-icon-play"}})
            .click(function() {
                const s = self.state;
                // If no baseline has been captured yet, fit directly to the
                // raw spectrum (treat the prior baseline as flat zeros). The
                // previous behaviour silently produced NaNs from the
                // undefined-baseline subtraction.
                const baseline = s.spectrum_baseline
                                 || new Array(s.num_freqs).fill(0);
                // Sample the residual at every bin OUTSIDE the spectral-line
                // mask, then fit a 4th-order polynomial.
                const x = [], y = [];
                for (let idx = 0; idx < s.num_freqs; idx++) {
                    const f = s.freq_list[idx];
                    const in_disp = f > s.disp_freq[0] && f < s.disp_freq[1];
                    const off_line = f < self._line_mask[0] || f > self._line_mask[1];
                    if (in_disp && off_line) {
                        x.push(idx);
                        y.push(s.spectrum[idx] - baseline[idx]);
                    }
                }
                if (x.length === 0) {
                    console.warn("Fit Polynomial Baseline: no bins to fit "
                                 + "(display window empty, or entirely inside "
                                 + "the HI-line mask).");
                    return;
                }
                const solver = Polyfit(x, y).getPolynomial(4);
                const polybase = Array.from(Array(s.num_freqs).keys()).map(solver);
                s.spectrum_baseline = baseline.map((e, i) => e + polybase[i]);
                self._reveal_checkbox();
            });
    }

    _add_subtract_checkbox() {
        const self = this;
        // jQuery UI's checkboxradio needs the canonical structure
        //     <input type="checkbox" id="X"/><label for="X">...</label>
        // -- input FIRST, label as its next sibling with a matching `for`.
        // The wrap-the-input-in-the-label shorthand renders but the widget's
        // overlay swallows clicks instead of forwarding them to the input
        // (had the symptom of an unclickable Remove-Baseline box).
        this._checkbox_wrapper = $("<div/>").uniqueId().appendTo($("#" + this._target))
            .css({position: "relative", display: "block", margin: "4px 8px",
                  visibility: "hidden"});
        const input = $("<input type='checkbox'/>").uniqueId()
            .appendTo(this._checkbox_wrapper)
            .click(function() {
                self.state.baseline_enabled = this.checked;
                self.bus.emit("state:redraw_requested");
            });
        $("<label/>")
            .attr("for", input.attr("id"))
            .text("Remove Baseline from Display")
            .appendTo(this._checkbox_wrapper);
        input.checkboxradio({icon: false});
    }

    _reveal_checkbox() {
        if (this._checkbox_wrapper) this._checkbox_wrapper.css({visibility: "visible"});
        this.bus.emit("state:redraw_requested");
    }

    _add_autocal() {
        const self = this;
        const stage = this._autocal_stage;
        // Single source of truth for the button label so the progress / reset
        // text doesn't drift from the initial one. The previous code had two
        // copies and reset the button to a stale "Autocalibrate Bandpass" on
        // completion.
        this._bandpass_button_label = `Take ${this._autocal_freqs[0]}MHz Bandpass`;
        const wrapper = $("<div/>").uniqueId().appendTo($("#" + this._target))
            .css({position: "relative", display: "block", margin: "4px 8px"});
        const button = $("<button/>").appendTo(wrapper)
            .button({label: this._bandpass_button_label, icons: {primary: "ui-icon-play"}})
            .click(function() {
                self.app.kotekan.stagePost(stage, "set_config", {freq: self._autocal_freqs[0]})
                    .then(() => {
                        self.state.bandpass_data = [];
                        self.state.scroll_data = [];
                        self.state.timearr = [];
                        self.state.mode = "bandpass";
                        self.bus.emit("state:mode_changed", {mode: "bandpass"});
                    });
            });
        this._bandpass_button = button;

        $("<input type='number'/>")
            .attr({min: 16, max: 4096})
            .css({width: "17%", display: "inline", "font-size": "16pt", "margin-top": 5})
            .val(this.state.autocal_length)
            .appendTo(wrapper)
            .change(function() {
                let v = parseInt(this.value);
                if (v < $(this).attr("min")) v = +$(this).attr("min");
                if (v > $(this).attr("max")) v = +$(this).attr("max");
                this.value = v;
                self.state.autocal_length = v;
            })
            .numeric();
    }

    _on_frame({data}) {
        const s = this.state;
        if (s.mode === "bandpass") {
            s.bandpass_data.push(data);
            const total = s.autocal_length + s.skip_length;
            const pct = (s.bandpass_data.length / total * 100).toFixed(2);
            if (this._bandpass_button) {
                this._bandpass_button.button({label: "Taking calibration: " + pct + "%"});
            }
            if (s.bandpass_data.length >= total) {
                s.mode = "idle";
                for (let i = 0; i < s.skip_length; i++) s.bandpass_data.shift();
                this._finish_bandpass();
            }
        }
        // ``skip`` mode is handled by WaterfallView's _on_frame; once enough
        // frames have arrived after the retune, it transitions back to normal.
    }

    _finish_bandpass() {
        const self = this;
        const s = this.state;
        s.spectrum_baseline = _.map(_.unzip(s.bandpass_data), _mean);
        this.app.kotekan.stagePost(this._autocal_stage, "set_config", {freq: this._autocal_freqs[1]})
            .then(() => {
                self._reveal_checkbox();
                if (self._bandpass_button) {
                    self._bandpass_button.button({label: self._bandpass_button_label});
                }
                s.scroll_data = [];
                s.mode = "skip";
                self.bus.emit("state:mode_changed", {mode: "skip"});
            });
    }
}
