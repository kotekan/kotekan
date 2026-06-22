// Three vertical sliders (LNA / mixer / IF gain) + an ADC stats readout for
// one airspy producer stage. Talks to kotekan's REST endpoints
// ``<stage>/set_config`` and ``<stage>/adcstat``.

export class AirspyGainPanel {
    constructor({app, target, stage}) {
        this.app = app;
        this.state = app.state;
        this.stage = stage;

        const self = this;
        const slider_width = 50;
        // Row layout: vertical gain sliders on the left, ADC stats on the right.
        const wrapper = $("<div/>").css({
            display: "flex", "align-items": "flex-start", gap: "10px",
            padding: "6px 8px", "box-sizing": "border-box",
        }).appendTo($("#" + target));

        const gainwrap = $("<div/>").uniqueId()
            .css({display: "flex", "flex-shrink": "0"})
            .appendTo(wrapper);

        const adcwrap = $("<div/>").uniqueId()
            .css({"font-family": "sans-serif", "font-size": "10pt",
                  "flex-shrink": "0"})
            .appendTo(wrapper);
        $("<p>").text("ADC Stats").css({"font-size": "14pt", "text-align": "center"}).appendTo(adcwrap);
        const adcmean     = $("<div/>").css({position: "relative", left: "30px"}).text("Mean: ").appendTo(adcwrap);
        const adcrms      = $("<p/>"  ).css({position: "relative", left: "30px"}).text("RMS: ").appendTo(adcwrap);
        const adcrailfrac = $("<p/>"  ).css({position: "relative", left: "30px"}).text("Rail %: ").appendTo(adcwrap);

        const check_adcstats = function() {
            self.app.kotekan.stageGet(self.stage, "adcstat")
                .then(r => r.json().then(data => {
                    adcmean.text("Mean: " + data.mean.toFixed(2));
                    adcrms.text("RMS: " + data.rms.toFixed(2));
                    adcrailfrac.text("Rail %: " + (data.railfrac * 100).toFixed(2));
                }));
        };
        const change_gain = function(type, value) {
            self.app.kotekan.stagePost(self.stage, "set_config", {[type]: value})
                .then(check_adcstats);
        };

        const make_slider = (label, key, max) => {
            const col = $("<div style='float:left'/>").width(slider_width)
                .css({"font-family": "sans-serif", "text-align": "center", margin: 2})
                .appendTo(gainwrap);
            $("<p/>").css({margin: 2, "margin-bottom": 15}).text(label).appendTo(col);
            const slider = $("<div/>").uniqueId().appendTo(col).css({margin: "auto"})
                .slider({
                    min: 0, max: max, value: 10, step: 1,
                    orientation: "vertical",
                    slide: function(event, ui) {
                        change_gain(key, ui.value);
                        readout.text(ui.value);
                    },
                });
            const readout = $("<p/>").css({margin: 2}).text("10").appendTo(col);
            return {slider, readout};
        };

        const lna = make_slider("LNA", "gain_lna", 14);
        const mix = make_slider("MIX", "gain_mix", 15);
        const ifg = make_slider("IF",  "gain_if",  15);

        // Read back current settings from kotekan so the sliders start where
        // the producer is actually configured.
        this.app.kotekan.stageGet(this.stage, "get_config")
            .then(r => r.json().then(data => {
                lna.slider.slider("value", data.lna_gain); lna.readout.text(data.lna_gain);
                mix.slider.slider("value", data.mix_gain); mix.readout.text(data.mix_gain);
                ifg.slider.slider("value", data.if_gain);  ifg.readout.text(data.if_gain);
            }));
        check_adcstats();
    }
}
