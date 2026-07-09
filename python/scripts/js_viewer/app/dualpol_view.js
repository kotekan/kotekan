// Side-by-side waterfalls for the ARO dual-pol pipeline (nvis == 2).
//
// ARO's computeDualpolPower emits two power-spectrum streams per integration
// -- XX and YY (Stokes ids -5, -6). This view mirrors CrosscorrView but with
// a 1x2 layout: one WaterfallView per pol, sharing the Display card's dB
// colorbar (``state.cb``) since both are power-like quantities on the same
// scale. The first ("primary") view owns scroll_data + gap detection; the
// second just reads the same shared buffer.
//
// The frame from the server is ``nvis * num_freqs`` floats laid out as two
// contiguous num_freqs blocks: [XX | YY].

import {WaterfallView} from "./waterfall_view.js";

export class DualpolView {
    constructor({app, target, labels}) {
        this.app = app;
        const pol_labels = labels && labels.length ? labels : ["XX", "YY"];

        const host = $("#" + target);
        host.css({
            display: "grid",
            "grid-template-columns": "1fr 1fr",
            "grid-template-rows": "1fr",
            gap: "6px",
            padding: "4px",
            height: "100%",
            "min-height": "0",
            "box-sizing": "border-box",
        });

        // Each cell: a label strip + a mount div the WaterfallView attaches to.
        // ``min-height: 0`` lets a CSS-grid cell shrink past its content when
        // the card is resized (same trick CrosscorrView uses).
        const make_cell = (label) => {
            const cell = $("<div/>")
                .css({position: "relative", "min-height": "0", "min-width": "0",
                      display: "flex", "flex-direction": "column",
                      border: "1px solid #ddd", "border-radius": "3px",
                      overflow: "hidden"})
                .appendTo(host);
            $("<div/>")
                .text(label)
                .css({"font-family": "sans-serif", "font-size": "12px",
                      "font-weight": "600", padding: "2px 6px",
                      background: "#eef2f6", "border-bottom": "1px solid #d0d7de"})
                .appendTo(cell);
            const mount = $("<div/>").uniqueId()
                .css({flex: "1 1 0", "min-height": "0"})
                .appendTo(cell);
            return mount[0].id;
        };

        // Stream extractors: contiguous num_freqs blocks, [pol0 | pol1].
        // subarray() is a view, no allocation.
        const extract_p0 = (row, nf) => row.subarray(0 * nf, 1 * nf);
        const extract_p1 = (row, nf) => row.subarray(1 * nf, 2 * nf);

        const p0_id = make_cell(pol_labels[0]);
        const p1_id = make_cell(pol_labels[1] || "pol1");

        // Primary pol: owns scroll_data + gap detection.
        this.pol0 = new WaterfallView({
            app, target: p0_id,
            stream_extractor: extract_p0,
            enable_baseline: false, is_primary: true,
        });
        this.pol1 = new WaterfallView({
            app, target: p1_id,
            stream_extractor: extract_p1,
            enable_baseline: false, is_primary: false,
        });
    }
}
