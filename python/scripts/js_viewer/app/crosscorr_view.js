// 2x2 grid of WaterfallView instances for the crosscorr pipeline.
//
// Each frame from the server is ``nvis * num_freqs`` floats long; we slice
// it into AA, BB, |AB|, and arg(AB) sub-views. The four ``WaterfallView``
// children share ``state.scroll_data`` -- only the first ("primary") view
// runs frame-append + gap-detection; the others just read.
//
// Colormap sharing:
//   * AA / BB / |AB| share ``state.cb`` -- the dB-scale colorbar the
//     Display card's slider drives. They're all power-like quantities, so
//     it makes sense to give them one knob.
//   * arg(AB) ranges over [-pi, pi] and isn't dB-meaningful; it gets a
//     local fixed-range colorbar instead.

import {WaterfallView} from "./waterfall_view.js";

export class CrosscorrView {
    constructor({app, target}) {
        this.app = app;
        const host = $("#" + target);
        host.css({
            display: "grid",
            "grid-template-columns": "1fr 1fr",
            "grid-template-rows":    "1fr 1fr",
            gap: "6px",
            padding: "4px",
            height: "100%",
            "min-height": "0",
            "box-sizing": "border-box",
        });

        // Each cell: a small label strip on top + a mount div the
        // WaterfallView attaches to. ``min-height: 0`` + ``flex: 1 1 0``
        // is what lets a CSS-grid cell shrink past its content's natural
        // size when the card is resized.
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

        const aa_id = make_cell("AA");
        const bb_id = make_cell("BB");
        const ab_id = make_cell("|AB|");
        const ph_id = make_cell("∠AB");   // ∠

        // Stream extractors. The frame is laid out as four contiguous
        // num_freqs-float blocks in [AA | BB | Re{AB*} | Im{AB*}] order
        // (matching networkPowerStream's per-element output). Subarrays for
        // the bare streams cost no allocation; |AB| and arg(AB) compose
        // from Re/Im so they allocate a fresh row.
        const extract_aa  = (row, nf) => row.subarray(0 * nf, 1 * nf);
        const extract_bb  = (row, nf) => row.subarray(1 * nf, 2 * nf);
        const extract_mag = (row, nf) => {
            const re = row.subarray(2 * nf, 3 * nf);
            const im = row.subarray(3 * nf, 4 * nf);
            const out = new Float32Array(nf);
            for (let i = 0; i < nf; i++) out[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
            return out;
        };
        const extract_phase = (row, nf) => {
            const re = row.subarray(2 * nf, 3 * nf);
            const im = row.subarray(3 * nf, 4 * nf);
            const out = new Float32Array(nf);
            for (let i = 0; i < nf; i++) out[i] = Math.atan2(im[i], re[i]);
            return out;
        };

        // Phase gets its own colorbar with a fixed [-pi, pi] range. The
        // shared ColorPanel slider (which writes state.cb) is in dB and
        // would map badly here.
        const phase_cb = new imgPlotter();
        phase_cb.min = -Math.PI;
        phase_cb.max =  Math.PI;
        phase_cb.gradientScale(phase_cb.colormaps.jet);

        // Primary view: AA. It owns scroll_data + gap detection.
        // Baseline subtraction is left off across the board for crosscorr
        // until the BaselinePanel learns per-stream baselines.
        this.aa = new WaterfallView({
            app, target: aa_id,
            stream_extractor: extract_aa,
            enable_baseline: false, is_primary: true,
        });
        this.bb = new WaterfallView({
            app, target: bb_id,
            stream_extractor: extract_bb,
            enable_baseline: false, is_primary: false,
        });
        this.mag = new WaterfallView({
            app, target: ab_id,
            stream_extractor: extract_mag,
            enable_baseline: false, is_primary: false,
        });
        this.phase = new WaterfallView({
            app, target: ph_id,
            stream_extractor: extract_phase,
            value_transform: (v) => v,     // already radians, no dB
            enable_baseline: false, is_primary: false,
            cb: phase_cb,
        });
    }
}
