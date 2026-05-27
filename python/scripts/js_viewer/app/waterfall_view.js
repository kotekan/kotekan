// The main "waterfall" plot: a scrolling 2D canvas where each row is a power
// spectrum, colour-mapped through ``app.state.cb`` (an ``imgPlotter`` from
// lib/img_plotting.js, which is loaded as a global).
//
// Pulls its data from ``app.state.scroll_data`` and ``app.state.timearr``; the
// ``Socket`` module appends to those on each TIMESTEP and emits a
// ``state:redraw_requested`` event that WaterfallView listens for.
//
// Layout
// ------
// The view fills its GridStack card via a CSS grid:
//
//   +---------+--------------------------+
//   | yaxis   | scroll_canvas            |
//   | (svg)   |                          |
//   |         |                          |
//   |         |                          |
//   +---------+--------------------------+
//   | (empty) | freq axis (svg)          |
//   +---------+--------------------------+
//
// The y-axis column is a fixed pixel width (``margin[0]``); the x-axis row
// is a fixed pixel height. Everything else flexes; a ResizeObserver on the
// host element rescales the d3 axes and re-renders the canvas.

const Y_AXIS_WIDTH    = 100;   // px, room for "Time" rotated label + tick text
const X_AXIS_HEIGHT   = 50;    // px, room for tick text + "Frequency [MHz]" with breathing space
const RIGHT_PAD       = 30;    // px, keeps the rightmost x-tick label inside the card
const Y_LABEL_FONTSIZE = 16;
const X_LABEL_FONTSIZE = 16;

// Time-gap detection (missing-data rendering). A frame with timestamp more
// than ``GAP_THRESHOLD`` integration intervals after the previous one is
// treated as "frames went missing between then and now"; NaN-filled rows
// are inserted into scroll_data so the waterfall paints a grey strip
// across the gap. Row count is capped at ``buffer_length - 1`` (leaving
// room for the new real frame); the synthetic timestamps are spaced to
// span the *full* gap duration, so a long disconnect stretches the gap
// rows but keeps ``timearr`` monotonically uniform -- d3's linear time
// axis then stays consistent without us having to redrive it from row
// timestamps.
const GAP_THRESHOLD       = 1.8;

// Wall-clock staleness threshold. If a frame's timestamp is more than this
// many seconds behind ``Date.now()``, we drop it without appending to
// scroll_data and without advancing ``_last_frame_time``. Two ways to get
// stale frames:
//   * the server fell back catching up after backpressure, and a queue is
//     draining,
//   * the browser fell behind processing inbound frames (cheap CPU bound).
// In both cases, watching ancient data play out is misleading -- we'd
// rather skip to live and let the gap detector paint the missed span grey
// once a fresh frame arrives.
const STALE_FRAME_AGE_S   = 2.0;

// Max waterfall redraw rate. _dodraw is the per-frame hot path -- it
// iterates the whole scroll_data buffer to compute img_mean and writes
// num_freqs * ntimes pixels into an ImageData. Capping draws at 10 Hz
// keeps the browser responsive even when kotekan is pushing 40 Hz of new
// frames. Lower this further if the canvas still feels heavy.
const MIN_DRAW_INTERVAL_MS = 100;

export class WaterfallView {
    /**
     * @param {Object}  args
     * @param {Object}  args.app
     * @param {string}  args.target            DOM id to mount into
     * @param {Function} [args.stream_extractor]  (raw_row, num_freqs) -> Float32Array.
     *   Slices one visibility stream out of a multi-stream frame; default is identity
     *   (autocorr stores num_freqs floats per row, no extraction needed).
     * @param {Function} [args.value_transform]   (v) -> v applied per pixel; default
     *   is the dB conversion ``10 * log10(v)`` autocorr has always used. Phase
     *   plots pass identity, mag plots leave the default.
     * @param {boolean}  [args.enable_baseline=true]  Whether display-time baseline
     *   subtraction (the BaselinePanel toggle) should apply here. Phase plots
     *   want this off.
     * @param {boolean}  [args.is_primary=true]  Only the primary view manages
     *   ``state.scroll_data`` / ``state.timearr`` updates + gap detection; in
     *   crosscorr mode the four sub-views share one buffer, so only one runs
     *   ``_on_frame``. The primary also publishes its mean spectrum into
     *   ``state.spectrum`` for SpectrumView to consume.
     * @param {Object}   [args.cb]              imgPlotter to colour through.
     *   Defaults to ``state.cb`` (the shared dB-scale instance the color
     *   slider drives). Phase views pass a local fixed-range cb.
     */
    constructor({app, target,
                 stream_extractor, value_transform,
                 enable_baseline, is_primary, cb}) {
        this.app = app;
        this.bus = app.bus;
        this.state = app.state;
        this.stream_extractor = stream_extractor || ((row, _nf) => row);
        this.value_transform  = value_transform  || ((v) => 10 * Math.log10(v));
        this.enable_baseline  = (enable_baseline !== false);
        this.is_primary       = (is_primary       !== false);
        this.cb               = cb || this.state.cb;

        // The host div fills its panel-body card; everything inside is laid
        // out by CSS grid (see file header).
        // Three columns: y-axis (fixed), plot (flex), right-side spacer
        // (fixed) so the rightmost x-tick label has room and an in-card
        // scrollbar wouldn't sit on top of the waterfall.
        this.jqcontainer = $("#" + target).css({
            position: "relative",
            display: "grid",
            "grid-template-rows":    `1fr ${X_AXIS_HEIGHT}px`,
            "grid-template-columns": `${Y_AXIS_WIDTH}px 1fr ${RIGHT_PAD}px`,
            height: "100%",
            "min-height": "0",
            width:  "100%",
        });

        // Cells.
        const yaxis_cell = $("<div/>").uniqueId()
            .css({"grid-row": "1", "grid-column": "1", position: "relative",
                  "font-size": "10pt", overflow: "hidden"})
            .attr("class", "axis")
            .appendTo(this.jqcontainer);

        const canvas_cell = $("<div/>").uniqueId()
            .css({"grid-row": "1", "grid-column": "2", position: "relative",
                  overflow: "hidden"})
            .appendTo(this.jqcontainer);

        const freq_cell = $("<div/>").uniqueId()
            .css({"grid-row": "2", "grid-column": "2", position: "relative",
                  "font-size": "10pt", overflow: "visible"})
            .attr("class", "axis")
            .appendTo(this.jqcontainer);

        // The waterfall canvas itself fills the canvas cell. Its bitmap
        // dimensions get set in _dodraw to match the current data shape;
        // the CSS dimensions are always 100%/100% and the browser scales
        // the bitmap to fit (with image-rendering: pixelated so individual
        // bin/frame cells stay crisp).
        this.scroll_canvas = $("<canvas/>")
            .attr("width", this.state.num_freqs)
            .attr("height", this.state.waterfall_buffer_display_length)
            .css({position: "absolute", inset: 0, width: "100%", height: "100%",
                  "image-rendering": "pixelated"})
            .appendTo(canvas_cell);

        // Y axis (time). d3.axis().orient('left') draws ticks pointing left
        // from the <g>'s origin; translating the <g> to x=Y_AXIS_WIDTH puts
        // the axis line flush against the canvas's left edge.
        this.yaxis_scale = d3.time.scale.utc().range([0, 1]);    // range set below
        this.yaxis = d3.svg.axis().scale(this.yaxis_scale).orient("left")
            .tickFormat(d3.time.format("%H:%M:%S"));
        const yaxis_svg = d3.select("#" + yaxis_cell[0].id).append("svg")
            .attr("width", "100%").attr("height", "100%")
            .style("position", "absolute").style("top", "0").style("left", "0");
        this.yaxisplot = yaxis_svg.append("g")
            .attr("transform", `translate(${Y_AXIS_WIDTH}, 0)`);
        // The vertical "Time" label hangs off the left of the axis line.
        this.yaxis_label = yaxis_svg.append("text")
            .attr("text-anchor", "middle")
            .attr("font-size", Y_LABEL_FONTSIZE)
            .attr("transform", "rotate(-90)")
            .text("Time");

        // X (freq) axis: a <g> at the top of the freq cell, ticks pointing
        // down. Like the y-axis, the axis line itself stays flush with the
        // canvas above it.
        this.freq_scale = d3.scale.linear().range([0, 1]).domain([1, 2]);
        this.freq_axis = d3.svg.axis().scale(this.freq_scale).orient("bottom");
        const freq_svg = d3.select("#" + freq_cell[0].id).append("svg")
            .attr("width", "100%").attr("height", "100%")
            .style("position", "absolute").style("top", "0").style("left", "0")
            .style("overflow", "visible");
        this.freq_axisplot = freq_svg.append("g")
            .attr("transform", "translate(0, 0)");
        this.freq_label = freq_svg.append("text")
            .attr("text-anchor", "middle")
            .attr("font-size", X_LABEL_FONTSIZE)
            .text("Frequency [MHz]");

        // Cache DOM nodes for resize calls.
        this._canvas_cell = canvas_cell[0];
        this._freq_cell   = freq_cell[0];
        this._yaxis_cell  = yaxis_cell[0];
        this._yaxis_svg   = yaxis_svg.node();
        this._freq_svg    = freq_svg.node();

        // Initial sizing.
        this._on_resize();

        // Track redraw state.
        this._last_draw = new Date().getTime();
        this._r = 0;

        // Timestamp of the most recent real frame appended to scroll_data,
        // used by the gap detector in _on_frame. Cleared on user-initiated
        // disconnects so resuming after Stop -> Start doesn't insert a gap
        // for the user's own pause.
        this._last_frame_time = null;

        // Re-render whenever the card changes shape.
        new ResizeObserver(() => this._on_resize()).observe(this.jqcontainer[0]);

        this.bus.on("state:redraw_requested",   () => this.draw());
        this.bus.on("state:frame_received",     (d) => this._on_frame(d));
        this.bus.on("state:freq_list_changed",  () => this.draw());
        this.bus.on("state:buffer_length_changed", () => this._trim_buffers());
        this.bus.on("ws:close", (d) => {
            if (d && d.user_initiated) this._last_frame_time = null;
        });
    }

    _on_resize() {
        const canvas_w = this._canvas_cell.clientWidth;
        const canvas_h = this._canvas_cell.clientHeight;
        const freq_w   = this._freq_cell.clientWidth;

        this.yaxis_scale.range([0, canvas_h]);
        this.freq_scale .range([0, freq_w]);

        // The Time label sits halfway down the axis, rotated -90deg. After
        // rotation, the original x maps to -y in screen space, so put the
        // label at x = -canvas_h/2 (which lands at vertical centre) and
        // y at a small inset from the left edge.
        this.yaxis_label
            .attr("x", -canvas_h / 2)
            .attr("y", Y_LABEL_FONTSIZE + 4);

        // Label sits at the bottom of the freq cell, leaving a ~25px gap
        // between the d3 tick text (which lives near y=20) and the label.
        this.freq_label
            .attr("x", freq_w / 2)
            .attr("y", X_AXIS_HEIGHT - 4);

        // Drop a tick density hint that doesn't crowd the y axis at small
        // heights. d3 v3's ``ticks(N)`` is advisory.
        this.yaxis.ticks(Math.max(2, Math.floor(canvas_h / 60)));
        this.yaxisplot.call(this.yaxis);
        this.freq_axisplot.call(this.freq_axis);

        this.draw();
    }

    _on_frame({timestamp, data}) {
        // Only the primary view owns ``state.scroll_data`` / ``state.timearr``;
        // secondary crosscorr sub-views read those but mustn't double-append.
        if (!this.is_primary) return;
        const s = this.state;
        if (s.mode !== "normal" && s.mode !== "skip") return;

        // Drop frames whose timestamp is too far behind wall-clock. This is
        // what catches the case where the browser fell behind processing
        // (the server already paused, but a backlog drained into the page's
        // WS buffer before kotekan died). We deliberately do NOT advance
        // ``_last_frame_time`` here, so when fresh frames resume the gap
        // detector below sees the full wall-clock lag and paints grey.
        const wall_now = Date.now() / 1000;
        if (wall_now - timestamp > STALE_FRAME_AGE_S) {
            this._stale_dropped = (this._stale_dropped || 0) + 1;
            // Throttle the console log to once per second so a long drain
            // doesn't spam.
            if (!this._stale_log_at || wall_now - this._stale_log_at > 1.0) {
                console.warn(`Dropping stale frame (${(wall_now - timestamp).toFixed(1)}s behind wall-clock); ${this._stale_dropped} dropped so far`);
                this._stale_log_at = wall_now;
            }
            return;
        }

        // If we missed frames since the last one we appended, pad the gap
        // with NaN-filled rows so the waterfall paints a grey strip there.
        // ``ms_per_datum`` is the nominal integration interval (negotiated
        // from the server in apply_viewer_config); gaps larger than
        // GAP_THRESHOLD of that interval count as "frames lost".
        //
        // Spacing strategy: every NaN row -- and the new real frame -- is
        // placed at exactly ``expected_s`` apart, anchored so the *last*
        // NaN sits one interval before the new real frame's timestamp.
        // That keeps ``timearr`` uniformly spaced across the entire buffer
        // (real -> NaN -> new real, all at the same cadence), so d3's
        // linear time scale lines up with row positions and the gap-edge
        // labels stay honest. When the cap clamps the row count (very
        // long outages), we visually "lose" the earliest part of the gap
        // rather than introduce non-uniform spacing.
        if (this._last_frame_time != null && s.ms_per_datum > 0) {
            const expected_s = s.ms_per_datum / 1000;
            const gap_s = timestamp - this._last_frame_time;
            if (gap_s > GAP_THRESHOLD * expected_s) {
                let missing = Math.round(gap_s / expected_s) - 1;
                const cap = Math.max(1, s.waterfall_buffer_length - 1);
                if (missing > cap) missing = cap;
                for (let k = missing; k >= 1; k--) {
                    // ``data.length`` so gap rows match the multi-stream wire
                    // size in crosscorr mode (nvis * num_freqs), not just one
                    // stream's worth.
                    const row = new Float32Array(data.length);
                    row.fill(NaN);
                    s.scroll_data.push(row);
                    s.timearr.push(timestamp - k * expected_s);
                }
            }
        }

        s.scroll_data.push(data);
        s.timearr.push(timestamp);
        this._last_frame_time = timestamp;
        this._trim_buffers();
    }

    _trim_buffers() {
        const s = this.state;
        while (s.scroll_data.length > s.waterfall_buffer_length) s.scroll_data.shift();
        while (s.timearr.length     > s.waterfall_buffer_length) s.timearr.shift();
    }

    draw() {
        const s = this.state;
        if (s.mode !== "normal" && s.mode !== "stopped" && s.mode !== "skip") return;
        const now = new Date().getTime();
        if (now - this._last_draw < MIN_DRAW_INTERVAL_MS) return;
        this._last_draw = now;
        if (this._r > 0) return;
        // ``try/finally`` so a transient exception inside _dodraw (eg an
        // empty buffer or out-of-range index just after a reconnect) can't
        // strand ``_r`` non-zero, which would freeze every later redraw.
        this._r = requestAnimationFrame(() => {
            try { this._dodraw(); }
            catch (e) { console.error("WaterfallView._dodraw threw:", e); }
            finally { this._r = 0; }
        });
    }

    _dodraw() {
        const s = this.state;
        if (!s.freq_list || s.freq_list.length < 2) return;
        if (s.scroll_data.length === 0) return;

        const scd = s.scroll_data;
        const nfreq = s.num_freqs;
        const ntimes = Math.min(scd.length, s.waterfall_buffer_display_length);

        // Extract the per-stream view of each row once and reuse for both the
        // mean computation and the canvas paint. For autocorr / AA / BB the
        // extractor returns a subarray (no copy); for |AB| and phase it
        // allocates a fresh Float32Array per row.
        const extracted = new Array(scd.length);
        for (let j = 0; j < scd.length; j++) {
            extracted[j] = this.stream_extractor(scd[j], nfreq);
        }

        // Per-bin mean across the in-buffer time range. Gap rows have NaN
        // entries -- the ``v === v`` test skips them so the mean stays valid
        // even when the buffer holds a missing-data strip.
        const img_mean = new Float32Array(nfreq);
        const img_count = new Uint32Array(nfreq);
        for (let j = 0; j < extracted.length; j++) {
            const row = extracted[j];
            for (let i = 0; i < nfreq; i++) {
                const v = row[i];
                if (v === v) { img_mean[i] += v; img_count[i]++; }
            }
        }
        for (let i = 0; i < nfreq; i++) {
            img_mean[i] = img_count[i] > 0 ? img_mean[i] / img_count[i] : NaN;
        }
        // SpectrumView reads ``state.spectrum``; only the primary view writes
        // it so multiple crosscorr sub-views don't fight over it.
        if (this.is_primary) s.spectrum = img_mean;

        // Pick the displayed bin window. Clamp because ``disp_freq`` can sit
        // slightly outside ``fl[0]..fl[last]`` (slider range is bin-edges,
        // ``fl`` is bin-centres).
        const fl = s.freq_list;
        const freq_sc = fl[fl.length - 1] - fl[0];
        const freq_lo = Math.max(0,     Math.floor(nfreq * (s.disp_freq[0] - fl[0]) / freq_sc));
        const freq_hi = Math.min(nfreq, Math.ceil (nfreq * (s.disp_freq[1] - fl[0]) / freq_sc));
        let nbins = Math.round(freq_hi - freq_lo);
        if (nbins < 1) nbins = 1;

        // Set the bitmap dimensions; CSS dimensions stay at 100%/100% so
        // the browser scales nbins x ntimes frames to fill the canvas
        // cell. ``image-rendering: pixelated`` keeps the per-bin/frame
        // cells crisp.
        this.scroll_canvas.attr("width", nbins);
        this.scroll_canvas.attr("height", ntimes);

        const c = this.scroll_canvas[0].getContext("2d");
        c.imageSmoothingEnabled = false;

        const imageData = c.createImageData(nbins, ntimes);
        const disp_start = Math.max(0, scd.length - ntimes);
        const baseline_on = this.enable_baseline && !!s.baseline_enabled;
        const baseline = s.spectrum_baseline;
        const tx = this.value_transform;
        const cb = this.cb;
        for (let j = disp_start; j < scd.length; j++) {
            const row = extracted[j];
            for (let i = 0; i < nbins; i++) {
                const ii = i + freq_lo;
                let v = tx(row[ii]);
                if (baseline_on) v -= tx(baseline[ii]);
                cb.setPixel(imageData, i, j - disp_start, v);
            }
        }
        c.putImageData(imageData, 0, 0);

        // Update axes to match the displayed data.
        this.freq_scale.domain([fl[freq_lo], fl[Math.min(freq_hi, fl.length - 1)]]);
        this.freq_axisplot.call(this.freq_axis);
        this.yaxis_scale.domain([
            new Date(s.timearr[s.timearr.length - ntimes] * 1e3),
            new Date(s.timearr[s.timearr.length - 1]      * 1e3),
        ]);
        this.yaxisplot.call(this.yaxis);

        // SpectrumView keys off this to repaint its plots.
        this.bus.emit("state:waterfall_drawn");
    }
}
