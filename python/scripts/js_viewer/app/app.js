// Autocorr viewer orchestrator.
//
// Owns the shared state, the bus, the kotekan REST adapter, and the
// WebSocket client. ``start()`` opens the WebSocket; the server replies
// with a ``viewer_config`` JSON, which ``apply_viewer_config()`` consumes
// to build out the page (waterfall + spectrum + the optional panels).

import {Bus} from "./bus.js";
import {KotekanRest} from "./kotekan_rest.js";
import {LayoutManager} from "./layout.js";
import {Socket} from "./socket.js";
import {WaterfallView} from "./waterfall_view.js";
import {CrosscorrView} from "./crosscorr_view.js";
import {SpectrumView} from "./spectrum_view.js";

import {ColorPanel}               from "./panels/color.js";
import {FreqRangePanel}           from "./panels/freq_range.js";
import {BufferControlPanel,
        WaterfallControlPanel}     from "./panels/buffer.js";
import {StartStopPanel}           from "./panels/start_stop.js";
import {LocalRecordPanel}         from "./panels/record.js";
import {BaselinePanel}            from "./panels/baseline.js";
import {AirspyGainPanel}          from "./panels/airspy_gain.js";
import {LagAlignPanel}            from "./panels/lag_align.js";
import {CCERAPointingPanel}       from "./panels/ccera.js";
import {GalaxyViewPanel}          from "./panels/galaxy.js";


function default_state() {
    const cb = new imgPlotter();
    cb.min = 3;
    cb.max = 7;
    return {
        // Set by the first JSON message:
        num_freqs: 1024,
        // Number of visibility streams per frame (1 for autocorr, 4 for
        // crosscorr -- AA/BB/Re{AB*}/Im{AB*}). Each TIMESTEP carries
        // ``nvis * num_freqs`` floats of spectrum data.
        nvis: 1,
        freq_list: null,
        spectrum_baseline: new Array(1024).fill(0),
        // Streamed:
        scroll_data: [],
        timearr: [],
        spectrum: null,
        // Mode the WaterfallView consults to decide what to do with frames:
        // "normal", "stopped", "bandpass", "skip", "idle".
        mode: "normal",
        // Bandpass-mode workspace; the BaselinePanel owns the semantics.
        bandpass_data: [],
        autocal_length: 128,
        skip_length: 20,
        // Waterfall sizing.
        waterfall_buffer_length:         1000,
        waterfall_buffer_max_length:     2500,
        waterfall_buffer_display_length: 300,
        ms_per_datum: 25.,
        // Displayed freq window in MHz; overridden by viewer_config.ui.freq_range_mhz.
        disp_freq: [1417., 1425.],
        // Colormap renderer; shared by WaterfallView and ColorPanel.
        cb,
        // CCERA pointing object; populated by CCERAPointingPanel.
        CCERA: null,
        // Display-time baseline subtraction flag; flipped by BaselinePanel.
        baseline_enabled: false,
    };
}


export class App {
    constructor() {
        this.bus = new Bus();
        this.state = default_state();
        this.kotekan = null;   // populated in apply_viewer_config
        this.socket = null;    // populated in start()
        this.waterfall = null;
        this.spectrum = null;
        this.panels = [];
    }

    start() {
        // LayoutManager + WS get set up immediately; the actual GridStack
        // widgets and the waterfall / spectrum view instances are deferred
        // to ``apply_viewer_config`` because their shape depends on the
        // pipeline mode (autocorr vs. crosscorr).
        this.layout = new LayoutManager({root: "#layout-root"});

        this.socket = new Socket({app: this,
                                  url: "ws://" + location.hostname + ":8539"});
        this._wire_status_banner();
        this.socket.connect();
    }

    // Right-column cards. The mode branch in ``apply_viewer_config`` passes
    // per-mode tweaks: crosscorr drops the Baseline card and needs a taller
    // Airspy card so two AirspyGainPanels fit stacked.
    _add_controls_cards({include_baseline, airspy_h}) {
        let y = 0;
        this.layout.addWidget({mount_id: "display_card", title: "Display",
                               x: 8, y, w: 4, h: 6, min_w: 3, min_h: 6});
        y += 6;
        this.layout.addWidget({mount_id: "buffer_card",  title: "Buffer",
                               x: 8, y, w: 4, h: 3, min_w: 3, min_h: 3});
        y += 3;
        if (include_baseline) {
            this.layout.addWidget({mount_id: "baseline_card", title: "Baseline",
                                   x: 8, y, w: 4, h: 4, min_w: 3, min_h: 4});
            y += 4;
        }
        const ah = airspy_h || 3;
        this.layout.addWidget({mount_id: "airspy_card",  title: "Airspy",
                               x: 8, y, w: 4, h: ah, min_w: 3, min_h: ah});
        y += ah;
        this.layout.addWidget({mount_id: "control_card", title: "Control",
                               x: 8, y, w: 4, h: 2, min_w: 3, min_h: 2});
    }

    // Toolbar banner: surfaces protocol-version mismatches and WS-drop /
    // reconnect notices. Hidden when the connection is healthy.
    _wire_status_banner() {
        const banner = document.getElementById("ws-banner");
        if (!banner) return;
        const show = (text) => { banner.textContent = text; banner.style.display = ""; };
        const hide = () => { banner.style.display = "none"; banner.textContent = ""; };
        this.bus.on("ws:version_mismatch", ({client, server}) => {
            show(`Protocol mismatch: client v${client} vs server v${server} -- reload after upgrading.`);
        });
        this.bus.on("ws:close", (d) => {
            if (d && d.user_initiated) hide();
            else show("Disconnected from server; reconnecting...");
        });
        this.bus.on("ws:reconnect_scheduled", ({delay_ms}) => {
            show(`Disconnected; retrying in ${(delay_ms/1000).toFixed(1)}s...`);
        });
        this.bus.on("ws:open", () => hide());
    }

    apply_viewer_config(cfg) {
        const cfg_mode = cfg.mode || "autocorr";
        const cfg_nvis = cfg.nvis || 1;
        if (this._configured) {
            // Already wired up. A reconnect after the user swapped kotekan
            // pipelines (eg autocorr -> crosscorr) carries a fresh
            // viewer_config with a different shape; the in-page UI was
            // built for the previous shape and can't be safely repurposed
            // (different mount points, different stream count). Easiest
            // recovery: reload, which re-runs ``start()`` against the new
            // server config.
            if (cfg_mode !== this._mode || cfg_nvis !== this.state.nvis) {
                console.warn(
                    `Pipeline changed under us (was ${this._mode} nvis=${this.state.nvis}, ` +
                    `now ${cfg_mode} nvis=${cfg_nvis}); reloading.`);
                const banner = document.getElementById("ws-banner");
                if (banner) {
                    banner.textContent = "Pipeline changed; reloading...";
                    banner.style.display = "";
                }
                // Defer one tick so the banner paints before the reload.
                setTimeout(() => location.reload(), 100);
            }
            return;
        }
        this._configured = true;
        this._mode = cfg_mode;

        const k = cfg.kotekan || {};
        this.kotekan = new KotekanRest({
            host: location.hostname,
            port: k.rest_port || 12048,
            airspy_stages: k.airspy_stages || ["airspy_input"],
            lag_align_stage: k.lag_align_stage || null,
        });

        const ui  = cfg.ui || {};
        const opt = cfg.optional_modules || {};

        // Apply UI defaults that the rest of the wiring will read.
        // Default the *displayed* window to a 10% inset from the full freq
        // range (1416..1426 -> 1417..1425). The bins right at DC / Nyquist
        // tend to read near-zero (which becomes -Inf in dB and shows as a
        // gap in Plotly), and this matches the pre-refactor default. The
        // slider's range stays the full freq_range_mhz so the user can drag
        // the window out if they want to inspect the edges.
        if (ui.freq_range_mhz) {
            const [lo, hi] = ui.freq_range_mhz;
            const margin = (hi - lo) * 0.10;
            this.state.disp_freq = [lo + margin, hi - margin];
        }

        // Server-driven integration interval. WaterfallView's gap detector
        // and the NaN-row time spacing both key off this; if the server
        // doesn't supply one, fall back to the default_state() value.
        if (cfg.frame_period_s && cfg.frame_period_s > 0) {
            this.state.ms_per_datum = cfg.frame_period_s * 1000;
        }
        this.state.nvis = cfg_nvis;

        const is_crosscorr = (cfg_mode === "crosscorr");

        // Layout: autocorr stacks waterfall + spectrum on the left; crosscorr
        // gives the 2x2 waterfall card the full left column (no spectrum).
        if (is_crosscorr) {
            this.layout.addWidget({mount_id: "img_holder", title: "Crosscorr",
                                   x: 0, y: 0, w: 8, h: 18, min_w: 5, min_h: 8});
        } else {
            this.layout.addWidget({mount_id: "img_holder", title: "Waterfall",
                                   x: 0, y: 0, w: 8, h: 10, min_w: 5, min_h: 6});
            this.layout.addWidget({mount_id: "spectrum_holder", title: "Spectral Power",
                                   x: 0, y: 10, w: 8, h: 8, min_w: 4, min_h: 4});
        }
        // Baseline is autocorr-only (crosscorr's cross-baseline is usually
        // ~0). Airspy card needs more vertical room in crosscorr so the
        // per-stage AirspyGainPanels (one per dongle) fit stacked.
        this._add_controls_cards({
            include_baseline: !is_crosscorr,
            airspy_h: is_crosscorr ? 6 : 3,
        });

        // Instantiate the view(s) now that the mount divs exist.
        if (is_crosscorr) {
            this.waterfall = new CrosscorrView({app: this, target: "img_holder"});
        } else {
            this.waterfall = new WaterfallView({app: this, target: "img_holder"});
            this.spectrum  = new SpectrumView ({app: this, target: "spectrum_holder"});
        }

        // Display card: colormap controls + frequency window + waterfall
        // display length (all things that change what gets *shown*).
        this.panels.push(new ColorPanel({
            app: this, target: "display_card",
            color_range: ui.color_range || [-20, 20],
        }));
        this.panels.push(new FreqRangePanel({
            app: this, target: "display_card",
            range: ui.freq_range_mhz || [1416, 1426],
        }));
        this.panels.push(new WaterfallControlPanel({app: this, target: "display_card"}));

        // Buffer card: ring-buffer length + local save.
        this.panels.push(new BufferControlPanel({app: this, target: "buffer_card"}));
        this.panels.push(new LocalRecordPanel  ({app: this, target: "buffer_card"}));

        if (!is_crosscorr) {
            // Spectrum-pane extras + baseline tools are autocorr-only for
            // now; baseline subtraction would need per-stream state in
            // crosscorr mode, and the spectrum view assumes a single stream.
            this.spectrum.add_excess({target: "spectrum_holder"});

            // Pass `autocal_stage: opt.airspy_controls ? this.kotekan.airspy_stages[0] : null`
            // to re-enable BaselinePanel's "Take 1416MHz Bandpass" auto-cal
            // button (the code lives in BaselinePanel._add_autocal). Disabled
            // by default: it briefly retunes the airspy off whatever you're
            // looking at and isn't useful for everyday viewing.
            this.panels.push(new BaselinePanel({
                app: this, target: "baseline_card",
                autocal_stage: null,
            }));
        }

        // Airspy card: per-stage gain + ADC stats.
        if (opt.airspy_controls) {
            for (const stage of this.kotekan.airspy_stages) {
                this.panels.push(new AirspyGainPanel({
                    app: this, target: "airspy_card", stage}));
            }
        }

        // Lag-align card -- crosscorr only, and only if the server named a
        // lag-align stage. Full-width strip below both columns since it
        // carries a correlation plot that wants the horizontal room.
        if (is_crosscorr && this.kotekan.lag_align_stage) {
            this.layout.addWidget({mount_id: "lag_align_card", title: "Lag Align",
                                   x: 0, y: 18, w: 12, h: 6, min_w: 4, min_h: 4});
            this.panels.push(new LagAlignPanel({
                app: this, target: "lag_align_card",
                lag_align_stage: this.kotekan.lag_align_stage,
                airspy_stages: this.kotekan.airspy_stages,
            }));
        }

        // Pointing card -- only added if the server announces CCERA controls.
        if (opt.ccera_pointing) {
            this.layout.addWidget({mount_id: "pointing_card", title: "Pointing",
                                   x: 8, y: 18, w: 4, h: 3, min_w: 3, min_h: 3});
            this.panels.push(new CCERAPointingPanel({
                app: this, target: "pointing_card"}));
        }

        // Control card: master start / stop.
        this.panels.push(new StartStopPanel({app: this, target: "control_card"}));

        if (opt.galaxy_view && opt.galaxy_view_url) {
            this.layout.addWidget({mount_id: "gal_viewer", title: "Galaxy View",
                                   x: 0, y: 18, w: 12, h: 8, min_w: 4, min_h: 4});
            this.panels.push(new GalaxyViewPanel({
                app: this, target: "gal_viewer",
                image_url: opt.galaxy_view_url,
            }));
        }

        // Now that the full widget set exists, apply any saved layout the
        // user customised in a previous session.
        this.layout.restore_from_storage();
    }
}
