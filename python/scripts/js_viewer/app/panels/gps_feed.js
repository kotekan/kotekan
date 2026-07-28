// Shared GNSS data feed for the sky + detections panels: ONE poll loop merging
//   * /gps_sky                 (viewer server)  -> {prn, az, el, const} positions
//   * <search>/get_detections  (kotekan REST)   -> acquired PRNs + search SNR
//   * <combiner>/get_status    (kotekan REST)   -> per-PRN amplitudes / deep / dop / coh
//   * <airspy>/adcstat         (kotekan REST)   -> ADC rms
// per constellation chain, keyed "G12"/"E25"/"C19". Panels subscribe with
// feed.on(cb) and re-render on every merged tick; per-constellation visibility
// toggles live here too (persisted), so the sky and the table always agree.

const POLL_MS = 1500;
const AMP_LOCK = 0.30;      // |A| fallback lock (only where no significance is reported)
const SNR_LOCK = 3.0;       // significance (sigma above noise) for a lock
const PREFS_KEY = "gps_viewer_prefs_v1";

// Constellation identity: display name (sky legend), record period (for C/N0 = x / t_rec) and
// the display colour (matches the matplotlib composite-map palette G/E/C = blue/orange/red so
// plots and viewer read the same). These are the L1 DEFAULTS; the server tells us the actual
// band via /wsport (configure_chains below) so an L2C/L5 viewer names the right signal. A 404
// on a missing gal_/bds_ stage just skips that chain.
export let CHAINS = [
    {tag: "G", name: "GPS L1 C/A",  t_rec: 1e-3,  color: "#4d9de0"},
    {tag: "E", name: "Galileo E1C", t_rec: 4e-3,  color: "#e8923c"},
    {tag: "C", name: "BeiDou B1C",  t_rec: 10e-3, color: "#d64550"},
];
export const chain_color = tag =>
    (CHAINS.find(c => c.tag === tag) || {}).color || "#8a8f98";

let _active_feed = null;
// Swap the L1 default chain table for the band this viewer actually serves (delivered by the
// server via /wsport: L2C = [GPS L2C], L5 = [GPS L5, Galileo E5a, BeiDou B2a]). Legend names,
// colours and C/N0 t_rec all follow. Idempotent; safe to call before or after the feed exists.
export function configure_chains(defs) {
    if (!Array.isArray(defs) || !defs.length) return;
    CHAINS = defs;
    if (_active_feed) _active_feed._reconfigure();
}

// UNIFIED viewer (2026-07-28): one row per satellite, every signal it carries side by side.
// SIGNALS is the full inventory from the server's /wsport (UNIFIED_SIGNALS): each entry is one
// signal {tag, band ("L1"/"L2"/"L5"), col, name, combiner (absolute stage), search|null,
// t_rec, peel}. null = per-band mode (the historical path). RF_BANDS drives the spectrum
// selector. The feed keys satellites by tag+prn ACROSS bands and hangs each signal's metrics
// off r.sig_by[combiner]; the table renders one column per signal.
export let SIGNALS = null;
export let RF_BANDS = null;
export function configure_signals(sigs, rf) {
    if (!Array.isArray(sigs) || !sigs.length) { SIGNALS = null; return; }
    SIGNALS = sigs.map(s => Object.assign({key: s.combiner}, s));
    RF_BANDS = Array.isArray(rf) && rf.length ? rf : null;
    if (_active_feed) _active_feed._reconfigure();
}

// One combiner get_status record -> the derived per-signal metrics the panels display. Pulled
// out of _merge so the unified path (per-signal) and the legacy path (one set per row) compute
// them identically. t_rec sets the incoherent C/N0 density; has_peel gates the peel column.
export function signal_metrics(s, t_rec) {
    const m = {
        amp: s.amplitude || 0, coh: s.coh_amplitude || 0, deep: s.deep_amplitude || 0,
        dbi: s.deep_amplitude || s.unbiased_amplitude || 0,
        dop: s.doppler_hz != null ? s.doppler_hz : null,
        coh_s: s.coherence_s != null ? s.coherence_s : null,
        deep_snr: s.deep_snr || 0, dr: s.deep_records || 0,
        cn0: null, cn0_coh: null, peel_db: null, peel_bound: false,
    };
    // significance: deep counts only where the combiner certified it beat its rectification
    // floor (coherence_s > 0); a floored deep (~7-12 sigma) is noise wearing a lock's clothes.
    m.sig = (s.coherence_s || 0) > 0
        ? Math.max(s.deep_snr || 0, s.amp_snr || 0) : (s.amp_snr || 0);
    // coherent C/N0 (dB-Hz) = 10 log10(deep_snr^2 / T_coh), defined only where certified.
    if (m.coh_s > 0 && m.deep_snr > 0)
        m.cn0_coh = 20 * Math.log10(m.deep_snr) - 10 * Math.log10(m.coh_s);
    // incoherent C/N0 (dB-Hz, pipeline zero-point): x = u^2/(a^2-u^2), density x/t_rec.
    const a = s.amplitude || 0, u = s.unbiased_amplitude || 0;
    if (a > u && u > 0) m.cn0 = 10 * Math.log10(((u * u) / (a * a - u * u)) / t_rec);
    // peel depth (dB) = 20 log10(deep/peel_deep), valid only where the PRIMARY deep cleared
    // its floor; at-floor residual -> a lower bound (the detection-limit ratio), rendered ">=".
    const pdeep = s.peel_deep || 0, psnr = s.peel_deep_snr || 0;
    const dfl = s.deep_floor || 0, dsnr = s.deep_snr || 0;
    if (pdeep > 0 && m.deep > 0 && dfl > 0 && dsnr > 1.1 * dfl) {
        m.peel_bound = psnr <= 1.1 * dfl;
        m.peel_db = m.peel_bound ? 20 * Math.log10(dsnr / dfl)
                                 : 20 * Math.log10(m.deep / pdeep);
    }
    return m;
}

export class GpsFeed {
    constructor({app, search_stage, combiner_stage, airspy_stage}) {
        this.app = app;
        this.airspy_stage = airspy_stage || "airspy_in";
        this._search_stage = search_stage;
        this._combiner_stage = combiner_stage;
        this._listeners = [];
        this._inflight = false;
        this._last = {};
        this.vis = {};
        this._prefs = null;
        try { this._prefs = JSON.parse(localStorage.getItem(PREFS_KEY)); } catch (e) { /* fresh */ }
        this.unified = false;
        this.signals = null;
        this._rebuild();                      // sets chains/signals/vis from CHAINS(+SIGNALS)
        _active_feed = this;                  // configure_chains/_signals re-target this instance
        this._tick();
        this._timer = setInterval(() => this._tick(), POLL_MS);
    }

    // Pick per-band vs unified from the module SIGNALS, and (re)build the descriptor lists +
    // per-constellation visibility. chains stays the per-CONSTELLATION list (sky legend, colours,
    // G/E/C vis chips) in BOTH modes; signals is the flat 9-signal list, non-null only unified.
    _rebuild() {
        this.chains = this._build_chains();
        this.unified = Array.isArray(SIGNALS) && SIGNALS.length > 0;
        this.signals = this.unified ? this._build_signals() : null;
    }

    // Unified signal descriptors: the server inventory verbatim (absolute stage names), with
    // visibility inherited from the constellation tag (one G/E/C toggle hides all that sat's
    // signals -- the row itself). t_rec/peel/band/col ride along for the table + C/N0.
    _build_signals() {
        for (const s of SIGNALS)
            if (!(s.tag in this.vis))
                this.vis[s.tag] = (this._prefs && this._prefs.vis && s.tag in this._prefs.vis)
                    ? !!this._prefs.vis[s.tag] : true;
        return SIGNALS.map(s => Object.assign({}, s));
    }

    // Build the per-chain descriptors from the current module CHAINS, merging in the kotekan
    // stage names (constant across bands: gps_/gal_/bds_). Visibility defaults ON per tag,
    // overridden by any persisted preference. Rerun by _reconfigure when the band table swaps.
    _build_chains() {
        // gps_* names throughout (symmetric with gal_*/bds_*); KotekanRest.resolveStage
        // maps them onto the bare search/combiner spelling on older configs.
        const chains = CHAINS.map(c => Object.assign({}, c, c.tag === "G"
            ? {search: c.search || this._search_stage || "gps_search",
               combiner: c.combiner || this._combiner_stage || "gps_combiner"}
            : {search: c.search || (c.tag === "E" ? "gal_search" : "bds_search"),
               combiner: c.combiner || (c.tag === "E" ? "gal_combiner" : "bds_combiner")}));
        for (const c of chains)
            if (!(c.tag in this.vis))
                this.vis[c.tag] = (this._prefs && this._prefs.vis && c.tag in this._prefs.vis)
                    ? !!this._prefs.vis[c.tag] : true;
        return chains;
    }

    // The server delivered this band's tables (configure_chains / configure_signals) after
    // construction: rebuild and re-render so the legend/colours/columns reflect the real config.
    _reconfigure() {
        this._rebuild();
        this._emit();
    }

    on(cb) { this._listeners.push(cb); }

    set_vis(tag, on) {
        this.vis[tag] = !!on;
        try {
            const p = JSON.parse(localStorage.getItem(PREFS_KEY)) || {};
            p.vis = this.vis;
            localStorage.setItem(PREFS_KEY, JSON.stringify(p));
        } catch (e) { /* private mode etc */ }
        this._emit();   // instant re-render from the cached data
    }

    _tick() {
        if (this._inflight) return;
        this._inflight = true;
        const k = this.app.kotekan;
        const jget = (p) => p ? p.then(r => r.ok ? r.json() : null).catch(() => null)
                              : Promise.resolve(null);
        // Poll units: unified = one per SIGNAL (search may be null -> skipped); else one per
        // constellation chain. Each yields [detections, status].
        const units = this.unified ? this.signals : this.chains;
        const per_unit = units.map(u => Promise.all([
            u.search ? jget(k.stageGet(u.search, "get_detections")) : Promise.resolve(null),
            jget(k.stageGet(u.combiner, "get_status")),
        ]));
        Promise.all([
            fetch("/gps_sky").then(r => r.ok ? r.json() : null).catch(() => null),
            jget(k.stageGet(this.airspy_stage, "adcstat")),
            k.metrics(),
            ...per_unit,
        ]).then(([sky, adc, metrics, ...res]) => {
            this._inflight = false;
            // Hold the last good value per feed: one slow/failed poll must not
            // blank every |A| for a frame (that would look like a mass drop).
            const last = this._last;
            if (sky) last.sky = sky;
            if (adc) last.adc = adc;
            if (metrics) last.valve = this._valve(metrics);
            // Unified stores per-SIGNAL (keyed by combiner); legacy stores per-CONSTELLATION.
            const store = (last.units = last.units || {});
            units.forEach((u, i) => {
                const key = this.unified ? u.key : u.tag;
                const [det, status] = res[i];
                const l = (store[key] = store[key] || {});
                if (det) l.det = det;
                if (status) l.status = status;
            });
            this._emit();
        });
    }

    // THIS BAND'S Valve counters out of the pipeline-wide metrics dump.
    //
    // The Valve drops a frame whenever its output buffer is full -- i.e. whenever the GPU
    // chain misses real time -- and that loss is otherwise INVISIBLE to every downstream
    // observable: the frame simply never arrives, sample_seq jumps, and the tracker's ring
    // zero-fills the gap. It reads as signal that decohered, not as data that was lost.
    // (That mis-read cost 2026-07-27: the L5 peel looked like broken add-back arithmetic
    // for a day, and was really this.) The counter is the only honest witness, so the
    // viewer carries it beside the ADC's own drop counters.
    //
    // Band selection: a merged multi-band instance prefixes every stage, so our valve wears
    // the same prefix as our airspy stage ("l5_airspy_in" -> "/l5_valve"). A single-band
    // pipeline has no prefix and exactly one valve, so fall back to the only one present.
    _valve(metrics) {
        const drop = metrics["kotekan_valve_dropped_frames_total"];
        if (!drop) return null;
        const pass = metrics["kotekan_valve_passed_frames_total"] || {};
        const prefix = String(this.airspy_stage).replace(/airspy_in$/, "");
        const names = Object.keys(drop);
        let key = names.find(n => n === `/${prefix}valve` || n === `${prefix}valve`);
        if (!key && names.length === 1) key = names[0];
        if (!key) return null;
        // passed is absent on kotekan builds before 2026-07-27 -> rate only, no fraction.
        return {stage: key, dropped: drop[key],
                passed: pass[key] != null ? pass[key] : null};
    }

    // Merge into one list keyed "G12"/"E25"/"C19". Sky positions + (per-band) one metric set
    // per row, or (unified) a metric set per signal under r.sig_by, plus a row-level summary
    // (best sig/cn0 for sorting + the lock gate + sky colour).
    _merge() {
        const last = this._last;
        const sats = new Map();
        const get = (tag, prn) => {
            const id = tag + prn;
            if (!sats.has(id)) sats.set(id, {
                id, tag, prn, az: null, el: null, snr: null, detected: false,
                amp: 0, coh: 0, deep: 0, dbi: 0, sig: 0, deep_snr: 0, dr: 0,
                cn0: null, cn0_coh: null, dop: null, coh_s: null,
                peel_db: null, peel_bound: false,
                sig_by: {},     // unified: combiner-key -> signal_metrics()
                n_sig: 0,       // unified: signals this sat is locked on
            });
            return sats.get(id);
        };
        if (last.sky && Array.isArray(last.sky.sats))
            for (const p of last.sky.sats) {
                const r = get(p.const || "G", p.prn);
                r.az = p.az; r.el = p.el;
            }
        const store = last.units || {};
        if (this.unified) {
            for (const sg of this.signals) {
                const l = store[sg.key] || {};
                if (Array.isArray(l.det))
                    for (const d of l.det) get(sg.tag, d.prn).detected = true;
                if (Array.isArray(l.status))
                    for (const s of l.status) {
                        if (!s.prn) continue;
                        const r = get(sg.tag, s.prn);
                        const m = signal_metrics(s, sg.t_rec);
                        r.sig_by[sg.key] = m;
                        if ((m.sig || 0) >= SNR_LOCK) r.n_sig += 1;
                        // Row summary = the BEST signal (sky colour, sort default, lock gate).
                        if ((m.sig || 0) > (r.sig || 0)) {
                            r.sig = m.sig; r.deep_snr = m.deep_snr; r.coh_s = m.coh_s;
                            r.dop = m.dop; r.amp = m.amp;
                        }
                        if (m.cn0 != null && (r.cn0 == null || m.cn0 > r.cn0)) r.cn0 = m.cn0;
                    }
            }
        } else {
            for (const c of this.chains) {
                const l = store[c.tag] || {};
                if (Array.isArray(l.det))
                    for (const d of l.det) {
                        const r = get(c.tag, d.prn);
                        r.snr = d.snr; r.detected = true;
                    }
                if (Array.isArray(l.status))
                    for (const s of l.status) {
                        if (!s.prn) continue;
                        Object.assign(get(c.tag, s.prn), signal_metrics(s, c.t_rec));
                    }
            }
        }
        const list = [...sats.values()];
        // "Locked" = significant, not raw |A| (noise-biased). |A| fallback only
        // where nothing reports a significance at all.
        const have_sig = list.some(r => r.sig > 0);
        for (const r of list)
            r.active = have_sig ? (r.sig >= SNR_LOCK) : (r.detected || r.amp >= AMP_LOCK);
        return list;
    }

    _emit() {
        const payload = {
            sats: this._merge(),
            sky: this._last.sky, adc: this._last.adc, valve: this._last.valve,
            vis: this.vis, chains: this.chains,
            unified: this.unified, signals: this.signals,
        };
        for (const cb of this._listeners) {
            try { cb(payload); } catch (e) { console.error("gps feed listener:", e); }
        }
    }
}
