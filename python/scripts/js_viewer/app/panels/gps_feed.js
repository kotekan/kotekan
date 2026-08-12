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
// CAPABILITY: signal key -> Set of PRNs whose satellite BLOCK actually broadcasts it (server
// side, from the cached Celestrak block names). Lets the table separate "this satellite does
// not transmit that signal" from "it does and we are not seeing it" -- a Block IIR sat blank
// across L1C/L2C/L5 is correct, the same blanks on a Block III sat are a fault. Empty/absent
// -> every cell is treated as capable, i.e. exactly the old behaviour: we never claim a
// satellite is incapable on the strength of information we could not obtain.
export let CAPABILITY = null;
/// Every PRN the block map has an opinion about. A PRN outside this set is UNKNOWN -- see
/// not_transmitted(). The server sends it as caps._prns; without it we fall back to the
/// union of the per-signal sets, which is the same thing whenever any signal is unfiltered.
export let CAPABILITY_PRNS = null;
export function configure_signals(sigs, rf, caps) {
    if (!Array.isArray(sigs) || !sigs.length) { SIGNALS = null; return; }
    SIGNALS = sigs.map(s => Object.assign({key: s.combiner}, s));
    RF_BANDS = Array.isArray(rf) && rf.length ? rf : null;
    CAPABILITY = null;
    CAPABILITY_PRNS = null;
    if (caps && typeof caps === "object") {
        const by_key = {};
        for (const s of SIGNALS)
            if (s.sigid && Array.isArray(caps[s.sigid]))
                by_key[s.key] = new Set(caps[s.sigid]);
        if (Object.keys(by_key).length) {
            CAPABILITY = by_key;
            CAPABILITY_PRNS = Array.isArray(caps._prns) && caps._prns.length
                ? new Set(caps._prns)
                : new Set([].concat(...Object.values(by_key).map(s => [...s])));
        }
    }
    if (_active_feed) _active_feed._reconfigure();
}
/// true when `prn` is known NOT to transmit the signal `key`. False when capable OR unknown.
///
/// ⚠️ ABSENT IS UNKNOWN, NOT INCAPABLE. A PRN missing from the block map has to fall through
/// as capable, or missing DATA becomes a positive CLAIM -- which is the one thing the
/// CAPABILITY docstring above says this must never do. It did exactly that: PRN 2 is absent
/// from the cached TLE (so are 1 and 8), and the table marked it "not broadcast by this
/// satellite" while the blind search was detecting it at 28 sigma on its own ephemeris
/// Doppler (+1122 measured vs +1122 predicted, 2026-08-10). CAPABILITY_PRNS is the set of
/// PRNs the map actually knows about; anything outside it is unknown and renders as
/// "no detection" rather than "not transmitted".
export function not_transmitted(key, prn) {
    if (!(CAPABILITY && CAPABILITY[key])) return false;
    if (CAPABILITY_PRNS && !CAPABILITY_PRNS.has(prn)) return false;   // absent => unknown
    return !CAPABILITY[key].has(prn);
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
        // PROMPT LOCK (task #47): is the prompt tap on the signal at all? Every C/N0 below is
        // built from deep_snr, and the deep fold RE-SEARCHES rate and phase -- it re-finds the
        // satellite wherever the tap was commanded, so this panel can show 41 dB-Hz and a
        // dll_disc of 0.01 while E/P/L sit on pure noise. That is not a hypothetical: on
        // 2026-08-12 15:20-15:45 UTC all five chains did exactly that for 25 minutes and it
        // was read off this display as the array's best look of the day. Default TRUE so a
        // pre-#47 broker renders unchanged rather than greying every satellite.
        prompt_lock: s.prompt_lock != null ? !!s.prompt_lock : true,
    };
    // significance: deep counts only where the combiner certified it beat its rectification
    // floor (coherence_s > 0); a floored deep (~7-12 sigma) is noise wearing a lock's clothes.
    m.sig = (s.coherence_s || 0) > 0
        ? Math.max(s.deep_snr || 0, s.amp_snr || 0) : (s.amp_snr || 0);
    // coherent C/N0 (dB-Hz): PREFER the broker-published cn0_coh_db (task #35) -- the
    // best single instance over its own span, one estimator, one normalisation. The
    // local derivation from deep_snr is WRONG whenever the broker serves the fleet
    // override or the quadrature fallback (quad inflates by 10log10(N) ~ 10.8 dB, and
    // every fleet<->quad flip stepped this display ~8 dB -- the measured bulk of the
    // "C/N0 scatter", docs 11.31). Kept only as a fallback for pre-#35 brokers.
    if (s.cn0_coh_db != null)
        m.cn0_coh = s.cn0_coh_db;
    else if (m.coh_s > 0 && m.deep_snr > 0)
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
        for (const s of SIGNALS) {
            // Register visibility under the CONSTELLATION (sys), not the display tag, so the
            // one GPS toggle hides C/A and L1C together -- they are the same satellite. Keying
            // this by tag created a phantom "L" entry that no chip ever rendered and nothing
            // could turn off.
            const k = s.sys || s.tag;
            if (!(k in this.vis))
                this.vis[k] = (this._prefs && this._prefs.vis && k in this._prefs.vis)
                    ? !!this._prefs.vis[k] : true;
        }
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
        // Stream health, unified: watch all three front ends but poll ONE airspy's /adcstat
        // per tick, ROUND-ROBIN -- NOT all three concurrently. /adcstat runs on restServer's
        // single libevent thread with a bounded cv.wait (THE WEDGE, 5988f657); three
        // concurrent adcstat waits every tick tripled this viewer's REST-thread load over any
        // per-band viewer and is a plausible aggravator of the silent-kill (2026-07-28). The
        // valve counters -- the important silent-loss signal -- ride the single /metrics call
        // (all bands, no per-airspy blocking) and stay fresh every tick; only ADC rms/rail/USB
        // refreshes round-robin (~4.5 s per band, plenty for a health strip).
        let adcStage, adcBand = null;
        if (this.unified && RF_BANDS) {
            this._rr = ((this._rr || 0) + 1) % RF_BANDS.length;
            adcStage = RF_BANDS[this._rr].airspy;
            adcBand = RF_BANDS[this._rr].band;
        } else {
            adcStage = this.airspy_stage;
        }
        Promise.all([
            fetch("/gps_sky").then(r => r.ok ? r.json() : null).catch(() => null),
            // No airspy stage -> no ADC health to poll. CHORD's front end is the F-engine,
            // which exposes none of this; skip rather than fabricate a request.
            adcStage ? jget(k.stageGet(adcStage, "adcstat")) : Promise.resolve(null),
            k.metrics(),
            ...per_unit,
        ]).then(([sky, adc, metrics, ...res]) => {
            this._inflight = false;
            // Hold the last good value per feed: one slow/failed poll must not
            // blank every |A| for a frame (that would look like a mass drop).
            const last = this._last;
            if (sky) last.sky = sky;
            if (adc) last.adc = adc;                   // primary (single-band consumers)
            if (this.unified && RF_BANDS) {
                // Keep the freshest ADC per band; rebuild rf_health from those + fresh valves.
                this._adc_by_band = this._adc_by_band || {};
                if (adc && adcBand) this._adc_by_band[adcBand] = adc;
                if (metrics)
                    last.rf_health = RF_BANDS.map(b => ({
                        band: b.band, label: b.label,
                        adc: this._adc_by_band[b.band] || null,
                        valve: this._valve_for(metrics, b.airspy)}));
            }
            if (metrics) last.valve = this._valve_for(metrics, this.airspy_stage);
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
    _valve_for(metrics, airspy_stage) {
        const drop = metrics["kotekan_valve_dropped_frames_total"];
        if (!drop) return null;
        const pass = metrics["kotekan_valve_passed_frames_total"] || {};
        const prefix = String(airspy_stage).replace(/airspy_in$/, "");
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
                // Paired with cn0_coh below, NOT summarised independently -- see there.
                prompt_lock: true,
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
                    for (const d of l.det) {
                        // sys, NOT tag: the satellite ROW is keyed by constellation, the signal
                        // COLUMN by tag. GPS L1C carries tag "L" so its series stays distinct
                        // from C/A on the same PRN, but it is a GPS bird -- keying the row by
                        // tag split one Block III satellite into a G20 row and an L20 row,
                        // against the whole "one satellite per row" premise of this viewer.
                        const r = get(sg.sys || sg.tag, d.prn);
                        r.detected = true;
                        if (d.snr != null) {
                            // Keep the search SNR PER SIGNAL, not just as a row maximum: it is
                            // the one metric that reports on the ACQUISITION rather than the
                            // deep integration, so "strong in search, wrong downstream" is
                            // exactly the state it exists to show -- and a row max hides which
                            // signal is which. The row-level r.snr (sky colour, sort) is
                            // unchanged.
                            const e = (r.sig_by[sg.key] = r.sig_by[sg.key] || {});
                            e.snr = Math.max(e.snr || 0, d.snr);
                            if (r.snr == null || d.snr > r.snr) r.snr = d.snr;
                        }
                    }
                if (Array.isArray(l.status))
                    for (const s of l.status) {
                        if (!s.prn) continue;
                        const r = get(sg.sys || sg.tag, s.prn);   // see the det loop above
                        const m = signal_metrics(s, sg.t_rec);
                        // The det loop above may already have stashed this signal's search SNR;
                        // signal_metrics() knows nothing about detections, so carry it across
                        // rather than letting the status overwrite it.
                        const prev = r.sig_by[sg.key];
                        if (prev && prev.snr != null) m.snr = prev.snr;
                        r.sig_by[sg.key] = m;
                        if ((m.sig || 0) >= SNR_LOCK) r.n_sig += 1;
                        // Row summary = the BEST signal (sky colour, sort default, lock gate).
                        if ((m.sig || 0) > (r.sig || 0)) {
                            r.sig = m.sig; r.deep_snr = m.deep_snr; r.coh_s = m.coh_s;
                            r.dop = m.dop; r.amp = m.amp;
                        }
                        if (m.cn0 != null && (r.cn0 == null || m.cn0 > r.cn0)) r.cn0 = m.cn0;
                        // ...and the COHERENT one alongside, or the table's C/N0-coh column
                        // and its sort would read empty in flat mode while the unified cells
                        // showed values. Both are row summaries = best signal, same rule.
                        // ⚠️ THE FLAG MOVES WITH THE NUMBER IT QUALIFIES (task #47). Taking the
                        // row's prompt_lock as, say, "any signal locked" while the row's
                        // cn0_coh comes from the max would let a blind signal's C/N0 be shown
                        // under a different signal's lock -- the same mismatched-pairing shape
                        // that produced the phantom C/N0 scatter in #35. One assignment, one
                        // measurement.
                        if (m.cn0_coh != null && (r.cn0_coh == null || m.cn0_coh > r.cn0_coh)) {
                            r.cn0_coh = m.cn0_coh;
                            r.prompt_lock = m.prompt_lock;
                        }
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
            rf_health: this._last.rf_health || null,
        };
        for (const cb of this._listeners) {
            try { cb(payload); } catch (e) { console.error("gps feed listener:", e); }
        }
    }
}
