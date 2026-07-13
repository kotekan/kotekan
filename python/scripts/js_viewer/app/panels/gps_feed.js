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

// Constellation identity: stage names, record period (for C/N0 = x / t_rec) and
// the display colour (matches the matplotlib composite-map palette G/E/C =
// blue/orange/red so plots and viewer read the same). All three share the
// 1575.42 MHz tune; a 404 on a missing gal_/bds_ stage just skips that chain.
export const CHAINS = [
    {tag: "G", name: "GPS L1 C/A",  t_rec: 1e-3,  color: "#4d9de0"},
    {tag: "E", name: "Galileo E1C", t_rec: 4e-3,  color: "#e8923c"},
    {tag: "C", name: "BeiDou B1C",  t_rec: 10e-3, color: "#d64550"},
];
export const chain_color = tag =>
    (CHAINS.find(c => c.tag === tag) || {}).color || "#8a8f98";

export class GpsFeed {
    constructor({app, search_stage, combiner_stage, airspy_stage}) {
        this.app = app;
        this.airspy_stage = airspy_stage || "airspy_in";
        // gps_* names throughout (symmetric with gal_*/bds_*); KotekanRest.resolveStage
        // maps them onto the bare search/combiner spelling on older configs.
        this.chains = CHAINS.map(c => Object.assign({}, c, c.tag === "G"
            ? {search: search_stage || "gps_search",
               combiner: combiner_stage || "gps_combiner"}
            : {search: c.tag === "E" ? "gal_search" : "bds_search",
               combiner: c.tag === "E" ? "gal_combiner" : "bds_combiner"}));
        this._listeners = [];
        this._inflight = false;
        this._last = {};
        this.vis = {G: true, E: true, C: true};
        try {
            const p = JSON.parse(localStorage.getItem(PREFS_KEY));
            if (p && p.vis) Object.assign(this.vis, p.vis);
        } catch (e) { /* fresh browser */ }
        this._tick();
        this._timer = setInterval(() => this._tick(), POLL_MS);
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
        const jget = (p) => p.then(r => r.ok ? r.json() : null).catch(() => null);
        const per_chain = this.chains.map(c => Promise.all([
            jget(k.stageGet(c.search, "get_detections")),
            jget(k.stageGet(c.combiner, "get_status")),
        ]));
        Promise.all([
            fetch("/gps_sky").then(r => r.ok ? r.json() : null).catch(() => null),
            jget(k.stageGet(this.airspy_stage, "adcstat")),
            ...per_chain,
        ]).then(([sky, adc, ...chains]) => {
            this._inflight = false;
            // Hold the last good value per feed: one slow/failed poll must not
            // blank every |A| for a frame (that would look like a mass drop).
            const last = this._last;
            if (sky) last.sky = sky;
            if (adc) last.adc = adc;
            last.chains = last.chains || {};
            this.chains.forEach((c, i) => {
                const [det, status] = chains[i];
                const l = (last.chains[c.tag] = last.chains[c.tag] || {});
                if (det) l.det = det;
                if (status) l.status = status;
            });
            this._emit();
        });
    }

    // Merge every constellation into one list keyed "G12"/"E25"/"C19".
    _merge() {
        const last = this._last;
        const sats = new Map();
        const get = (tag, prn) => {
            const id = tag + prn;
            if (!sats.has(id)) sats.set(id, {
                id, tag, prn, az: null, el: null, snr: null, detected: false,
                amp: 0, coh: 0, deep: 0, dbi: 0, sig: 0, deep_snr: 0, dr: 0,
                cn0: null, cn0_coh: null, dop: null, coh_s: null,
            });
            return sats.get(id);
        };
        if (last.sky && Array.isArray(last.sky.sats))
            for (const p of last.sky.sats) {
                const r = get(p.const || "G", p.prn);
                r.az = p.az; r.el = p.el;
            }
        for (const c of this.chains) {
            const l = (last.chains && last.chains[c.tag]) || {};
            if (Array.isArray(l.det))
                for (const d of l.det) {
                    const r = get(c.tag, d.prn);
                    r.snr = d.snr; r.detected = true;
                }
            if (Array.isArray(l.status))
                for (const s of l.status) {
                    if (!s.prn) continue;
                    const r = get(c.tag, s.prn);
                    r.amp = s.amplitude || 0;
                    r.coh = s.coh_amplitude || 0;
                    r.deep = s.deep_amplitude || 0;
                    // unbiased |A| (Â): noise-debiased signal amplitude -- ~0 for
                    // noise. Deep (nav-wiped) when available, else moment-debiased.
                    r.dbi = s.deep_amplitude || s.unbiased_amplitude || 0;
                    // significance: sigma above noise. deep counts only when the combiner
                    // certified it beat its rectification floor (coherence_s > 0) --
                    // a floored deep (~7 sigma) is noise wearing a lock's clothing.
                    r.sig = (s.coherence_s || 0) > 0
                        ? Math.max(s.deep_snr || 0, s.amp_snr || 0)
                        : (s.amp_snr || 0);
                    r.dop = s.doppler_hz != null ? s.doppler_hz : null;
                    r.coh_s = s.coherence_s != null ? s.coherence_s : null;
                    r.deep_snr = s.deep_snr || 0;
                    r.dr = s.deep_records || 0;   // ladder window (records) behind this emit
                    // COHERENT C/N0 (dB-Hz) = 10 log10(deep_snr^2 / T_coh): the deep
                    // estimator, sqrt(T)-deep but defined only where the combiner
                    // certified the coherence (a floored deep is rectification noise).
                    if (r.coh_s > 0 && r.deep_snr > 0)
                        r.cn0_coh = 20 * Math.log10(r.deep_snr)
                                    - 10 * Math.log10(r.coh_s);
                    // Incoherent C/N0 (dB-Hz, pipeline zero-point): per-record
                    // power ratio x = u^2/(a^2-u^2), density x/t_rec -- the same
                    // estimator as gps_cn0_map.py, needing only 1 record of
                    // coherence. Guarded: needs a real debiased amplitude.
                    const a = s.amplitude || 0, u = s.unbiased_amplitude || 0;
                    if (a > u && u > 0) {
                        const x = (u * u) / (a * a - u * u);
                        r.cn0 = 10 * Math.log10(x / c.t_rec);
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
            sky: this._last.sky, adc: this._last.adc,
            vis: this.vis, chains: this.chains,
        };
        for (const cb of this._listeners) {
            try { cb(payload); } catch (e) { console.error("gps feed listener:", e); }
        }
    }
}
