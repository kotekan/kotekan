// Shared GNSS data feed for the sky, table, history and stream-health panels. One poll loop
// merges /gps_sky (viewer server: {prn, az, el, const} positions) with, per signal, the broker
// publisher's
//   * <search>/get_detections   -> acquired PRNs + search SNR
//   * <combiner>/get_status     -> per-PRN metrics (signal_metrics)
//   * <combiner>/get_elements   -> per-element gains (beam_summary)
// into one row per satellite keyed "G12"/"E25"/"C19", and passes one receiver-wide get_rf
// document through as gnss_rf. Panels subscribe with feed.on(cb) and re-render on every merged
// tick. The per-constellation visibility toggles live here too (persisted), so the sky and the
// table agree.

const POLL_MS = 1500;
const AMP_LOCK = 0.30;      // |A| lock bar where no row reports sig
const SNR_LOCK = 3.0;       // sig lock bar (sigma above noise) for rows without q
// The lock bars on q. gps_table.js colours q with the same numbers, so the panels agree:
// >= Q_LOCK is a lock (green), <= q_floor * Q_FLOOR_MARGIN is at the measured noise (red).
// q_floor is published by the broker each cycle.
const Q_LOCK = 2.2;
const Q_FLOOR_MARGIN = 1.15;
// The fallback incoherent C/N0 in signal_metrics() has a pole at u -> a; refuse u/a above
// this. 0.95 caps it at ~40 dB-Hz for a 1 ms t_rec.
const CN0_INC_MAX_RATIO = 0.95;
const PREFS_KEY = "gps_viewer_prefs_v1";

// The constellation legend, [{tag, name, color}], delivered by the server's /wsport
// (configure_chains below).
export let CHAINS = [];
export const chain_color = tag =>
    (CHAINS.find(c => c.tag === tag) || {}).color || "#8a8f98";

let _active_feed = null;
// Adopt the constellation legend the server delivered via /wsport. Idempotent; safe to call
// before or after the feed exists.
export function configure_chains(defs) {
    if (!Array.isArray(defs) || !defs.length) return;
    CHAINS = defs;
    if (_active_feed) _active_feed._reconfigure();
}

// The signal inventory from the server's /wsport (UNIFIED_SIGNALS), one entry per broker chain:
// {tag, sys, band ("high"/"mid"/"low"), col, name, sigid, combiner, search|null, t_rec}.
// combiner and search are chain ids, used as the REST path segment. SIGNALS is null until
// delivered. The feed keys satellites by constellation+PRN across bands and hangs each signal's
// metrics off r.sig_by[combiner]; the table renders one column per signal.
export let SIGNALS = null;
// Signal key -> Set of PRNs whose satellite block broadcasts it (the server's
// _gps_signal_capability, from the IGS registry). Lets the table separate "this satellite does
// not transmit that signal" from "it does and we are not seeing it": a Block IIR satellite
// blank across L1C/L2C/L5 is correct, the same blanks on a Block III satellite are a fault.
// null -> every cell is treated as capable: never claim a satellite is incapable on
// information that could not be obtained.
export let CAPABILITY = null;
/// Constellation letter -> Set of the PRNs the registry knows there (the server's caps._prns).
/// A PRN outside its constellation's set, or a constellation with no set, is unknown -- see
/// not_transmitted(). An older server sends one GPS list; it is taken as "G".
export let CAPABILITY_PRNS = null;
export function configure_signals(sigs, caps) {
    if (!Array.isArray(sigs) || !sigs.length) { SIGNALS = null; return; }
    SIGNALS = sigs.map(s => Object.assign({key: s.combiner}, s));
    CAPABILITY = null;
    CAPABILITY_PRNS = null;
    if (caps && typeof caps === "object") {
        const by_key = {};
        for (const s of SIGNALS)
            if (s.sigid && Array.isArray(caps[s.sigid]))
                by_key[s.key] = new Set(caps[s.sigid]);
        if (Object.keys(by_key).length) {
            CAPABILITY = by_key;
            CAPABILITY_PRNS = {};
            const pr = caps._prns;
            if (Array.isArray(pr)) CAPABILITY_PRNS.G = new Set(pr);
            else if (pr && typeof pr === "object")
                for (const c of Object.keys(pr))
                    if (Array.isArray(pr[c])) CAPABILITY_PRNS[c] = new Set(pr[c]);
        }
    }
    if (_active_feed) _active_feed._reconfigure();
}
/// true when `prn` of constellation `sys` is known not to transmit the signal `key`; false
/// when capable or unknown.
///
/// A PRN absent from its constellation's CAPABILITY_PRNS set, or a constellation the registry
/// does not cover, is unknown and falls through as capable, so missing registry data renders
/// as "no detection" rather than as a claim that the satellite does not transmit. (Until
/// 10-18 the server built every signal's list from the GPS block map, so GLONASS and BeiDou
/// lists came empty and every blank cell there read "not broadcast", #206.)
export function not_transmitted(key, prn, sys) {
    if (!(CAPABILITY && CAPABILITY[key])) return false;
    const known = CAPABILITY_PRNS && CAPABILITY_PRNS[sys || "G"];
    if (!known || !known.has(prn)) return false;   // no registry here, or absent => unknown
    return !CAPABILITY[key].has(prn);
}

// One combiner get_status row -> the per-signal metrics the panels display. t_rec (the
// signal's code period, s) scales the fallback incoherent C/N0.
export function signal_metrics(s, t_rec) {
    const m = {
        amp: s.amplitude || 0,
        dop: s.doppler_hz != null ? s.doppler_hz : null,
        coh_s: s.coherence_s != null ? s.coherence_s : null,
        cn0: null,
    };
    // sig: the broker's published significance, from the first source that applies (the
    // known-rate fold when it cohered, else the incoherent t-statistic, ...; sig_src names it).
    // det_duty_s is the fraction of the broker's samples in the last 120 s in which the median
    // instance's deep SNR beat the fleet floor. Read them together: sig says how loud, duty
    // says how often.
    m.sig = s.sig != null ? s.sig : null;
    // duty is a deep-stage duty, not "was the satellite tracked": the deep fold can fail to
    // certify a satellite the despread holds continuously. prompt_hold (fleet prompt power over
    // the probes' noise median) is the number that does not flicker with the deep stage.
    m.duty = s.det_duty_s != null ? s.det_duty_s : null;
    m.hold = s.prompt_hold != null ? s.prompt_hold : null;
    // q: the fleet discriminator's quality ratio (prompt vs shoulders), the statistic the DLL
    // gates on. It says whether the code loop is on the correlation peak, which no power
    // metric can. Judge lock on q, never on sig/deep/C/N0: those ride the deep fold, which
    // re-searches rate and phase and will certify a satellite whose prompt tap is on noise.
    // q_floor is the broker's presence bar on q, measured from the noise probes each cycle
    // (fleet.apply_presence; its q_fallback when fewer than 3 probes report); q_inst counts
    // the contributing instances.
    m.q = s.fleet_q != null ? s.fleet_q : null;
    m.q_floor = s.fleet_q_floor != null ? s.fleet_q_floor : null;
    m.q_inst = s.fleet_instances != null ? s.fleet_instances : null;
    // cn0 is the broker's cn0_prompt_db (combdll.prompt_cn0): per-record prompt power, kept
    // where presence is established and the prompt is the tallest tap, debiased by the noise
    // probes; no fit. cn0_duty is the fraction of records kept: at low duty the value rests on
    // few records, so the duty travels with the value to every display.
    m.cn0_duty = s.cn0_prompt_duty != null ? s.cn0_prompt_duty : null;
    // The noise probes (seeded PRNs where nothing transmits) ride the same rows; the broker
    // flags them so no consumer draws a noise reference as a satellite.
    m.noise_probe = !!s.noise_probe;
    if (s.cn0_prompt_db != null) {
        m.cn0 = s.cn0_prompt_db;
    } else if (s.cn0_prompt_duty !== undefined) {
        // The broker ran the estimator and declined (presence not established, or too few
        // on-peak records). A gap is the honest render: falling through to the fallback
        // below would replace the refusal with a number.
        m.cn0 = null;
    } else {
        // Neither key: the row has no prompt C/N0 (a broker or chain without one, e.g. no
        // noise probes). Fallback incoherent estimate, bounded because the expression is not
        // safe near no-signal:
        //
        //   C/N0 = (S/N)/T  with  S = u^2  and  N = a^2 - u^2
        //
        // There is a pole at u -> a, where a row with no noise reference sits (the debias
        // subtracts nothing), so refuse above CN0_INC_MAX_RATIO and whenever q says the loop
        // is off the peak. The absolute scale is also off on CHORD: t_rec is the code period,
        // but a CHORD record is 2048 hops x 5.12 us = 10.486 ms, so for a 1 ms code this
        // overstates the density by 10*log10(10.486) = 10.2 dB. Correct on airspy, where a
        // record is one code period. Read it as a relative number only.
        const a = s.amplitude || 0, u = s.unbiased_amplitude || 0;
        const ratio = a > 0 ? u / a : 1.0;
        const q_says_locked = (m.q == null) || (m.q >= Q_LOCK);
        if (a > u && u > 0 && ratio <= CN0_INC_MAX_RATIO && q_says_locked)
            m.cn0 = 10 * Math.log10(((u * u) / (a * a - u * u)) / t_rec);
    }
    // cn0_kcoh: the ~1 s known-rate coherent fold (combdll.coh_cn0), centred on the previous
    // cycle's record-stream rate rather than searched. ~10log10(n_rec) more sensitive than the
    // per-record number above, and on a strong satellite the two must agree: a coherent value
    // far below its per-record partner means coherence was lost, not signal.
    //   kcoh_eta  coherence efficiency: n_rec when the fold was fully coherent, ~1 on
    //             noise. A wrong rate or a phase discontinuity shows here rather than as a
    //             silently low C/N0.
    // Not cn0_coh_db, which comes from the deep fold's per-integration rate re-search.
    m.cn0_kcoh = s.cn0_kcoh_db != null ? s.cn0_kcoh_db : null;
    m.kcoh_eta = s.kcoh_eta != null ? s.kcoh_eta : null;
    m.kcoh_n = s.kcoh_n != null ? s.kcoh_n : null;
    return m;
}

// One /get_elements row -- {probe, inst: {tag: {keff, amp[], ph[], sig[]}}} -- reduced to the
// three numbers a table cell can carry:
//   beam_n    live elements (sig >= 3 against the per-element probe floor), median over
//             instances. null when the row has no sig (no probe rows); every element then
//             counts as live.
//   beam_amp  median over instances of the median live-element amplitude, scaled by
//             (n_elem - 1) so a uniform array reads ~1.0 (the raw amp is relative to the sum
//             of the other elements).
//   beam_ph   median over instances of the circular RMS spread of the live elements'
//             phases (deg). The phase is relative to the array mean (the peel coefficient),
//             so this is how far the array is from phased up on this satellite: ~0 after a
//             good cal.
// The full per-element vectors are on the broker's /get_elements and in its element archive
// (--element-archive-dir).
export function beam_summary(row) {
    if (!row || !row.inst) return null;
    const med = a => {
        if (!a.length) return null;
        const s = [...a].sort((x, y) => x - y);
        return s[Math.floor(s.length / 2)];
    };
    const per_amp = [], per_ph = [], per_n = [];
    for (const tag of Object.keys(row.inst)) {
        const ir = row.inst[tag];
        if (!ir || !Array.isArray(ir.amp) || !ir.amp.length) continue;
        const L = ir.amp.length;
        const live = [];
        for (let e = 0; e < L; e++)
            if (!Array.isArray(ir.sig) || ir.sig[e] == null || ir.sig[e] >= 3.0) live.push(e);
        if (!live.length) { per_n.push(0); continue; }
        per_n.push(Array.isArray(ir.sig) ? live.length : null);
        per_amp.push(med(live.map(e => ir.amp[e])) * Math.max(1, L - 1));
        // circular RMS: R = |<e^{i ph}>|, rms = sqrt(-2 ln R)
        let cr = 0, ci = 0;
        for (const e of live) { cr += Math.cos(ir.ph[e]); ci += Math.sin(ir.ph[e]); }
        const R = Math.hypot(cr, ci) / live.length;
        per_ph.push(R > 0 ? Math.sqrt(Math.max(0, -2 * Math.log(Math.min(1, R)))) * 180 / Math.PI
                          : 180);
    }
    if (!per_amp.length) return null;
    const ns = per_n.filter(x => x != null);
    return {beam_amp: med(per_amp), beam_ph: med(per_ph),
            beam_n: ns.length ? med(ns) : null, probe: !!row.probe};
}

export class GpsFeed {
    constructor({app}) {
        this.app = app;
        this._listeners = [];
        this._inflight = false;
        this._last = {};
        this.vis = {};
        this._prefs = null;
        try { this._prefs = JSON.parse(localStorage.getItem(PREFS_KEY)); } catch (e) { /* fresh */ }
        this._rebuild();                      // sets chains/signals/vis from CHAINS + SIGNALS
        _active_feed = this;                  // configure_chains/_signals re-target this instance
        this._tick();
        this._timer = setInterval(() => this._tick(), POLL_MS);
    }

    // (Re)build the descriptor lists and the per-constellation visibility. chains is the
    // per-constellation list (sky legend, colours, visibility chips); signals is the flat
    // per-signal list.
    _rebuild() {
        this.chains = this._build_chains();
        this.signals = this._build_signals();
    }

    // Signal descriptors: the server inventory verbatim, with visibility inherited from the
    // constellation (one toggle hides the satellite's row, all its signals). t_rec/band/col
    // ride along for the table and the C/N0.
    _build_signals() {
        if (!SIGNALS) return [];
        for (const s of SIGNALS) {
            // Keyed by the satellite's constellation (sys), the key the toggles and the rows
            // use, so one toggle hides every signal of a satellite.
            const k = s.sys || s.tag;
            if (!(k in this.vis))
                this.vis[k] = (this._prefs && this._prefs.vis && k in this._prefs.vis)
                    ? !!this._prefs.vis[k] : true;
        }
        return SIGNALS.map(s => Object.assign({}, s));
    }

    // The per-constellation descriptors from the current module CHAINS. Visibility defaults on
    // per tag, overridden by any persisted preference. Rerun by _reconfigure.
    _build_chains() {
        const chains = CHAINS.map(c => Object.assign({}, c));
        for (const c of chains)
            if (!(c.tag in this.vis))
                this.vis[c.tag] = (this._prefs && this._prefs.vis && c.tag in this._prefs.vis)
                    ? !!this._prefs.vis[c.tag] : true;
        return chains;
    }

    // The server's tables (configure_chains / configure_signals) arrived after construction:
    // rebuild and re-render so the legend, colours and columns match them.
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
        // One poll unit per signal (a null search is skipped), each yielding
        // [detections, status, elements].
        const units = this.signals;
        const per_unit = units.map(u => Promise.all([
            u.search ? jget(k.stageGet(u.search, "get_detections")) : Promise.resolve(null),
            jget(k.stageGet(u.combiner, "get_status")),
            // Per-element gain table, served by the broker publisher on the same chain path
            // (broker --element-poll). Until it has rows the beam cells have no value.
            jget(k.stageGet(u.combiner, "get_elements")),
        ]));
        // RF-path health (get_rf): one call per tick, not per unit. The voltage tap is per GPU
        // and serves every signal on it, so the broker publishes it receiver-wide and any
        // chain's path returns the same document. Its instances map stays empty until the
        // broker polls the taps (--rf-stats-endpoints).
        const rfP = units.length
            ? jget(k.stageGet(units[0].combiner, "get_rf")) : Promise.resolve(null);
        Promise.all([
            fetch("/gps_sky").then(r => r.ok ? r.json() : null).catch(() => null),
            rfP,
            ...per_unit,
        ]).then(([sky, gnss_rf, ...res]) => {
            this._inflight = false;
            // Hold the last good value per feed: one slow/failed poll must not
            // blank every |A| for a frame (that would look like a mass drop).
            const last = this._last;
            if (sky) last.sky = sky;
            // Hold the last good RF document: an empty instances map (not polled yet) must not
            // evict a good one, the same rule as `elem` below.
            if (gnss_rf && gnss_rf.instances
                && Object.keys(gnss_rf.instances).length) last.gnss_rf = gnss_rf;
            // Stored per signal (keyed by combiner).
            const store = (last.units = last.units || {});
            units.forEach((u, i) => {
                const [det, status, elem] = res[i];
                const l = (store[u.key] = store[u.key] || {});
                if (det) l.det = det;
                if (status) l.status = status;
                // Only a table with rows counts as data; {} (not polled yet) must not evict
                // the last good one.
                if (elem && elem.prns && Object.keys(elem.prns).length) l.elem = elem;
            });
            this._emit();
        });
    }

    // Merge into one list keyed "G12"/"E25"/"C19". Sky positions + a metric set per signal
    // under r.sig_by, plus a row-level summary (best sig/cn0 for sorting + the lock gate + sky
    // colour).
    _merge() {
        const last = this._last;
        const sats = new Map();
        const get = (tag, prn) => {
            const id = tag + prn;
            if (!sats.has(id)) sats.set(id, {
                id, tag, prn, az: null, el: null, snr: null, detected: false,
                amp: 0, sig: 0, cn0: null,
                q: null, q_floor: null, noise_probe: false,
                n_status: 0, n_probe: 0,   // status rows seen / of them flagged noise_probe
                sig_by: {},     // combiner-key -> signal_metrics()
            });
            return sats.get(id);
        };
        if (last.sky && Array.isArray(last.sky.sats))
            for (const p of last.sky.sats) {
                const r = get(p.const || "G", p.prn);
                r.az = p.az; r.el = p.el;
            }
        const store = last.units || {};
        for (const sg of this.signals) {
            const l = store[sg.key] || {};
            if (Array.isArray(l.det))
                for (const d of l.det) {
                    // The row is keyed by the satellite's constellation (sys), the column by
                    // signal, so every signal a satellite carries lands in one row.
                    const r = get(sg.sys || sg.tag, d.prn);
                    r.detected = true;
                    if (d.snr != null) {
                        // Keep the search SNR per signal as well as the row maximum: it is
                        // the one metric that reports on acquisition rather than the deep
                        // integration, so "strong in search, wrong downstream" must stay
                        // visible per signal. The row-level r.snr (sky colour, sort) is the
                        // maximum.
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
                    r.n_status += 1;
                    // A noise probe is that chain's floor reference, not a measurement of the
                    // satellite, so it contributes nothing to the row summary. With
                    // --probe-dark (glo_l2oc, 10-18) a probe is an overhead satellite that
                    // cannot broadcast this one signal while another chain tracks it, so the
                    // flag stays with the signal (the cell says "probe") and the row is judged
                    // on the others. Until 10-18 one probe flag greyed the whole row (#206).
                    if (m.noise_probe) { r.n_probe += 1; continue; }
                    // Row summary = the best non-probe signal (sky colour, sort, the sig lock
                    // rule).
                    if ((m.sig || 0) > (r.sig || 0)) {
                        r.sig = m.sig; r.amp = m.amp;
                    }
                    if (m.cn0 != null && (r.cn0 == null || m.cn0 > r.cn0)) r.cn0 = m.cn0;
                    // The lock rule's inputs: the signal with the highest q, judged against its
                    // own floor.
                    if (m.q != null && (r.q == null || m.q > r.q)) {
                        r.q = m.q; r.q_floor = m.q_floor;
                    }
                }
            // Beam summary per (satellite, signal), attached to the same sig_by entry the
            // status built, so a beam cell and its C/N0 cell describe the same signal. A
            // table whose utc is 300 s or older renders as absent, not as fresh.
            const et = l.elem;
            if (et && et.prns && (et.utc == null || Date.now() / 1000 - et.utc < 300)) {
                for (const pk of Object.keys(et.prns)) {
                    const b = beam_summary(et.prns[pk]);
                    if (!b) continue;
                    const r = get(sg.sys || sg.tag, +pk);
                    const e = (r.sig_by[sg.key] = r.sig_by[sg.key] || {});
                    e.beam_amp = b.beam_amp;
                    e.beam_ph = b.beam_ph;
                    e.beam_n = b.beam_n;
                }
            }
        }
        const list = [...sats.values()];
        // A probe row is a probe on every signal that reported it: the below-horizon probes
        // of a chain without --probe-dark, which no chain tracks. A dark probe on one signal
        // beside a track on another keeps its row.
        for (const r of list) r.noise_probe = r.n_status > 0 && r.n_probe === r.n_status;
        // "Locked" (r.active) means the code loop is on the peak, judged on q, not that
        // something was significant: sig rides the deep and known-rate folds, which re-search
        // rate and phase and certify whatever they land on, noise included, so a dead band can
        // serve sig in the thousands while q sits at its noise floor.
        //
        // A row carrying q is active when q >= Q_LOCK and q is above q_floor * Q_FLOOR_MARGIN.
        // Marginal rows (above the floor, under Q_LOCK; amber in the table) render idle, the
        // conservative direction. A probe row is never active. A row without q falls back to
        // sig >= SNR_LOCK, or to detection / |A| where no row reports sig.
        const have_sig = list.some(r => r.sig > 0);
        for (const r of list) {
            if (r.noise_probe)
                r.active = false;
            else if (r.q != null)
                r.active = (r.q >= Q_LOCK)
                           && !(r.q_floor != null && r.q <= r.q_floor * Q_FLOOR_MARGIN);
            else
                r.active = have_sig ? (r.sig >= SNR_LOCK)
                                    : (r.detected || r.amp >= AMP_LOCK);
        }
        return list;
    }

    _emit() {
        const payload = {
            sats: this._merge(),
            sky: this._last.sky,
            vis: this.vis, chains: this.chains,
            signals: this.signals,
            gnss_rf: this._last.gnss_rf || null,      // receiver-wide RF health (get_rf)
        };
        for (const cb of this._listeners) {
            try { cb(payload); } catch (e) { console.error("gps feed listener:", e); }
        }
    }
}
