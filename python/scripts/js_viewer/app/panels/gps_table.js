// GNSS detections table, its own resizable card (the skyplot is gps_sky.js; both consume the
// shared GpsFeed). One row per satellite that is either locked or above the horizon; unlocked
// rows are dimmed rather than hidden, so what is overhead waiting to be acquired stays visible.
// One column per signal, grouped by band; the cell shows the selected metric (METRICS below).
//
// Click a header to sort on it, again to flip direction. The sort is stable (satellite id
// tiebreak) and persisted per browser, so the table does not reshuffle every poll.

import {chain_color, not_transmitted} from "./gps_feed.js";
import {snr_color} from "./gps_sky.js";

const PREFS_KEY = "gps_viewer_prefs_v1";

// The cell metric, switchable. Each maps to a field of the per-signal metrics object
// (gps_feed.signal_metrics) and a formatter. Integer C/N0 and sig keep the many columns tight.
const METRICS = {
    // Search detection significance, the only metric here that reports on acquisition rather
    // than on the deep integration: a strong SNR beside a dead sig/coh says the signal is there
    // and something downstream of the search is wrong. Blank for a chain with no search stage.
    snr:  {label: "SNR",  field: "snr",     unit: "σ",
           fmt: m => m && m.snr ? m.snr.toFixed(0) : null},
    // Two C/N0s that measure different things. `cn0` is cn0_prompt_db, per-record prompt power
    // (presence- and on-peak-gated, probe-debiased): no coherence assumed. It is conditioned on
    // lock, so below 70% duty it is greyed with the duty in the hover (read it as "when held").
    // `cn0k` is the ~1 s known-rate coherent fold, ~10log10(n_rec) more sensitive, meaningful
    // wherever the carrier stayed coherent. On a strong satellite they must agree; a low
    // coherent value beside a healthy prompt one means coherence was lost, not signal, and
    // eta says so.
    cn0:  {label: "C/N0 prompt", field: "cn0", unit: "dB-Hz",
           fmt: m => {
               if (!m || m.cn0 == null) return null;
               const v = m.cn0.toFixed(0);
               if (m.cn0_duty == null) return v + "*";   // no duty: the fallback estimator
               return m.cn0_duty < 0.7
                   ? `<span style="opacity:.55" title="lock duty ${(100 * m.cn0_duty).toFixed(0)}%`
                     + ` -- the C/N0 of the records where tracking held">${v}</span>`
                   : v;
           }},
    cn0k: {label: "C/N0 coh", field: "cn0_kcoh", unit: "dB-Hz",
           fmt: m => {
               if (!m || m.cn0_kcoh == null) return null;
               const v = m.cn0_kcoh.toFixed(0);
               // Greyed when the fold did not cohere (eta < n_rec/4): the value is then a
               // lower bound, and the hover says why.
               const eta = m.kcoh_eta, n = m.kcoh_n;
               if (eta != null && n && eta < 0.25 * n)
                   return `<span style="opacity:.55" title="coherence efficiency ${eta.toFixed(0)}`
                        + `/${n} records -- the fold barely cohered (stale or wrong residual`
                        + ` rate, or a phase discontinuity), so this is a LOWER BOUND">${v}</span>`;
               return v;
           }},
    // The per-element complex gain, summarised (gps_feed.beam_summary).
    //   beam A   = median live-element gain amplitude, 1.0 = uniform array.
    //   φ spread = the circular RMS of the live elements' phases about the array mean, in
    //              degrees: a dispersion, not an angle. 0 means the array is phased up on
    //              this satellite, 60+ means the elements disagree. The per-element phases
    //              (the peel coefficients) are on /get_elements.
    beam_amp: {label: "beam A", field: "beam_amp", unit: "x",
               fmt: m => m && m.beam_amp != null ? m.beam_amp.toFixed(2) : null},
    beam_ph:  {label: "φ spread", field: "beam_ph", unit: "° rms",
               fmt: m => m && m.beam_ph != null ? m.beam_ph.toFixed(0) + "°" : null},
    // duty: the fraction of the last 2 minutes in which the median instance's deep SNR beat
    // the fleet floor (det_duty_s). sig says how loud, duty how often. It is a deep-stage duty;
    // hold (prompt power over the probes' noise median) says whether the despread is on the
    // satellite.
    duty: {label: "duty", field: "duty",    unit: "",
           fmt: m => m && m.duty != null ? (100 * m.duty).toFixed(0) + "%" : null},
    hold: {label: "hold", field: "hold",    unit: "x",
           fmt: m => m && m.hold ? m.hold.toFixed(1) : null},
    // q: the code-loop quality, the fleet discriminator's prompt-vs-shoulder ratio. It says
    // whether the code loop is on the peak, which every power metric in this table is blind to
    // (the deep fold re-searches, so sig and C/N0 can look healthy while the prompt tap sits on
    // noise). Coloured against the broker's measured floor: >= 2.2 solid lock, <= 1.15 x floor
    // off-peak, marginal between (gps_feed's Q_LOCK and Q_FLOOR_MARGIN). The hover carries the
    // floor and the contributing instance count.
    q:    {label: "q", field: "q", unit: "",
           fmt: m => {
               if (!m || m.q == null) return null;
               const v = m.q.toFixed(2);
               const fl = m.q_floor, n = m.q_inst;
               const tip = `code-loop quality; floor ${fl != null ? fl.toFixed(2) : "?"}`
                         + `${n != null ? `, ${n} instances` : ""}`
                         + ` -- >=2.2 solid, near the floor means the prompt tap is OFF-PEAK`;
               const col = m.q >= 2.2 ? "#7fd67f"
                         : (fl != null && m.q <= fl * 1.15) ? "#e06666" : "#e0c060";
               return `<span style="color:${col}" title="${tip}">${v}</span>`;
           }},
    sig:  {label: "sig",  field: "sig",     unit: "σ",
           fmt: m => m && m.sig ? m.sig.toFixed(0) : null},
    coh:  {label: "coh",  field: "coh_s",   unit: "s",
           fmt: m => m && m.coh_s > 0 ? m.coh_s.toFixed(2) : null},
};

// Column groups shared by every constellation (livebeam_server._band_of): high >= 1.40 GHz
// (L1/E1/B1), mid 1.22-1.40 GHz (L2C, GLONASS L2, B3I, E6), low below that (the E5/B2 complex,
// GLONASS L3OC). The signals in each group differ per constellation, so each constellation is
// its own section below the shared header. Column-group key -> display name, and the
// left-to-right order; the keys are mirrored in app/panels/decode_health.js and
// livebeam_server.py's BAND_ORDER, so rename all three together.
const BAND_LABEL = {high: "High", mid: "Mid", low: "Low"};
const BAND_ORDER = ["high", "mid", "low"];
// Nominal boresight, for the Offset column: the dishes point at dec +40.73 through the local
// meridian (config/pointings.yaml), i.e. az 180, el 81.41 at the site -- the same constants as
// the broker's sky.BORESIGHT_AZ_DEG/EL_DEG.
// TODO: serve the pointing from the feed (derived from config/pointings.yaml) so a repoint
// changes one file, not three.
const BORESIGHT_AZ_DEG = 180.0;
const BORESIGHT_EL_DEG = 81.41;
const _unit = (el, az) => {
    const e = el * Math.PI / 180, a = az * Math.PI / 180;
    return [Math.cos(e) * Math.sin(a), Math.cos(e) * Math.cos(a), Math.sin(e)];
};
const _BS = _unit(BORESIGHT_EL_DEG, BORESIGHT_AZ_DEG);
// Angular distance from boresight, degrees (null without a position).
export const boresight_offset_deg = (el, az) => {
    if (el == null || az == null) return null;
    const u = _unit(el, az);
    const c = u[0] * _BS[0] + u[1] * _BS[1] + u[2] * _BS[2];
    return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
};

/// Constellations to section the table by, from the server inventory (`chains`), which is
/// itself derived from the broker's running chains.
const consts_of = (d) => (d.chains || []).map(c => ({tag: c.tag, name: c.name}));

export class GpsTablePanel {
    constructor({target, feed, has_site}) {
        this.feed = feed;
        this.has_site = !!has_site;
        this.usort = {key: "el", dir: -1};    // default: highest sat first
        // A stored preference wins only while it names a METRICS key.
        this.metric = "cn0";                  // cell metric
        try {
            const p = JSON.parse(localStorage.getItem(PREFS_KEY));
            if (p && p.usort) this.usort = p.usort;
            if (p && p.metric && METRICS[p.metric]) this.metric = p.metric;
        } catch (e) { /* fresh browser */ }

        const root = $("#" + target);
        const side = $("<div/>").css({
            width: "100%", height: "100%", overflow: "auto",
            fontSize: "12px", fontFamily: "sans-serif",
        }).appendTo(root);
        this._status = $("<div/>").css({
            color: "#444", marginBottom: "6px", lineHeight: "1.6",
        }).appendTo(side);
        this._table = $("<div/>").appendTo(side);
        this._last = null;

        this.feed.on(d => { this._last = d; this._render(d); });
    }

    _save_pref(key, val) {
        try {
            const p = JSON.parse(localStorage.getItem(PREFS_KEY)) || {};
            p[key] = val;
            localStorage.setItem(PREFS_KEY, JSON.stringify(p));
        } catch (e) { /* private mode etc */ }
    }

    // Which rows to show + the status line's sky/site summary + the constellation chips.
    _prep(d) {
        const {sats, sky, vis, chains} = d;
        const sky_ok = !!(sky && sky.ok);
        const below = r => sky_ok && r.active && (r.el == null || r.el < 0);
        const shown = sats.filter(r => vis[r.tag]
                                       && (r.active || (r.el != null && r.el >= 0))
                                       && !below(r));
        const locked = shown.filter(r => r.active);
        const visible = shown.filter(r => r.el != null && r.el >= 0).length;
        let sky_state;
        if (!this.has_site) sky_state = "<b>no site</b> (set LAT/LON)";
        else if (sky && sky.ok) sky_state = "ok";
        else if (sky && sky.computing) sky_state = "loading…";
        else sky_state = sky && sky.error ? "unavailable" : "—";
        const site = (sky && sky.lat != null)
            ? `${sky.lat.toFixed(3)}, ${sky.lon.toFixed(3)}` : "—";
        const chips = chains.map(c => {
            const on = vis[c.tag];
            return `<span class="gps-vis-chip" data-tag="${c.tag}" title="click to `
                + `${on ? "hide" : "show"} ${c.name}" style="cursor:pointer;`
                + `display:inline-block;padding:0 7px;margin-right:5px;`
                + `border-radius:9px;border:1px solid ${c.color};`
                + `background:${on ? c.color : "transparent"};`
                + `color:${on ? "#fff" : "#9aa4af"};font-weight:600;`
                + (on ? "" : "text-decoration:line-through;")
                + `">${c.tag}</span>`;
        }).join("");
        return {shown, locked, visible, sky_state, site, chips};
    }

    _bind_chips() {
        this._status.find(".gps-vis-chip").on("click", (ev) => {
            const tag = ev.currentTarget.dataset.tag;
            this.feed.set_vis(tag, !this.feed.vis[tag]);
        });
    }

    // One row per satellite, in one section per constellation (in the order of `chains`), under
    // one shared header: Sat · El · az · Offset · the High/Mid/Low band groups. Each section's
    // sub-header names its own signals in their band slots, so a signal is a named, sortable
    // column. The cell is the selected metric. Sorting is within each section; a band-slot
    // column sorts every section by the signal it holds in that slot.
    _render(d) {
        const {shown, locked, visible, sky_state, site, chips} = this._prep(d);
        const M = METRICS[this.metric] || METRICS.cn0;
        const CONSTS = consts_of(d);

        // Signals grouped per (constellation, band), in the server's listed order, keyed by sys
        // (the satellite's constellation), the same key as the rows.
        const byCB = {};                    // sys -> band -> [signal defs]
        const seen = {};
        for (const s of d.signals) {
            ((byCB[s.sys || s.tag] ??= {})[s.band] ??= []).push(s);
            seen[s.band] = true;
        }
        const BANDS = BAND_ORDER.filter(b => seen[b]);
        // Aligned column count per band = the most any constellation puts in it, so the shared
        // header spans consistently; a constellation with fewer pads with blanks.
        const maxCols = {};
        for (const b of BANDS)
            maxCols[b] = Math.max(0, ...CONSTS.map(c => ((byCB[c.tag] || {})[b] || []).length));
        const ncol = 4 + BANDS.reduce((n, b) => n + maxCols[b], 0);

        // A sort key is "id"/"el"/"az"/"off" (shared) or "<band>#<slot>" (a per-constellation
        // signal slot, built from BANDS). A saved key that does not validate falls back to the
        // el default.
        const valid = (k) => k === "id" || k === "el" || k === "az" || k === "off"
            || /^(high|mid|low)#\d+$/.test(k);
        if (!valid(this.usort.key)) this.usort = {key: "el", dir: -1};

        // Metric toggle and status line.
        const mbtn = Object.keys(METRICS).map(k =>
            `<span class="gps-metric" data-m="${k}" style="cursor:pointer;padding:0 6px;`
            + `border-radius:8px;margin-right:3px;`
            + (k === this.metric ? "background:#1a1e24;color:#fff;" : "color:#5a6472;")
            + `">${METRICS[k].label}</span>`).join("");
        this._status.html(
            chips + `&nbsp;<b>${locked.length}</b> locked / <b>${visible}</b> up`
            + `&nbsp; · &nbsp;cell: ${mbtn}<span style="color:#8a929b;">(${M.unit})</span>`
            + `&nbsp; · &nbsp;sky ${sky_state} · site ${site}`);
        this._bind_chips();
        this._status.find(".gps-metric").on("click", (ev) => {
            this.metric = ev.currentTarget.dataset.m;
            this._save_pref("metric", this.metric);
            if (this._last) this._render(this._last);
        });

        // Per-satellite sort value for the current key.
        const sval = (r, key) => {
            if (key === "id") return r.id;
            if (key === "el") return r.el;
            if (key === "az") return r.az;
            if (key === "off") return boresight_offset_deg(r.el, r.az);
            const [band, slot] = key.split("#");
            const sg = ((byCB[r.tag] || {})[band] || [])[+slot];
            const m = sg && r.sig_by && r.sig_by[sg.key];
            return m ? m[M.field] : null;
        };
        const {key, dir} = this.usort;
        const cmp = (a, b) => {
            const va = sval(a, key), vb = sval(b, key);
            if (va == null && vb == null) return a.id < b.id ? -1 : 1;
            if (va == null) return 1;
            if (vb == null) return -1;
            const c = key === "id" ? (va < vb ? -1 : va > vb ? 1 : 0) : va - vb;
            return c * dir || (a.id < b.id ? -1 : 1);
        };
        // Sats grouped by constellation, each section sorted independently.
        const byTag = {};
        for (const c of CONSTS) byTag[c.tag] = [];
        for (const r of shown) if (byTag[r.tag]) byTag[r.tag].push(r);
        for (const c of CONSTS) byTag[c.tag].sort(cmp);

        const arrow = (k) => this.usort.key !== k ? "" : (this.usort.dir > 0 ? " ▲" : " ▼");
        const sortTh = (k, label, align) =>
            `<th class="gps-uth" data-key="${k}" style="text-align:${align};padding:2px 5px;`
            + `cursor:pointer;user-select:none;white-space:nowrap;`
            + (k === "id" ? "width:1%;" : "")
            + (this.usort.key === k ? "color:#1a1e24;" : "color:#666;") + `">${label}${arrow(k)}</th>`;

        // Shared header: Sat | El | az | Offset | High | Mid | Low. Band groups span their columns.
        let head = "<tr style='border-bottom:1px solid #ddd;'>"
            + sortTh("id", "Sat", "left") + sortTh("el", "El", "right")
            + sortTh("az", "az", "right") + sortTh("off", "Offset", "right");
        for (const b of BANDS)
            head += `<th colspan="${maxCols[b]}" style="text-align:center;padding:2px 5px;`
                + `color:#8a929b;border-left:1px solid #eee;font-weight:600;letter-spacing:.3px;">`
                + `${BAND_LABEL[b] || b}</th>`;
        head += "</tr>";

        // Empty cells: "·" = this satellite's block does not broadcast this signal (nothing to
        // detect); "—" = it does, or that is unknown, and there is no value; "probe" = the
        // chain seeded this satellite as a noise reference on this signal (a dark probe,
        // --probe-dark: it cannot broadcast the signal), under every metric, with the value
        // in the hover; blank = this constellation has no signal in that band slot (the pad
        // to the widest constellation).
        const mcell = (m, sg, prn, extra) => {
            const v = M.fmt(m);
            if (m && m.noise_probe) {
                // Always the word, never the metric's own markup: a probe's q ~1 in red is the
                // floor doing its job, not a satellite failing. The value rides the hover, read
                // from the metric's field: a formatter's markup carries its own quotes and
                // ">" characters (q's hover says ">=2.2"), so stripping tags from it is not safe
                // inside an attribute.
                const raw = M.field ? m[M.field] : null;
                const plain = typeof raw === "number" ? raw.toFixed(2)
                            : (raw != null ? String(raw) : null);
                const tip = `noise probe on this signal${plain != null ? ` (${M.label} ${plain})` : ""}`
                    + `: the chain's floor reference, not a measurement of the satellite`;
                return `<td title="${tip.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;")}"`
                    + ` style="padding:1px 5px;text-align:right;color:#b0b6bd;${extra || ""}">`
                    + `probe</td>`;
            }
            if (v != null)
                return `<td style="padding:1px 5px;text-align:right;${extra || ""}">${v}</td>`;
            const nx = sg && not_transmitted(sg.key, prn, sg.sys || sg.tag);
            return `<td title="${nx ? "not broadcast by this satellite" : "no detection"}"`
                + ` style="padding:1px 5px;text-align:right;${nx ? "color:#c8ccd2;" : ""}`
                + `${extra || ""}">${nx ? "·" : "—"}</td>`;
        };

        // A constellation section: a divider/sub-header row (name + its signal columns) then rows.
        const sectionHeader = (c) => {
            let h = `<tr class="gps-section" style="background:#fafbfc;border-top:2px solid #e6e8eb;">`
                + `<td style="padding:3px 5px;font-weight:700;white-space:nowrap;`
                + `color:${chain_color(c.tag)};">${c.name}</td><td></td><td></td><td></td>`;
            for (const b of BANDS) {
                const sigs = (byCB[c.tag] || {})[b] || [];
                for (let i = 0; i < maxCols[b]; i++) {
                    const edge = i === 0 ? "border-left:1px solid #eee;" : "";
                    const sg = sigs[i];
                    if (!sg) { h += `<td style="${edge}"></td>`; continue; }
                    const k = b + "#" + i;
                    h += `<th class="gps-uth" data-key="${k}" title="${sg.name} — click to sort `
                        + `by ${M.label}" style="text-align:right;padding:2px 5px;cursor:pointer;`
                        + `user-select:none;white-space:nowrap;font-weight:600;${edge}`
                        + (this.usort.key === k ? "color:#1a1e24;" : "color:#6a7280;")
                        + `">${sg.col}${arrow(k)}</th>`;
                }
            }
            return h + "</tr>";
        };
        const satRow = (r) => {
            const cc = chain_color(r.tag);
            const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;`
                + `background:${r.active ? snr_color(r.sig >= 6 ? r.sig : r.snr) : "transparent"};`
                + `border:2px solid ${cc};margin-right:5px;"></span>`;
            let cells = `<td style="padding:1px 5px;width:1%;white-space:nowrap;">`
                + `<b>${dot}${r.id}</b></td>`
                + `<td style="padding:1px 5px;text-align:right;">`
                + (r.el != null ? r.el.toFixed(0) + "°" : "—") + "</td>";
            const az = r.az != null ? r.az.toFixed(0) + "°" : "—";
            cells += `<td style="padding:1px 5px;text-align:right;color:#8a929b;">${az}</td>`;
            const off = boresight_offset_deg(r.el, r.az);
            cells += `<td title="angular distance from boresight (az ${BORESIGHT_AZ_DEG}, el `
                + `${BORESIGHT_EL_DEG})" style="padding:1px 5px;text-align:right;">`
                + (off != null ? off.toFixed(1) + "°" : "—") + "</td>";
            for (const b of BANDS) {
                const sigs = (byCB[r.tag] || {})[b] || [];
                for (let i = 0; i < maxCols[b]; i++) {
                    const edge = i === 0 ? "border-left:1px solid #eee;" : "";
                    const sg = sigs[i];
                    if (!sg) { cells += `<td style="padding:1px 5px;${edge}"></td>`; continue; }
                    cells += mcell(r.sig_by[sg.key], sg, r.prn, edge);
                }
            }
            return `<tr style="${r.active ? "" : "color:#8a929b;"}">${cells}</tr>`;
        };

        let body = "";
        for (const c of CONSTS) {
            if (!d.vis[c.tag] || !byCB[c.tag]) continue;   // hidden, or constellation not present
            body += sectionHeader(c);
            const rows = byTag[c.tag];
            body += rows.length ? rows.map(satRow).join("")
                : `<tr><td colspan="${ncol}" style="padding:2px 6px;color:#aaa;`
                  + `font-style:italic;">no ${c.name} sats up</td></tr>`;
        }
        if (!body)
            body = `<tr><td colspan="${ncol}" style="padding:4px 6px;color:#999;">`
                + "no satellites…</td></tr>";

        this._table.html(`<table style="border-collapse:collapse;width:100%;font-variant-numeric:`
            + `tabular-nums;">${head}${body}</table>`);
        this._table.find(".gps-uth").on("click", (ev) => {
            const k = ev.currentTarget.dataset.key;
            if (this.usort.key === k) this.usort.dir = -this.usort.dir;
            else this.usort = {key: k, dir: k === "id" ? 1 : -1};
            this._save_pref("usort", this.usort);
            if (this._last) this._render(this._last);
        });
    }
}
