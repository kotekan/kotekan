// GNSS detections table, its own resizable card (the skyplot is gps_sky.js;
// both consume the shared GpsFeed). One row per satellite that is either
// locked or above the horizon; unlocked rows are dimmed rather than hidden so
// you can see what's overhead waiting to be acquired. One column per signal,
// grouped by band; the cell shows the selected metric (METRICS below).
//
// Columns are CLICK-SORTABLE: click a header to sort on it, click again to
// flip direction. The sort is stable (satellite id tiebreak) and sticky
// (persisted per-browser), so the table doesn't reshuffle every poll.

import {chain_color, not_transmitted} from "./gps_feed.js";
import {snr_color} from "./gps_sky.js";

const PREFS_KEY = "gps_viewer_prefs_v1";

// UNIFIED matrix (2026-07-28): the cell metric, switchable. Each maps to a field on the
// per-signal metrics object (gps_feed.signal_metrics) + a formatter. Integer C/N0 / sig keep
// the many columns tight; peel carries the ">=" lower-bound flag.
const METRICS = {
    // SNR = SEARCH detection significance, the only metric here that reports on ACQUISITION
    // rather than on the deep integration. Restored 2026-07-29 (it existed as a fixed column
    // before the unified matrix): the diagnostic value is precisely the DISAGREEMENT -- a
    // strong SNR beside a dead sig/coh says the signal is there and something downstream of
    // the search is wrong, which no deep metric can tell you on its own. Blank for a DERIVED
    // signal (L2C-CL, L5-I): those are seeded from a sibling and never searched.
    snr:  {label: "SNR",  field: "snr",     unit: "σ",
           fmt: m => m && m.snr ? m.snr.toFixed(0) : null},
    // ONE C/N0 (task #57, 2026-08-15). The two-estimator display (coh + inc, both named) is
    // retired: the incoherent number was biased (undebised sigma^2 -- the rectification
    // floor) and the coherent one rode the deep fold's per-integration rate re-search
    // (~20 dB of its own paired scatter, #47/#66). This is cn0_prompt_db -- per-record
    // prompt power, q-gated, probe-debiased, sky-validated -- and it is conditioned on
    // lock, so its DUTY travels with it: a value at duty < 70% is greyed with the duty in
    // the hover (few gate-passing records; read it as "when held", not "always").
    // TWO C/N0s AGAIN (2026-08-15) -- but both HONEST this time, and they mean different
    // things rather than duplicating one measurement badly. `cn0` is per-record prompt
    // power (q-gated, probe-debiased): no coherence assumed, defined whenever tracking
    // held. `cn0k` is the ~1 s KNOWN-RATE coherent fold: ~10log10(n_rec) more sensitive --
    // the deep-sidelobe instrument -- and available wherever the carrier stayed coherent.
    // On a strong satellite they must AGREE; a low coherent value beside a healthy
    // incoherent one means COHERENCE was lost, not signal, and `eta` says so directly.
    cn0:  {label: "C/N0 prompt", field: "cn0", unit: "dB-Hz",
           fmt: m => {
               if (!m || m.cn0 == null) return null;
               const v = m.cn0.toFixed(0);
               if (m.cn0_duty == null) return v + "*";   // pre-#57 broker: old estimator
               return m.cn0_duty < 0.7
                   ? `<span style="opacity:.55" title="lock duty ${(100 * m.cn0_duty).toFixed(0)}%`
                     + ` -- the C/N0 of the records where tracking held">${v}</span>`
                   : v;
           }},
    cn0k: {label: "C/N0 coh", field: "cn0_kcoh", unit: "dB-Hz",
           fmt: m => {
               if (!m || m.cn0_kcoh == null) return null;
               const v = m.cn0_kcoh.toFixed(0);
               // Greyed when the fold did not actually cohere (eta << n_rec): the number
               // is then a floor, not a measurement, and the reason is IN the hover.
               const eta = m.kcoh_eta, n = m.kcoh_n;
               if (eta != null && n && eta < 0.25 * n)
                   return `<span style="opacity:.55" title="coherence efficiency ${eta.toFixed(0)}`
                        + `/${n} records -- the fold barely cohered (stale or wrong residual`
                        + ` rate, or a phase discontinuity), so this is a LOWER BOUND">${v}</span>`;
               return v;
           }},
    // BEAM (task #57 step 2): the per-element complex gain, summarised.
    //   beam A   = median live-element gain amplitude, 1.0 = uniform array.
    //   φ spread = the CIRCULAR RMS of the live elements' phases about the array mean, in
    //              degrees. ⚠️ IT IS A DISPERSION, NOT AN ANGLE -- 0 means the array is
    //              phased up on this satellite, 60+ means the elements disagree. It was
    //              labelled "beam φ", which reads as a phase and was taken for one; the
    //              per-element phases themselves (the actual peel coefficients) are 32
    //              numbers per instance and live on /get_elements + the NFS archive.
    beam_amp: {label: "beam A", field: "beam_amp", unit: "x",
               fmt: m => m && m.beam_amp != null ? m.beam_amp.toFixed(2) : null},
    beam_ph:  {label: "φ spread", field: "beam_ph", unit: "° rms",
               fmt: m => m && m.beam_ph != null ? m.beam_ph.toFixed(0) + "°" : null},
    // DUTY (task #57): the fraction of the last 2 minutes in which the array actually
    // held this satellite. sig says how loud, duty says how often -- different
    // questions. Measured 2026-08-14: satellites sitting at sig 20-40 were being lost a
    // third of the time, all 12 instances dropping together on the seed-POST cadence,
    // and no instantaneous column could show it.
    duty: {label: "duty", field: "duty",    unit: "",
           fmt: m => m && m.duty != null ? (100 * m.duty).toFixed(0) + "%" : null},
    hold: {label: "hold", field: "hold",    unit: "x",
           fmt: m => m && m.hold ? m.hold.toFixed(1) : null},
    // q -- THE CODE-LOOP QUALITY (2026-08-18). The fleet discriminator's prompt-vs-shoulder
    // ratio: "is the code loop on the peak", which every power metric in this table is blind
    // to (the deep fold re-searches, so sig/C-N0 can be healthy while the prompt tap sits on
    // noise -- #47). Published by the broker as fleet_q since the fleet DLL landed; unshown
    // until now, which is why a 2x q regression on gal_e5b (#87) ran for 3.5 h with the C/N0
    // columns looking normal. Coloured against the broker's OWN measured floor rather than a
    // hardcoded bar: >= 2.2 is a solid lock, floor-to-2.2 is marginal, at/below floor is
    // off-peak. Hover carries the floor and the contributing instance count.
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
    peel: {label: "peel", field: "peel_db", unit: "dB",
           fmt: m => m && m.peel_db != null
                     ? (m.peel_bound ? "≥" : "") + m.peel_db.toFixed(0) : null},
};

// Frequency-band grouping shared across every constellation (2026-07-31). L1/E1/B1 = HIGH
// (~1575 MHz), mid (~1227-1279: L2C, B3I, E6), low (~1176-1207: the E5/B2 complex).
// GLONASS will slot into the same three when it lands. The per-band SIGNALS differ per constellation, so the
// constellation is its own section below the shared header rather than a grey side-note.
// Column-group key -> display name, and the left-to-right order. The keys are high/mid/low
// (they were L1/L2/L5 until 2026-09-17 -- GPS signal names for groups that are
// multi-constellation and have always been SHOWN as High/Mid/Low). Mirrored in
// app/panels/decode_health.js and livebeam_server.py -- rename in all three or not at all.
const BAND_LABEL = {high: "High", mid: "Mid", low: "Low"};
const BAND_ORDER = ["high", "mid", "low"];
// NOMINAL BORESIGHT, for the Offset column: the dishes point at dec +40.73 through the local
// meridian (config/pointings.yaml), i.e. az 180, el 81.41 at the site -- the same constants as
// the broker's sky.BORESIGHT_AZ_DEG/EL_DEG.
// TODO: serve the current pointing from the feed (derived from config/pointings.yaml) instead of
// hardcoding it here and in the broker; a repoint must change one file, not three.
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
        // Default to THE C/N0 (cn0_prompt, task #57). A stored preference still wins, but a
        // stale "cn0_coh" preference (the retired metric) fails the METRICS check below and
        // falls back here rather than resurrecting the deep-fold column.
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

    // ONE ROW PER SATELLITE, split into THREE CONSTELLATION SECTIONS (GPS, Galileo, BeiDou)
    // stacked in that fixed order, under ONE shared header: Sat · El · az · Offset · High/Mid/Low
    // band groups. Each constellation's actual signals are its own sub-header under the bands (GPS: CA/L1C
    // High, CM/CL Mid, Q/I Low; Galileo/BeiDou: pilot+data in High and Low, no Mid) -- so a signal
    // is a named, sortable column instead of a grey side-note. Cell = the selected metric (C/N0
    // default). Sorting sorts WITHIN each section (section order stays GPS→Gal→BeiDou); a band-slot
    // column sorts every section by the signal it holds in that slot (its High pilot, etc.).
    _render(d) {
        const {shown, locked, visible, sky_state, site, chips} = this._prep(d);
        const M = METRICS[this.metric] || METRICS.cn0;
        const CONSTS = consts_of(d);

        // Signals grouped per (constellation, band), in the server's listed order. Grouped by
        // SYS (the satellite's constellation), not by the display tag -- GPS L1C's tag is "L"
        // but its satellite is a GPS bird, so its columns belong in the GPS section beside C/A.
        const byCB = {};                    // sys -> band -> [signal defs]
        const seen = {};
        for (const s of d.signals) {
            ((byCB[s.sys || s.tag] ??= {})[s.band] ??= []).push(s);
            seen[s.band] = true;
        }
        const BANDS = BAND_ORDER.filter(b => seen[b]);
        // Aligned column count per band = the most any constellation puts in it (so the shared
        // High/Mid/Low header spans consistently; a constellation with fewer pads with blanks).
        const maxCols = {};
        for (const b of BANDS)
            maxCols[b] = Math.max(0, ...CONSTS.map(c => ((byCB[c.tag] || {})[b] || []).length));
        const ncol = 4 + BANDS.reduce((n, b) => n + maxCols[b], 0);

        // A sort key is "id"/"el"/"az"/"off" (shared) or "<band>#<slot>" (a per-constellation
        // signal slot). Guard a stale saved key from the old per-combiner layout back to the el
        // default. The slot keys are built from BANDS (high/mid/low), so that is what must pass.
        const valid = (k) => k === "id" || k === "el" || k === "az" || k === "off"
            || /^(high|mid|low)#\d+$/.test(k);
        if (!valid(this.usort.key)) this.usort = {key: "el", dir: -1};

        // Metric toggle + status line (unchanged behaviour).
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
        // detect); "—" = it does and we have no value; blank = this constellation has no signal in
        // that band slot at all (the aligned-Mid pad for Galileo/BeiDou).
        const mcell = (m, sg, prn, extra) => {
            const v = M.fmt(m);
            if (v != null)
                return `<td style="padding:1px 5px;text-align:right;${extra || ""}">${v}</td>`;
            const nx = sg && not_transmitted(sg.key, prn);
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
