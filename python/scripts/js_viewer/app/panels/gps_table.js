// GNSS detections table, its own resizable card (the skyplot is gps_sky.js;
// both consume the shared GpsFeed). One row per satellite that is either
// locked or above the horizon; unlocked rows are dimmed rather than hidden so
// you can see what's overhead waiting to be acquired.
//
// Columns are CLICK-SORTABLE: click a header to sort on it, click again to
// flip direction. The sort is stable (satellite id tiebreak) and sticky
// (persisted per-browser), so the table doesn't reshuffle every poll the way
// the old fixed sort-by-search-SNR did.
//
// Column choices: Sat (constellation chip + id), El, search SNR (sigma, from
// the acquire grid), C/N0 (dB-Hz, the calibrated incoherent estimator -- the
// beam-map observable), sig (deep/incoherent significance, sigma above noise),
// coh (the auto-ladder's winning coherent window), dop (tracked Doppler,
// matches the broker logs). The old raw |A| and Â columns are gone: C/N0 is
// the calibrated version of Â, and raw |A| is noise-biased (≈ the floor for
// weak sats) -- it mostly confused. Band is implied by the viewer instance
// (all three constellations here share the 1575.42 tune).

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
    cn0:  {label: "C/N0", field: "cn0",     unit: "dB-Hz",
           fmt: m => m && m.cn0 != null ? m.cn0.toFixed(0) : null},
    sig:  {label: "sig",  field: "sig",     unit: "σ",
           fmt: m => m && m.sig ? m.sig.toFixed(0) : null},
    coh:  {label: "coh",  field: "coh_s",   unit: "s",
           fmt: m => m && m.coh_s > 0 ? m.coh_s.toFixed(2) : null},
    peel: {label: "peel", field: "peel_db", unit: "dB",
           fmt: m => m && m.peel_db != null
                     ? (m.peel_bound ? "≥" : "") + m.peel_db.toFixed(0) : null},
};

// Frequency-band grouping shared across every constellation (2026-07-31). L1/E1/B1 = HIGH
// (~1575 MHz), L2 = MID (~1227, GPS only for now), L5/E5a/B2a = LOW (~1176). Glonass will slot
// in the same three when it lands. The per-band SIGNALS differ per constellation, so the
// constellation is its own section below the shared header rather than a grey side-note.
const BAND_LABEL = {L1: "High", L2: "Mid", L5: "Low"};
const BAND_ORDER = ["L1", "L2", "L5"];
const CONSTS = [{tag: "G", name: "GPS"}, {tag: "E", name: "Galileo"}, {tag: "C", name: "BeiDou"}];

const COLS = [
    {key: "id",   label: "Sat",  align: "left",  dir: 1,
     tip: "constellation + PRN (G GPS / E Galileo / C BeiDou)"},
    {key: "el",   label: "El",   align: "right", dir: -1,
     tip: "elevation (deg)"},
    {key: "snr",  label: "SNR",  align: "right", dir: -1,
     tip: "search detection significance (sigma above the acquire grid noise)"},
    {key: "cn0",  label: "C/N0", align: "right", dir: -1,
     tip: "incoherent C/N0 (dB-Hz, pipeline zero-point): needs only 1 record "
          + "of coherence -- the beam-map observable"},
    {key: "sig",  label: "sig",  align: "right", dir: -1,
     tip: "combined significance = signal / its uncertainty (sigma above "
          + "noise, deep nav-wiped when available): >>1 real, ~1 noise"},
    {key: "coh_s", label: "coh", align: "right", dir: -1,
     tip: "coherent window the deep integration held (s, auto-ladder winner)"},
    {key: "peel_db", label: "peel", align: "right", dir: -1,
     tip: "fused voltage-peel depth (dB, deep/residual-deep): how much of this "
          + "sat's signal the peel removed from the voltage. ≥ = residual at the "
          + "combiner floor (true depth deeper). — = chain not peeling / no valid "
          + "deep lock"},
    {key: "dop",  label: "dop",  align: "right", dir: -1,
     tip: "tracked Doppler (Hz, receiver convention -- matches broker logs)"},
];

export class GpsTablePanel {
    constructor({target, feed, has_site}) {
        this.feed = feed;
        this.has_site = !!has_site;
        this.sort = {key: "id", dir: 1};      // stable default: constellation+PRN
        this.usort = {key: "el", dir: -1};    // unified default: highest sat first
        this.metric = "cn0";                  // unified cell metric
        try {
            const p = JSON.parse(localStorage.getItem(PREFS_KEY));
            if (p && p.sort && COLS.some(c => c.key === p.sort.key)) this.sort = p.sort;
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

    _save_sort() { this._save_pref("sort", this.sort); }

    _click_sort(key) {
        if (this.sort.key === key) this.sort.dir = -this.sort.dir;
        else this.sort = {key, dir: COLS.find(c => c.key === key).dir};
        this._save_sort();
        if (this._last) this._render(this._last);
    }

    _render(d) {
        if (d.unified && Array.isArray(d.signals) && d.signals.length)
            return this._render_unified(d);
        return this._render_flat(d);
    }

    // Which rows to show + the status line's sky/site/ADC summary + the constellation chips.
    // Shared by both renderers (the flat per-band table and the unified matrix).
    _prep(d) {
        const {sats, sky, adc, vis, chains} = d;
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
        const rms = adc && adc.rms != null ? adc.rms.toFixed(0) : "—";
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
        return {shown, locked, visible, sky_state, rms, site, chips};
    }

    _bind_chips() {
        this._status.find(".gps-vis-chip").on("click", (ev) => {
            const tag = ev.currentTarget.dataset.tag;
            this.feed.set_vis(tag, !this.feed.vis[tag]);
        });
    }

    // ONE ROW PER SATELLITE, split into THREE CONSTELLATION SECTIONS (GPS, Galileo, BeiDou)
    // stacked in that fixed order, under ONE shared header: Sat · El · High/Mid/Low band groups ·
    // az. Each constellation's actual signals are its own sub-header under the bands (GPS: CA/L1C
    // High, CM/CL Mid, Q/I Low; Galileo/BeiDou: pilot+data in High and Low, no Mid) -- so a signal
    // is a named, sortable column instead of a grey side-note. Cell = the selected metric (C/N0
    // default). Sorting sorts WITHIN each section (section order stays GPS→Gal→BeiDou); a band-slot
    // column sorts every section by the signal it holds in that slot (its High pilot, etc.).
    _render_unified(d) {
        const {shown, locked, visible, sky_state, rms, site, chips} = this._prep(d);
        const M = METRICS[this.metric] || METRICS.cn0;

        // Signals grouped per (constellation, band), in the server's listed order.
        const byCB = {};                    // tag -> band -> [signal defs]
        const seen = {};
        for (const s of d.signals) {
            ((byCB[s.tag] ??= {})[s.band] ??= []).push(s);
            seen[s.band] = true;
        }
        const BANDS = BAND_ORDER.filter(b => seen[b]);
        // Aligned column count per band = the most any constellation puts in it (so the shared
        // High/Mid/Low header spans consistently; a constellation with fewer pads with blanks).
        const maxCols = {};
        for (const b of BANDS)
            maxCols[b] = Math.max(0, ...CONSTS.map(c => ((byCB[c.tag] || {})[b] || []).length));
        const ncol = 3 + BANDS.reduce((n, b) => n + maxCols[b], 0);

        // A sort key is "id"/"el"/"az" (shared) or "<band>#<slot>" (a per-constellation signal
        // slot). Guard a stale saved key from the old per-combiner layout back to the el default.
        const valid = (k) => k === "id" || k === "el" || k === "az" || /^(L1|L2|L5)#\d+$/.test(k);
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
            + `&nbsp; · &nbsp;sky ${sky_state} · site ${site} · ADC ${rms}`);
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

        // Shared header: Sat | El | High | Mid | Low | az. Band groups span their aligned columns.
        let head = "<tr style='border-bottom:1px solid #ddd;'>"
            + sortTh("id", "Sat", "left") + sortTh("el", "El", "right");
        for (const b of BANDS)
            head += `<th colspan="${maxCols[b]}" style="text-align:center;padding:2px 5px;`
                + `color:#8a929b;border-left:1px solid #eee;font-weight:600;letter-spacing:.3px;">`
                + `${BAND_LABEL[b] || b}</th>`;
        head += sortTh("az", "az", "right") + "</tr>";

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
                + `color:${chain_color(c.tag)};">${c.name}</td><td></td>`;
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
            return h + "<td></td></tr>";
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
            for (const b of BANDS) {
                const sigs = (byCB[r.tag] || {})[b] || [];
                for (let i = 0; i < maxCols[b]; i++) {
                    const edge = i === 0 ? "border-left:1px solid #eee;" : "";
                    const sg = sigs[i];
                    if (!sg) { cells += `<td style="padding:1px 5px;${edge}"></td>`; continue; }
                    cells += mcell(r.sig_by[sg.key], sg, r.prn, edge);
                }
            }
            const az = r.az != null ? r.az.toFixed(0) + "°" : "—";
            cells += `<td style="padding:1px 5px;text-align:right;color:#8a929b;">${az}</td>`;
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

    _render_flat({sats, sky, adc, vis, chains}) {
        // Below-horizon "locked" PRNs are noise probes (--noise-probes) or false locks:
        // kotekan runs a correlator on them so they report active, but the authoritative
        // BRDC sky -- which now covers the full constellation -- does not place them above
        // the horizon (el stays null). Hide them when the geometry is valid so they don't
        // jumble the list with flickering '--' rows; if the sky feed is down (not ok) fall
        // back to showing them, so a geometry outage never blanks the whole list.
        const sky_ok = !!(sky && sky.ok);
        // A sat is shown if it is above the horizon (real, visible) or actively tracked. HIDE
        // actively-tracked rows the authoritative BRDC sky places BELOW the horizon -- those are
        // noise probes / false locks. Transition-safe: catches both the server's real negative
        // elevation (el < 0) and, before the /gps_sky floor change is live, the masked-out null.
        // Fall back to showing them if the sky feed is down, so an outage never blanks the list.
        const below = r => sky_ok && r.active && (r.el == null || r.el < 0);
        const shown = sats.filter(r => vis[r.tag]
                                       && (r.active || (r.el != null && r.el >= 0))
                                       && !below(r));
        const locked = shown.filter(r => r.active);
        const visible = shown.filter(r => r.el != null && r.el >= 0).length;

        // Status line + constellation toggle chips.
        let sky_state;
        if (!this.has_site) sky_state = "<b>no site</b> (set LAT/LON)";
        else if (sky && sky.ok) sky_state = "ok";
        else if (sky && sky.computing) sky_state = "loading…";
        else sky_state = sky && sky.error ? "unavailable" : "—";
        const rms = adc && adc.rms != null ? adc.rms.toFixed(0) : "—";
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
        this._status.html(
            chips + `&nbsp;<b>${locked.length}</b> locked / <b>${visible}</b> up`
            + `&nbsp; · &nbsp;sky: ${sky_state}&nbsp; · &nbsp;site ${site}`
            + `&nbsp; · &nbsp;ADC rms ${rms}`);
        this._status.find(".gps-vis-chip").on("click", (ev) => {
            const tag = ev.currentTarget.dataset.tag;
            this.feed.set_vis(tag, !this.feed.vis[tag]);
        });

        // Sort: chosen column, satellite id as the stable tiebreak. null/missing
        // values always sink to the bottom regardless of direction.
        const {key, dir} = this.sort;
        shown.sort((a, b) => {
            const va = a[key], vb = b[key];
            if (va == null && vb == null) return a.id < b.id ? -1 : 1;
            if (va == null) return 1;
            if (vb == null) return -1;
            const d = key === "id" ? (va < vb ? -1 : va > vb ? 1 : 0) : va - vb;
            return d * dir || (a.id < b.id ? -1 : 1);
        });

        const cell = (s, align) =>
            `<td style="padding:1px 6px;text-align:${align};">${s}</td>`;
        const row = (r) => {
            const cc = chain_color(r.tag);
            const dot = `<span style="display:inline-block;width:8px;height:8px;`
                + `border-radius:50%;background:${r.active ? snr_color(r.snr) : "transparent"};`
                + `border:2px solid ${cc};margin-right:5px;"></span>`;
            const sig = r.sig
                ? `<span title="${(20 * Math.log10(r.sig)).toFixed(0)} dB">`
                  + r.sig.toFixed(1) + "σ</span>"
                : "—";
            const vals = {
                id: dot + "<b>" + r.id + "</b>",
                el: r.el != null ? r.el.toFixed(0) + "°" : "—",
                snr: r.snr != null ? r.snr.toFixed(1) : "—",
                cn0: r.cn0 != null ? r.cn0.toFixed(1) : "—",
                sig,
                coh_s: r.coh_s != null && r.coh_s > 0 ? r.coh_s.toFixed(2) : "—",
                peel_db: r.peel_db != null
                    ? (r.peel_bound ? "≥" : "") + r.peel_db.toFixed(1)
                    : "—",
                dop: r.dop != null ? r.dop.toFixed(0) : "—",
            };
            return `<tr style="${r.active ? "" : "color:#8a929b;"}">`
                + COLS.map(c => cell(vals[c.key], c.align)).join("") + "</tr>";
        };
        const arrow = (c) => this.sort.key !== c.key ? ""
            : (this.sort.dir > 0 ? " ▲" : " ▼");
        const head = "<tr style='color:#666;border-bottom:1px solid #ddd;'>"
            + COLS.map(c =>
                `<th class="gps-th" data-key="${c.key}" title="${c.tip} — click to sort"`
                + ` style="text-align:${c.align};padding:1px 6px;cursor:pointer;`
                + `user-select:none;white-space:nowrap;`
                + (this.sort.key === c.key ? "color:#1a1e24;" : "")
                + `">${c.label}${arrow(c)}</th>`).join("")
            + "</tr>";
        let body = shown.map(row).join("");
        if (!shown.length)
            body = `<tr><td colspan='${COLS.length}' style='padding:4px 6px;`
                + `color:#999;'>no satellites…</td></tr>`;
        this._table.html(`<table style="border-collapse:collapse;width:100%;">`
            + head + body + "</table>");
        this._table.find(".gps-th").on("click",
            (ev) => this._click_sort(ev.currentTarget.dataset.key));
    }
}
