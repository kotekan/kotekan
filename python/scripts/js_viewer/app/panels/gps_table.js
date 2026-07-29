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

import {chain_color} from "./gps_feed.js";
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

    // ONE ROW PER SATELLITE, one column per signal, grouped by frequency band -- the
    // shared-knowledge view: L1 C/A, L1C, L2C-CM, L2C-CL, L5 are the same satellite. GPS carries
    // the most signals per band, so its signals define the column skeleton; Galileo/BeiDou put
    // their single per-band signal in a cell spanning that band's columns. Cell = the selected
    // metric (C/N0 default). '—' = not carried / not locked.
    _render_unified(d) {
        const {shown, locked, visible, sky_state, rms, site, chips} = this._prep(d);
        const M = METRICS[this.metric] || METRICS.cn0;

        // Column skeleton from GPS; per-(const,band) lookup for the others.
        const gps = d.signals.filter(s => s.tag === "G");
        const bands = [];
        for (const s of gps) if (!bands.includes(s.band)) bands.push(s.band);
        const bandCols = {};
        for (const b of bands) bandCols[b] = gps.filter(s => s.band === b);
        const other = {};   // "E:L1" -> signal
        for (const s of d.signals) if (s.tag !== "G") other[s.tag + ":" + s.band] = s;

        // Metric toggle + status.
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

        // Sort. Keys: "id", "el", a signal's combiner (by the selected metric), or "best".
        const sval = (r, key) => {
            if (key === "id") return r.id;
            if (key === "el") return r.el;
            if (key === "best") return r.sig || 0;
            const m = r.sig_by && r.sig_by[key];
            return m ? m[M.field] : null;
        };
        const {key, dir} = this.usort;
        shown.sort((a, b) => {
            const va = sval(a, key), vb = sval(b, key);
            if (va == null && vb == null) return a.id < b.id ? -1 : 1;
            if (va == null) return 1;
            if (vb == null) return -1;
            const c = key === "id" ? (va < vb ? -1 : va > vb ? 1 : 0) : va - vb;
            return c * dir || (a.id < b.id ? -1 : 1);
        });

        const arrow = (k) => this.usort.key !== k ? ""
            : (this.usort.dir > 0 ? " ▲" : " ▼");
        // Two-row header: band groups over their signal columns; Sat/El/az span both.
        const th = (k, label, align, span) =>
            `<th class="gps-uth" data-key="${k}"${span ? ` rowspan="${span}"` : ""}`
            + ` style="text-align:${align};padding:1px 5px;cursor:pointer;user-select:none;`
            + `white-space:nowrap;` + (this.usort.key === k ? "color:#1a1e24;" : "color:#666;")
            + `">${label}${arrow(k)}</th>`;
        let h1 = "<tr style='border-bottom:1px solid #eee;'>"
            + th("id", "Sat", "left", 2) + th("el", "El", "right", 2);
        let h2 = "<tr style='color:#888;border-bottom:1px solid #ddd;'>";
        for (const b of bands) {
            h1 += `<th colspan="${bandCols[b].length}" style="text-align:center;`
                + `padding:1px 5px;color:#8a929b;border-left:1px solid #eee;`
                + `font-weight:600;">${b}</th>`;
            bandCols[b].forEach((sg, i) =>
                h2 += `<th class="gps-uth" data-key="${sg.key}" title="${sg.name} — `
                    + `click to sort by ${M.label}" style="text-align:right;padding:1px 5px;`
                    + `cursor:pointer;user-select:none;white-space:nowrap;`
                    + (i === 0 ? "border-left:1px solid #eee;" : "")
                    + (this.usort.key === sg.key ? "color:#1a1e24;" : "color:#666;")
                    + `">${sg.col}${arrow(sg.key)}</th>`);
        }
        h1 += th("best", "az", "right", 2) + "</tr>";
        h2 += "</tr>";

        const mcell = (m, extra) =>
            `<td style="padding:1px 5px;text-align:right;${extra || ""}">`
            + (M.fmt(m) != null ? M.fmt(m) : "—") + "</td>";
        const row = (r) => {
            const cc = chain_color(r.tag);
            const dot = `<span style="display:inline-block;width:8px;height:8px;`
                + `border-radius:50%;background:${r.active ? snr_color(r.sig >= 6 ? r.sig : r.snr) : "transparent"};`
                + `border:2px solid ${cc};margin-right:5px;"></span>`;
            let cells = `<td style="padding:1px 5px;"><b>${dot}${r.id}</b></td>`
                + `<td style="padding:1px 5px;text-align:right;">`
                + (r.el != null ? r.el.toFixed(0) + "°" : "—") + "</td>";
            for (const b of bands) {
                if (r.tag === "G") {
                    bandCols[b].forEach((sg, i) =>
                        cells += mcell(r.sig_by[sg.key],
                                       i === 0 ? "border-left:1px solid #eee;" : ""));
                } else {
                    const sg = other[r.tag + ":" + b];
                    const span = bandCols[b].length;
                    if (!sg) {
                        cells += `<td colspan="${span}" style="padding:1px 5px;text-align:`
                            + `center;color:#c9ced4;border-left:1px solid #eee;">·</td>`;
                    } else {
                        const m = r.sig_by[sg.key];
                        const v = M.fmt(m);
                        cells += `<td colspan="${span}" title="${sg.name}" style="padding:1px 5px;`
                            + `text-align:right;border-left:1px solid #eee;">`
                            + (v != null ? v : "—")
                            + ` <span style="color:#9aa4af;font-size:10px;">${sg.col}</span></td>`;
                    }
                }
            }
            const az = r.az != null ? r.az.toFixed(0) + "°" : "—";
            cells += `<td style="padding:1px 5px;text-align:right;color:#8a929b;">${az}</td>`;
            return `<tr style="${r.active ? "" : "color:#8a929b;"}">${cells}</tr>`;
        };
        const ncol = 3 + bands.reduce((n, b) => n + bandCols[b].length, 0);
        let body = shown.map(row).join("");
        if (!shown.length)
            body = `<tr><td colspan='${ncol}' style='padding:4px 6px;color:#999;'>`
                + "no satellites…</td></tr>";
        this._table.html(`<table style="border-collapse:collapse;width:100%;font-variant-numeric:`
            + `tabular-nums;">${h1}${h2}${body}</table>`);
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
