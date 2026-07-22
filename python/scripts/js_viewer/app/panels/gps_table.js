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
    {key: "dop",  label: "dop",  align: "right", dir: -1,
     tip: "tracked Doppler (Hz, receiver convention -- matches broker logs)"},
];

export class GpsTablePanel {
    constructor({target, feed, has_site}) {
        this.feed = feed;
        this.has_site = !!has_site;
        this.sort = {key: "id", dir: 1};      // stable default: constellation+PRN
        try {
            const p = JSON.parse(localStorage.getItem(PREFS_KEY));
            if (p && p.sort && COLS.some(c => c.key === p.sort.key)) this.sort = p.sort;
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

    _save_sort() {
        try {
            const p = JSON.parse(localStorage.getItem(PREFS_KEY)) || {};
            p.sort = this.sort;
            localStorage.setItem(PREFS_KEY, JSON.stringify(p));
        } catch (e) { /* private mode etc */ }
    }

    _click_sort(key) {
        if (this.sort.key === key) this.sort.dir = -this.sort.dir;
        else this.sort = {key, dir: COLS.find(c => c.key === key).dir};
        this._save_sort();
        if (this._last) this._render(this._last);
    }

    _render({sats, sky, adc, vis, chains}) {
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
