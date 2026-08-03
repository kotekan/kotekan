// Nav-decode health panel: what the on-node decoders are doing, per civil data signal.
//
// Data is the viewer server's /decode_health endpoint, which merges the per-broker
// decode_health/*.json files (see python/scripts/gnss/decode_health.py). One row per
// decoded signal, grouped GPS -> Galileo -> BeiDou; each shows how many satellites are
// SYNCED, how many yielded a full EPHEMERIS, the BRDC-agreement residual (dpos, the
// "is our decode trustworthy" number), and how fresh the newest decode is. A signal whose
// broker file goes stale drops out of the endpoint entirely -> the row disappears, which is
// the honest thing on a decode outage (no frozen last-value).
//
// Self-contained: fetches its own endpoint on a timer (same origin as the viewer, so no
// CORS), independent of the kotekan-REST GpsFeed the sky/table use.

const CONSTS = [
    {tag: "G", name: "GPS"},
    {tag: "E", name: "Galileo"},
    {tag: "C", name: "BeiDou"},
];
const BAND_LABEL = {L1: "High", L2: "Mid", L5: "Low"};
// Friendly signal names keyed on the decode_health signal id.
const SIG_NAME = {
    GPS_L1_LNAV: "L1 C/A · LNAV", GPS_L2C_CNAV: "L2C · CNAV", GPS_L5_CNAV: "L5-I · CNAV",
    GAL_E1B_INAV: "E1-B · I/NAV", GAL_E5AI_FNAV: "E5a-I · F/NAV",
    BDS_B1C_BCNAV1: "B1C-D · B-CNAV1", BDS_B2A_BCNAV2: "B2a-D · B-CNAV2",
};

function dot(color, title) {
    return `<span title="${title || ""}" style="display:inline-block;width:9px;height:9px;`
         + `border-radius:50%;background:${color};vertical-align:middle"></span>`;
}

// Overall per-signal status: green healthy, amber degraded/stale, grey no-sync.
// Freshest decode across a signal's satellites = the LIVENESS number. A single set /
// false-lock straggler must not make an actively-decoding chain look dead, so the headline is
// min(last_s), not the worst sat. `nStale` (last_s > 300 s) is surfaced separately.
function freshest(sig) {
    const v = (sig.sats || []).map(s => s.last_s).filter(x => x != null);
    return v.length ? Math.min(...v) : null;
}
function nStale(sig) {
    return (sig.sats || []).filter(s => s.last_s != null && s.last_s > 300).length;
}

function status(sig) {
    const f = freshest(sig);
    const fresh = f == null || f < 120;
    if (!sig.n_sync) return {c: "#8a8f98", t: "no satellite synced"};
    if (!fresh) return {c: "#d0a215", t: "freshest decode is stale (>120 s)"};
    // dpos gate only where an ephemeris (and thus a cross-check) exists.
    if (sig.worst_dpos_m != null && sig.worst_dpos_m > 5)
        return {c: "#d0a215", t: `BRDC dpos ${sig.worst_dpos_m.toFixed(1)} m (>5 m)`};
    return {c: "#2ea043", t: "synced, fresh, BRDC-agreeing"};
}

function fmt(v, unit, dp) {
    if (v == null) return "<span style='color:#8a8f98'>–</span>";
    return `${v.toFixed(dp == null ? 0 : dp)}${unit || ""}`;
}

export class DecodeHealthPanel {
    constructor({target, poll_ms}) {
        this.root = document.getElementById(target);
        this.poll_ms = poll_ms || 5000;
        if (this.root) {
            this.root.style.overflow = "auto";
            this.root.style.font = "12px/1.4 system-ui, sans-serif";
        }
        this._tick();
        this._timer = setInterval(() => this._tick(), this.poll_ms);
    }

    destroy() { if (this._timer) clearInterval(this._timer); }

    async _tick() {
        let data;
        try {
            const r = await fetch("decode_health", {cache: "no-store"});
            data = await r.json();
        } catch (e) {
            this._render_error(e);
            return;
        }
        this._render(data);
    }

    _render_error(e) {
        if (this.root) this.root.innerHTML =
            `<div style="color:#8a8f98;padding:8px">decode-health unavailable: ${e}</div>`;
    }

    _render(data) {
        if (!this.root) return;
        const signals = (data && data.signals) || {};
        const chains = (data && data.chains) || [];
        const staleWorst = chains.reduce((m, c) => Math.max(m, c.t_age_s || 0), 0);

        let html = `<table style="border-collapse:collapse;width:100%">`;
        html += `<thead><tr style="text-align:left;color:#8a8f98;border-bottom:1px solid #333">`
              + `<th style="padding:2px 6px">Signal</th><th>Band</th>`
              + `<th title="satellites synced / seen">Sync</th>`
              + `<th title="satellites with a full decoded ephemeris">Eph</th>`
              + `<th title="BRDC position-agreement residual, median / worst">dpos m</th>`
              + `<th title="age of the newest decode">Fresh</th><th></th></tr></thead><tbody>`;

        let any = false;
        for (const {tag, name} of CONSTS) {
            const rows = Object.entries(signals)
                .filter(([, s]) => s.sys === tag)
                .sort((a, b) => (a[1].band > b[1].band ? 1 : -1));
            if (!rows.length) continue;
            html += `<tr><td colspan="7" style="padding:6px 6px 1px;color:#cbd5e1;`
                  + `font-weight:600">${name}</td></tr>`;
            for (const [sig, s] of rows) {
                any = true;
                const st = status(s);
                const dp = (s.median_dpos_m == null && s.worst_dpos_m == null)
                    ? "<span style='color:#8a8f98'>–</span>"
                    : `${fmt(s.median_dpos_m, "", 2)} / ${fmt(s.worst_dpos_m, "", 2)}`;
                const ns = nStale(s);
                const freshCell = `${fmt(freshest(s), " s", 0)}`
                    + (ns ? ` <span style="color:#8a8f98" title="${ns} sat(s) not decoded `
                            + `in >300 s (set / false lock)">(+${ns})</span>` : "");
                html += `<tr style="border-bottom:1px solid #222">`
                      + `<td style="padding:2px 6px">${SIG_NAME[sig] || sig}</td>`
                      + `<td>${BAND_LABEL[s.band] || s.band}</td>`
                      + `<td>${s.n_sync}/${s.n_sats}</td>`
                      + `<td>${fmt(s.n_eph, "", 0)}</td>`
                      + `<td>${dp}</td>`
                      + `<td>${freshCell}</td>`
                      + `<td>${dot(st.c, st.t)}</td></tr>`;
            }
        }
        if (!any)
            html += `<tr><td colspan="7" style="padding:8px;color:#8a8f98">`
                  + `no decoders reporting (brokers not up, or --decode-health-file unset)</td></tr>`;
        html += `</tbody></table>`;
        html += `<div style="color:#8a8f98;padding:4px 6px;font-size:11px">`
              + `${chains.length} broker${chains.length === 1 ? "" : "s"} reporting`
              + (staleWorst > 30 ? ` · oldest ${staleWorst.toFixed(0)} s ago` : "")
              + ` · dpos = decoded-vs-BRDC position residual</div>`;
        html += this._renderPvt(data.pvt);
        this.root.innerHTML = html;
    }

    // Position self-survey (gnss_pvt): a from-the-code single-point position, per
    // (constellation, band) and a combined best-fit, reported as the OFFSET from the surveyed
    // config position with 1-sigma error bars. Two best-fits: the DUAL-FREQUENCY iono-free
    // (combined_if, few-metre) headlined, and the single-frequency (combined, iono-limited
    // ~10 m class) beneath it for comparison. The per-group horizontal/RMS (incl. the "-IF" rows)
    // doubles as a per-signal code-quality readout.
    _renderPvt(pvt) {
        if (!pvt) return "";
        if (pvt.error)
            return `<div style="color:#8a8f98;padding:6px;font-size:11px">PVT: ${pvt.error}</div>`;
        const hyp = (a, b) => Math.sqrt(a * a + b * b);
        let h = `<div style="border-top:1px solid #333;margin-top:4px;padding:6px 6px 2px">`
              + `<span style="color:#cbd5e1;font-weight:600">Position self-survey</span>`
              + `<span style="color:#8a8f98;font-size:11px"> · offset vs surveyed site</span></div>`;
        // best-fit line: bold label, horiz±/vert±, sats, PDOP, and the solved lat/lon/alt.
        const fitLine = (c, label, labelColor) => {
            const ho = hyp(c.d_e, c.d_n), sh = hyp(c.sigma_e, c.sigma_n);
            return `<div style="padding:2px 8px;font-size:12px">`
               + `<b style="color:${labelColor}">${label}:</b> ${ho.toFixed(1)} m horiz `
               + `<span style="color:#8a8f98">(±${sh.toFixed(0)})</span>, `
               + `${c.d_u >= 0 ? "+" : ""}${c.d_u.toFixed(1)} m vert `
               + `<span style="color:#8a8f98">(±${c.sigma_u.toFixed(0)})</span> · `
               + `${c.n_sats} sats${c.n_rejected ? ` (−${c.n_rejected})` : ""} · `
               + `PDOP ${c.pdop.toFixed(1)}<br>`
               + `<span style="color:#8a8f98">${c.lat.toFixed(6)}, ${c.lon.toFixed(6)}, `
               + `${c.alt.toFixed(0)} m</span></div>`;
        };
        if (pvt.combined_if) h += fitLine(pvt.combined_if, "iono-free", "#7dd3a8");
        if (pvt.combined) h += fitLine(pvt.combined, pvt.combined_if ? "single-freq" : "best-fit",
                                       "#cbd5e1");
        const groups = pvt.groups || {};
        const keys = Object.keys(groups).sort();
        if (keys.length) {
            h += `<table style="border-collapse:collapse;width:100%;font-size:11px;`
               + `margin-top:2px"><thead><tr style="color:#8a8f98;text-align:left">`
               + `<th style="padding:1px 6px">per signal</th><th>sats</th>`
               + `<th>horiz</th><th>vert</th><th>RMS</th></tr></thead><tbody>`;
            for (const k of keys) {
                const g = groups[k];
                const ho = hyp(g.d_e, g.d_n);
                h += `<tr><td style="padding:1px 6px">${k}</td>`
                   + `<td>${g.n_sats}${g.n_rejected ? `(−${g.n_rejected})` : ""}</td>`
                   + `<td>${ho.toFixed(0)} ±${hyp(g.sigma_e, g.sigma_n).toFixed(0)} m</td>`
                   + `<td>${g.d_u >= 0 ? "+" : ""}${g.d_u.toFixed(0)} m</td>`
                   + `<td>${g.resid_rms_m.toFixed(0)} m</td></tr>`;
            }
            h += `</tbody></table>`;
        }
        return h;
    }
}
