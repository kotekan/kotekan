// Nav-decode health panel: what the brokers' nav decoders are doing, per civil data signal.
//
// Data is the viewer server's /decode_health endpoint, which merges the per-broker
// decode_health/*.json files (python/scripts/gnss/decode_health.py). One row per decoded
// signal, grouped GPS -> Galileo -> BeiDou; each shows how many satellites are synced, how
// many yielded a full ephemeris, the BRDC-agreement residual (dpos, the "is our decode
// trustworthy" number), and how fresh the newest decode is. A signal whose broker file goes
// stale drops out of the endpoint, so its row disappears rather than freezing at its last
// value.
//
// Self-contained: fetches its own endpoint on a timer (same origin as the viewer, so no
// CORS), independent of the GpsFeed the sky and table use.

const CONSTS = [
    {tag: "G", name: "GPS"},
    {tag: "E", name: "Galileo"},
    {tag: "C", name: "BeiDou"},
];
// Friendly signal names keyed on the decode_health signal id.
const SIG_NAME = {
    GPS_L1_LNAV: "L1 C/A · LNAV", GPS_L1CD_CNAV2: "L1C-D · CNAV-2", GPS_L2C_CNAV: "L2C · CNAV",
    GPS_L5_CNAV: "L5-I · CNAV",
    GAL_E1B_INAV: "E1-B · I/NAV", GAL_E5AI_FNAV: "E5a-I · F/NAV", GAL_E5BI_INAV: "E5b-I · I/NAV",
    BDS_B1C_BCNAV1: "B1C-D · B-CNAV1", BDS_B2A_BCNAV2: "B2a-D · B-CNAV2",
    BDS_B2B_BCNAV3: "B2b-I · B-CNAV3",
};

function dot(color, title) {
    return `<span title="${title || ""}" style="display:inline-block;width:9px;height:9px;`
         + `border-radius:50%;background:${color};vertical-align:middle"></span>`;
}

// The freshest decode across a signal's satellites is the liveness number: a set or
// false-locked straggler must not make an actively decoding chain look dead, so the headline
// is min(last_s), not the worst satellite. nStale (last_s > 300 s) counts the stragglers.
function freshest(sig) {
    const v = (sig.sats || []).map(s => s.last_s).filter(x => x != null);
    return v.length ? Math.min(...v) : null;
}
function nStale(sig) {
    return (sig.sats || []).filter(s => s.last_s != null && s.last_s > 300).length;
}

// Overall per-signal status: green healthy, amber stale or BRDC-disagreeing, grey no sync.
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
                      + `<td>${s.band}</td>`
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
        this.root.innerHTML = html;
    }
}
