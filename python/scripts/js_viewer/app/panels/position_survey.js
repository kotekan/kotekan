// Position self-survey panel: a from-the-code single-point position (gnss_pvt), reported as the
// OFFSET from the surveyed config site with 1-sigma error bars.
//
// Split out of the nav-decode-health panel so the position solution stands on its own. Two
// best-fits: the DUAL-FREQUENCY iono-free solution (combined_if -- few-metre, removes the
// elevation-dependent ionospheric bias) headlined, and the single-frequency one (combined --
// iono-limited ~10 m class) beneath it for comparison. The per-group table (incl. the "-IF"
// iono-free rows) doubles as a per-signal code-quality readout.
//
// Data is the viewer server's /decode_health endpoint (its `pvt` block, computed by gnss_pvt and
// cached server-side). Same origin as the viewer -> no CORS; independent of the kotekan feed.

function hyp(a, b) { return Math.sqrt(a * a + b * b); }

export class PositionSurveyPanel {
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
            if (this.root) this.root.innerHTML =
                `<div style="color:#8a8f98;padding:8px">position survey unavailable: ${e}</div>`;
            return;
        }
        this._render((data && data.pvt) || null);
    }

    // best-fit line: bold label, horiz±/vert±, sats, PDOP, and the solved lat/lon/alt.
    _fitLine(c, label, labelColor) {
        const ho = hyp(c.d_e, c.d_n), sh = hyp(c.sigma_e, c.sigma_n);
        return `<div style="padding:3px 8px;font-size:12px">`
           + `<b style="color:${labelColor}">${label}:</b> ${ho.toFixed(1)} m horiz `
           + `<span style="color:#8a8f98">(±${sh.toFixed(0)})</span>, `
           + `${c.d_u >= 0 ? "+" : ""}${c.d_u.toFixed(1)} m vert `
           + `<span style="color:#8a8f98">(±${c.sigma_u.toFixed(0)})</span> · `
           + `${c.n_sats} sats${c.n_rejected ? ` (−${c.n_rejected})` : ""} · `
           + `PDOP ${c.pdop.toFixed(1)}<br>`
           + `<span style="color:#8a8f98">${c.lat.toFixed(6)}, ${c.lon.toFixed(6)}, `
           + `${c.alt.toFixed(0)} m</span></div>`;
    }

    _render(pvt) {
        if (!this.root) return;
        if (!pvt) {
            this.root.innerHTML = `<div style="color:#8a8f98;padding:8px;font-size:11px">`
                + `no position solution yet (needs code observations across satellites)</div>`;
            return;
        }
        if (pvt.error) {
            this.root.innerHTML =
                `<div style="color:#8a8f98;padding:8px">position survey: ${pvt.error}</div>`;
            return;
        }
        let h = `<div style="padding:6px 6px 2px">`
              + `<span style="color:#8a8f98;font-size:11px">offset vs surveyed site`
              + (pvt.n_meas != null ? ` · ${pvt.n_meas} obs` : "") + `</span></div>`;
        if (pvt.combined_if) h += this._fitLine(pvt.combined_if, "iono-free", "#7dd3a8");
        if (pvt.combined)
            h += this._fitLine(pvt.combined, pvt.combined_if ? "single-freq" : "best-fit", "#cbd5e1");
        if (!pvt.combined && !pvt.combined_if)
            h += `<div style="color:#8a8f98;padding:4px 8px;font-size:11px">`
               + `no combined fit (too few satellites / ill-conditioned geometry)</div>`;

        const groups = pvt.groups || {};
        const keys = Object.keys(groups).sort();
        if (keys.length) {
            h += `<table style="border-collapse:collapse;width:100%;font-size:11px;`
               + `margin-top:4px"><thead><tr style="color:#8a8f98;text-align:left`
               + `;border-bottom:1px solid #333">`
               + `<th style="padding:1px 6px" title="constellation-band group; -IF = dual-frequency `
               + `iono-free">per signal</th><th>sats</th>`
               + `<th>horiz</th><th>vert</th><th>RMS</th></tr></thead><tbody>`;
            for (const k of keys) {
                const g = groups[k];
                const ho = hyp(g.d_e, g.d_n);
                const isIf = k.endsWith("-IF");
                h += `<tr><td style="padding:1px 6px${isIf ? ";color:#7dd3a8" : ""}">${k}</td>`
                   + `<td>${g.n_sats}${g.n_rejected ? `(−${g.n_rejected})` : ""}</td>`
                   + `<td>${ho.toFixed(0)} ±${hyp(g.sigma_e, g.sigma_n).toFixed(0)} m</td>`
                   + `<td>${g.d_u >= 0 ? "+" : ""}${g.d_u.toFixed(0)} m</td>`
                   + `<td>${g.resid_rms_m.toFixed(0)} m</td></tr>`;
            }
            h += `</tbody></table>`;
        }
        h += `<div style="color:#8a8f98;padding:4px 6px;font-size:11px">`
           + `iono-free = dual-frequency (L1+L5) combination; -IF rows remove the ionosphere</div>`;
        this.root.innerHTML = h;
    }
}
