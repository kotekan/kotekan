// Live GPS status panel: an overhead-sky circle of the visible PRNs (locked
// ones highlighted) plus a table of the locked PRNs with SNR and |A|.
//
// Three data sources, merged by PRN every ~1.5 s:
//   * /gps_sky                 (this viewer's server) -> {prn, az, el} positions
//   * <search>/get_detections  (kotekan REST)         -> acquired PRNs + SNR
//   * <combiner>/get_status    (kotekan REST)         -> per-PRN |A| / coh / deep
// plus <airspy>/adcstat for an overall ADC noise (RMS) readout.
//
// The sky is a standard GNSS skyplot: zenith at centre, horizon at the rim,
// North up, azimuth clockwise (so East is to the right). Positions need orbit
// propagation (skyfield, server-side); if that's unavailable the rings still
// draw and the locked-PRN table still works off the REST data alone.

const SVG_NS = "http://www.w3.org/2000/svg";

// Skyplot geometry, in the 0..100 viewBox.
const CX = 50, CY = 50, R = 44;
const EL_RINGS = [0, 30, 60];           // elevation circles to draw (deg)

const POLL_MS = 1500;
const AMP_LOCK = 0.30;                   // |A| fallback lock (only where deep_snr is unavailable)
const SNR_LOCK = 3.0;                    // detection significance (sigma above noise) for a lock

function svg(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
}

// Project (azimuth, elevation) in degrees to skyplot x/y (viewBox units).
function project(az_deg, el_deg) {
    const az = az_deg * Math.PI / 180;
    const rr = R * (90 - el_deg) / 90;   // 0 at zenith, R at the horizon
    return {x: CX + rr * Math.sin(az), y: CY - rr * Math.cos(az)};
}

// SNR -> colour ramp (red-orange at the acquire floor, green when strong).
function snr_color(snr) {
    if (snr == null) return "#8a8f98";
    const t = Math.max(0, Math.min(1, (snr - 6) / (22 - 6)));
    const hue = 8 + (130 - 8) * t;       // 8 = red-orange, 130 = green
    return `hsl(${hue.toFixed(0)}, 70%, 45%)`;
}

export class GpsSkyPanel {
    constructor({app, target, search_stage, combiner_stage, airspy_stage,
                 mask_deg, has_site}) {
        this.app = app;
        this.search_stage = search_stage || "search";
        this.combiner_stage = combiner_stage || "combiner";
        this.airspy_stage = airspy_stage || "airspy_in";
        this.mask_deg = mask_deg || 5;
        this.has_site = !!has_site;
        this._inflight = false;

        const root = $("#" + target);
        const wrap = $("<div/>").css({
            display: "flex", flexDirection: "row", gap: "8px",
            width: "100%", height: "100%", boxSizing: "border-box",
            fontFamily: "sans-serif",
        }).appendTo(root);

        // Left: the square sky circle.
        const skybox = $("<div/>").css({
            flex: "0 0 auto", height: "100%", aspectRatio: "1 / 1",
            position: "relative", minWidth: "0",
        }).appendTo(wrap);
        const s = svg("svg", {viewBox: "0 0 100 100",
                              preserveAspectRatio: "xMidYMid meet"});
        s.style.width = "100%";
        s.style.height = "100%";
        s.style.display = "block";
        skybox[0].appendChild(s);
        this._svg = s;
        this._draw_static();
        this._sat_layer = svg("g", {});   // sats redrawn each tick on top
        s.appendChild(this._sat_layer);

        // Right: status line + locked-PRN table.
        const side = $("<div/>").css({
            flex: "1 1 auto", minWidth: "0", overflow: "auto",
            fontSize: "12px",
        }).appendTo(wrap);
        this._status = $("<div/>").css({
            color: "#444", marginBottom: "6px", lineHeight: "1.5",
        }).appendTo(side);
        this._table = $("<div/>").appendTo(side);

        this._tick();
        this._timer = setInterval(() => this._tick(), POLL_MS);
    }

    // The fixed scaffold (rings, cardinal ticks, labels) -- drawn once.
    _draw_static() {
        const s = this._svg;
        s.appendChild(svg("circle", {cx: CX, cy: CY, r: R,
            fill: "#0b1a2b", stroke: "#9aa4af", "stroke-width": 0.5}));
        for (const el of EL_RINGS) {
            if (el === 0) continue;
            const rr = R * (90 - el) / 90;
            s.appendChild(svg("circle", {cx: CX, cy: CY, r: rr,
                fill: "none", stroke: "#33424f", "stroke-width": 0.3}));
            s.appendChild(svg("text", {x: CX + 0.8, y: CY - rr - 0.4,
                "font-size": 2.4, fill: "#6b7785"})).textContent = el + "°";
        }
        // Cardinal cross-hairs.
        s.appendChild(svg("line", {x1: CX, y1: CY - R, x2: CX, y2: CY + R,
            stroke: "#33424f", "stroke-width": 0.3}));
        s.appendChild(svg("line", {x1: CX - R, y1: CY, x2: CX + R, y2: CY,
            stroke: "#33424f", "stroke-width": 0.3}));
        const card = [["N", CX, CY - R - 1.2], ["S", CX, CY + R + 3.2],
                      ["E", CX + R + 2.6, CY + 1], ["W", CX - R - 2.6, CY + 1]];
        for (const [t, x, y] of card) {
            const lab = svg("text", {x, y, "font-size": 3.4, fill: "#aeb6bf",
                "text-anchor": "middle", "font-weight": "bold"});
            lab.textContent = t;
            s.appendChild(lab);
        }
    }

    _tick() {
        if (this._inflight) return;
        this._inflight = true;
        const k = this.app.kotekan;
        const jget = (p) => p.then(r => r.ok ? r.json() : null).catch(() => null);
        Promise.all([
            fetch("/gps_sky").then(r => r.ok ? r.json() : null).catch(() => null),
            jget(k.stageGet(this.search_stage, "get_detections")),
            jget(k.stageGet(this.combiner_stage, "get_status")),
            jget(k.stageGet(this.airspy_stage, "adcstat")),
        ]).then(([sky, det, status, adc]) => {
            this._inflight = false;
            // Hold the last good value per feed: a single slow/failed poll (e.g.
            // the combiner momentarily busy) must not blank every |A| for a
            // frame -- that would look like all sats dropping lock at once. A
            // real drop still shows, because the combiner then *reports* a low
            // amplitude rather than failing the request.
            const last = (this._last = this._last || {});
            if (sky) last.sky = sky;
            if (det) last.det = det;
            if (status) last.status = status;
            if (adc) last.adc = adc;
            this._render(last.sky, last.det, last.status, last.adc);
        });
    }

    // Merge the three feeds into one per-PRN map and repaint.
    _render(sky, det, status, adc) {
        const sats = new Map();   // prn -> merged record
        const get = (prn) => {
            if (!sats.has(prn)) sats.set(prn, {prn, az: null, el: null,
                snr: null, amp: 0, coh: 0, deep: 0, sig: 0, detected: false});
            return sats.get(prn);
        };
        if (sky && Array.isArray(sky.sats))
            for (const p of sky.sats) { const r = get(p.prn); r.az = p.az; r.el = p.el; }
        if (Array.isArray(det))
            for (const d of det) { const r = get(d.prn); r.snr = d.snr; r.detected = true; }
        if (Array.isArray(status))
            for (const c of status) {
                if (!c.prn) continue;
                const r = get(c.prn);
                r.amp = c.amplitude || 0;
                r.coh = c.coh_amplitude || 0;
                r.deep = c.deep_amplitude || 0;
                // significance = how many sigma the signal is above the noise: deep (nav-wiped)
                // when available, else the noise-debiased incoherent SNR. >>1 real, ~1 noise.
                r.sig = Math.max(c.deep_snr || 0, c.amp_snr || 0);
            }
        // "Locked" = a SIGNIFICANT detection (sig), not the raw |A| -- the incoherent |A| is biased
        // by the noise floor (≈ floor for weak sats), so |A| >= AMP_LOCK flickered phantoms across
        // the line. Fall back to |A| only where no significance is reported.
        const have_sig = [...sats.values()].some(r => r.sig > 0);
        for (const r of sats.values())
            r.active = have_sig ? (r.sig >= SNR_LOCK) : (r.detected || r.amp >= AMP_LOCK);

        this._draw_sats([...sats.values()]);
        this._draw_table([...sats.values()], sky, adc);
    }

    _draw_sats(list) {
        const layer = this._sat_layer;
        while (layer.firstChild) layer.removeChild(layer.firstChild);
        for (const r of list) {
            if (r.az == null || r.el == null) continue;   // no orbit fix -> table only
            const {x, y} = project(r.az, r.el);
            const rad = r.active ? 2.4 + Math.min(1.4, r.amp) : 1.9;
            const dot = svg("circle", {cx: x, cy: y, r: rad,
                fill: r.active ? snr_color(r.snr) : "none",
                stroke: r.active ? "#e8edf2" : "#7c8794",
                "stroke-width": r.active ? 0.4 : 0.5});
            const tip = svg("title", {});
            tip.textContent = `PRN ${r.prn}  el ${r.el.toFixed(0)}° az ${r.az.toFixed(0)}°`
                + (r.snr != null ? `  snr ${r.snr.toFixed(1)}` : "")
                + (r.amp ? `  |A| ${r.amp.toFixed(2)}` : "");
            dot.appendChild(tip);
            layer.appendChild(dot);
            const lab = svg("text", {x: x + rad + 0.6, y: y + 1,
                "font-size": 2.7, fill: r.active ? "#e8edf2" : "#8a929b"});
            lab.textContent = r.prn;
            layer.appendChild(lab);
        }
    }

    _draw_table(list, sky, adc) {
        const visible = list.filter(r => r.el != null).length;
        const locked = list.filter(r => r.active);
        let sky_state;
        if (!this.has_site) sky_state = "<b>no site</b> (set LAT/LON)";
        else if (sky && sky.ok) sky_state = "ok";
        else if (sky && sky.computing) sky_state = "loading…";
        else sky_state = sky && sky.error ? "unavailable" : "—";
        const rms = adc && adc.rms != null ? adc.rms.toFixed(0) : "—";
        const site = (sky && sky.lat != null)
            ? `${sky.lat.toFixed(3)}, ${sky.lon.toFixed(3)}` : "—";

        this._status.html(
            `<b>${locked.length}</b> locked / <b>${visible}</b> up&nbsp; · &nbsp;`
            + `sky: ${sky_state}<br>`
            + `site ${site}&nbsp; · &nbsp;ADC rms ${rms}&nbsp; · &nbsp;N up, E right`);

        // Locked PRNs first (by SNR, then |A|); show a few visible-but-unlocked
        // below so you can see what's overhead waiting to be acquired.
        locked.sort((a, b) =>
            (b.snr ?? -1) - (a.snr ?? -1) || (b.amp - a.amp));
        const idle = list.filter(r => !r.active && r.el != null)
            .sort((a, b) => b.el - a.el).slice(0, 6);

        const cell = (s, extra = "") => `<td style="padding:1px 6px;${extra}">${s}</td>`;
        const num = (v, d = 2) => (v ? v.toFixed(d) : "—");
        const row = (r, on) => {
            const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;`
                + `background:${on ? snr_color(r.snr) : "transparent"};`
                + `border:1px solid ${on ? "#e8edf2" : "#7c8794"};margin-right:5px;"></span>`;
            // sig = deep significance (deep / its uncertainty), the honest "is it real": >>1 lock,
            // ~1 noise. dB tooltip = 20·log10(sig). |A| is the noise-biased incoherent amplitude.
            const sig = r.sig
                ? `<span title="${(20 * Math.log10(r.sig)).toFixed(0)} dB">`
                  + r.sig.toFixed(1) + "σ</span>"
                : "—";
            return `<tr style="${on ? "" : "color:#8a929b;"}">`
                + cell(dot + "<b>" + r.prn + "</b>")
                + cell(r.el != null ? r.el.toFixed(0) + "°" : "—")
                + cell(r.snr != null ? r.snr.toFixed(1) : "—")
                + cell(num(r.amp), "text-align:right;color:#8a929b;")
                + cell(num(r.deep || r.coh, 3), "text-align:right;")
                + cell(sig, "text-align:right;")
                + "</tr>";
        };
        const head = "<tr style='color:#666;border-bottom:1px solid #ddd;'>"
            + "<th style='text-align:left;padding:1px 6px;'>PRN</th>"
            + "<th style='text-align:left;padding:1px 6px;'>El</th>"
            + "<th style='text-align:left;padding:1px 6px;'>SNR</th>"
            + "<th style='text-align:right;padding:1px 6px;' title='incoherent |A| (noise-biased)'>|A|</th>"
            + "<th style='text-align:right;padding:1px 6px;'>deep</th>"
            + "<th style='text-align:right;padding:1px 6px;' title='deep significance = deep / its uncertainty'>sig</th></tr>";
        let body = locked.map(r => row(r, true)).join("");
        if (!locked.length)
            body = "<tr><td colspan='6' style='padding:4px 6px;color:#999;'>"
                + "no locks yet…</td></tr>";
        if (idle.length)
            body += "<tr><td colspan='6' style='padding:4px 6px 1px;color:#aaa;"
                + "font-size:11px;'>overhead, unlocked</td></tr>"
                + idle.map(r => row(r, false)).join("");
        this._table.html(`<table style="border-collapse:collapse;width:100%;">`
            + head + body + "</table>");
    }
}
