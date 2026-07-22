// Overhead-sky panel: a standard GNSS skyplot (zenith centre, horizon rim,
// North up, azimuth clockwise) of every visible satellite, its own resizable
// card. Data comes from the shared GpsFeed (see gps_feed.js); the detections
// table is a separate panel (gps_table.js) fed by the same poller.
//
// Constellation styling: dot STROKE + label carry the constellation colour
// (G blue / E orange / C red, matching the composite-map palette); the FILL
// stays the search-SNR ramp (red-orange at the acquire floor -> green when
// strong), so identity and signal strength read independently. The legend
// chips (bottom-left) toggle each constellation on/off, shared with the table.

import {chain_color} from "./gps_feed.js";

const SVG_NS = "http://www.w3.org/2000/svg";

// Skyplot geometry, in the 0..100 viewBox.
const CX = 50, CY = 50, R = 44;
const EL_RINGS = [0, 30, 60];           // elevation circles to draw (deg)

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
export function snr_color(snr) {
    if (snr == null) return "#8a8f98";
    const t = Math.max(0, Math.min(1, (snr - 6) / (22 - 6)));
    const hue = 8 + (130 - 8) * t;       // 8 = red-orange, 130 = green
    return `hsl(${hue.toFixed(0)}, 70%, 45%)`;
}

export class GpsSkyPanel {
    constructor({target, feed}) {
        this.feed = feed;

        const root = $("#" + target);
        const box = $("<div/>").css({
            width: "100%", height: "100%", position: "relative",
            display: "flex", alignItems: "center", justifyContent: "center",
        }).appendTo(root);
        const s = svg("svg", {viewBox: "0 0 100 100",
                              preserveAspectRatio: "xMidYMid meet"});
        s.style.width = "100%";
        s.style.height = "100%";
        s.style.display = "block";
        box[0].appendChild(s);
        this._svg = s;
        this._draw_static();
        this._sat_layer = svg("g", {});   // sats redrawn each tick on top
        s.appendChild(this._sat_layer);
        this._legend_layer = svg("g", {});
        s.appendChild(this._legend_layer);

        this.feed.on(d => this._render(d));
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

    _render({sats, vis, chains}) {
        const layer = this._sat_layer;
        while (layer.firstChild) layer.removeChild(layer.firstChild);
        for (const r of sats) {
            if (!vis[r.tag]) continue;
            if (r.az == null || r.el == null) continue;   // no orbit fix -> table only
            if (r.el < 0) continue;   // below the horizon (e.g. a noise probe) -> off the plot
            const cc = chain_color(r.tag);
            const {x, y} = project(r.az, r.el);
            // Compact symbols (user request 2026-07-12): ~35% smaller than v1 so dense
            // constellations don't overlap -- active 1.6..2.5 (was 2.4..3.8), idle 1.25.
            const rad = r.active ? 1.6 + Math.min(0.9, r.amp) : 1.25;
            const dot = svg("circle", {cx: x, cy: y, r: rad,
                fill: r.active ? snr_color(r.snr) : "none",
                stroke: r.active ? cc : cc + "66",   // idle ring = translucent
                "stroke-width": r.active ? 0.7 : 0.5});
            const tip = svg("title", {});
            tip.textContent = `${r.id}  el ${r.el.toFixed(0)}° az ${r.az.toFixed(0)}°`
                + (r.snr != null ? `  snr ${r.snr.toFixed(1)}` : "")
                + (r.cn0 != null ? `  C/N0 ${r.cn0.toFixed(1)} dB-Hz` : "")
                + (r.sig ? `  sig ${r.sig.toFixed(0)}σ` : "");
            dot.appendChild(tip);
            layer.appendChild(dot);
            const lab = svg("text", {x: x + rad + 0.5, y: y + 0.8,
                "font-size": 2.2,
                fill: r.active ? "#e8edf2" : "#8a929b"});
            lab.textContent = r.id;
            layer.appendChild(lab);
        }
        this._draw_legend(vis, chains);
    }

    // Clickable constellation chips, bottom-left corner of the sky square.
    _draw_legend(vis, chains) {
        const layer = this._legend_layer;
        while (layer.firstChild) layer.removeChild(layer.firstChild);
        chains.forEach((c, i) => {
            const y = 100 - 3 - (chains.length - 1 - i) * 4.4;
            const g = svg("g", {cursor: "pointer"});
            const on = vis[c.tag];
            g.appendChild(svg("circle", {cx: 3, cy: y - 0.9, r: 1.5,
                fill: on ? c.color : "none", stroke: c.color,
                "stroke-width": 0.4, opacity: on ? 1 : 0.45}));
            const t = svg("text", {x: 5.4, y, "font-size": 2.8,
                fill: on ? "#3c4650" : "#9aa4af"});
            t.textContent = `${c.tag} ${c.name}` + (on ? "" : " (hidden)");
            g.appendChild(t);
            const tip = svg("title", {});
            tip.textContent = `click to ${on ? "hide" : "show"} ${c.name}`;
            g.appendChild(tip);
            g.addEventListener("click", () => this.feed.set_vis(c.tag, !on));
            layer.appendChild(g);
        });
    }
}
