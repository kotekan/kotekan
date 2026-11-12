// Overhead-sky panel: a standard GNSS skyplot (zenith centre, horizon rim, North up, azimuth
// clockwise) of every visible satellite, its own resizable card. Data comes from the shared
// GpsFeed (gps_feed.js); the detections table is a separate panel (gps_table.js) fed by the
// same poller.
//
// The dot's stroke carries the constellation colour (from the server's legend); an active
// dot's fill is the search-SNR ramp (red-orange at the acquire floor -> green when strong), so
// identity and signal strength read independently. The legend chips (bottom-left) toggle each
// constellation, shared with the table. An arrowhead beside each moving dot shows its heading,
// from its own recent positions; hovering a dot draws its path over +-1 h (/gps_track).

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

// Heading arrows: the direction from a satellite's position ARROW_LAG_S ago (at least
// ARROW_MIN_S ago) to now, in plot coordinates; none below ARROW_MIN_MOVE viewBox units, so
// the SBAS GEOs, which stand still, get none.
const HIST_S = 180, ARROW_LAG_S = 60, ARROW_MIN_S = 20, ARROW_MIN_MOVE = 0.05;
const HOVER_R = 3;          // viewBox units from a dot that count as hovering it
const TRACK_TTL_MS = 60000; // re-fetch a hovered path after this

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
        this._track_layer = svg("g", {});   // the hovered satellite's path, under the dots
        s.appendChild(this._track_layer);
        this._sat_layer = svg("g", {});   // sats redrawn each tick on top
        s.appendChild(this._sat_layer);
        this._legend_layer = svg("g", {});
        s.appendChild(this._legend_layer);

        this._hist = new Map();     // id -> [{t, az, el}], the last HIST_S of positions
        this._drawn = [];           // [{id, tag, prn, x, y}] from the last render, for hover
        this._hover = null;         // id of the satellite under the pointer
        this._tracks = new Map();   // id -> {t, pts} from /gps_track (pts null: no path)
        this._last = null;          // the last feed update, to redraw on hover
        // The dots are rebuilt every tick, so hover is a hit test on the whole plot rather
        // than per-dot mouse events, which a rebuilt dot would miss.
        s.addEventListener("mousemove", e => this._on_move(e));
        s.addEventListener("mouseleave", () => this._set_hover(null));

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

    _render(d) {
        const {sats, vis, chains} = d;
        this._last = d;
        const now = Date.now() / 1000;
        const layer = this._sat_layer;
        while (layer.firstChild) layer.removeChild(layer.firstChild);
        this._drawn = [];
        for (const r of sats) {
            if (r.az == null || r.el == null) continue;   // no orbit fix -> table only
            this._record(r, now);
            if (!vis[r.tag]) continue;
            if (r.el < 0) continue;   // below the horizon (e.g. a noise probe) -> off the plot
            const cc = chain_color(r.tag);
            const {x, y} = project(r.az, r.el);
            this._drawn.push({id: r.id, tag: r.tag, prn: r.prn, x, y});
            // Small symbols so dense constellations do not overlap: active 1.6..2.5 by |A|,
            // idle 1.25. Colour by search SNR, else by sig: a chain with no search stage leaves
            // snr null on a strong satellite.
            const shade = r.snr != null ? r.snr : (r.sig || null);
            const rad = r.active ? 1.6 + Math.min(0.9, r.amp) : 1.25;
            const dot = svg("circle", {cx: x, cy: y, r: rad,
                fill: r.active ? snr_color(shade) : "none",
                stroke: r.active ? cc : cc + "66",   // idle ring = translucent
                "stroke-width": r.active ? 0.7 : 0.5});
            const tip = svg("title", {});
            tip.textContent = `${r.id}  el ${r.el.toFixed(0)}° az ${r.az.toFixed(0)}°`
                + (r.snr != null ? `  snr ${r.snr.toFixed(1)}` : "")
                + (r.cn0 != null ? `  C/N0 ${r.cn0.toFixed(1)} dB-Hz` : "")
                + (r.sig ? `  sig ${r.sig.toFixed(0)}σ` : "");
            dot.appendChild(tip);
            layer.appendChild(dot);
            const u = this._heading(r.id, x, y, now);
            if (u) {
                // A small arrowhead just outside the dot, pointing along the motion.
                const tip_d = rad + 1.5, base_d = rad + 0.4, w = 0.7;
                const bx = x + u.x * base_d, by = y + u.y * base_d;
                const pts = [[x + u.x * tip_d, y + u.y * tip_d],
                             [bx - u.y * w, by + u.x * w], [bx + u.y * w, by - u.x * w]];
                layer.appendChild(svg("polygon", {
                    points: pts.map(p => p[0].toFixed(2) + "," + p[1].toFixed(2)).join(" "),
                    fill: r.active ? cc : cc + "66"}));
            }
            const lab = svg("text", {x: x + rad + 0.5, y: y + 0.8,
                "font-size": 2.2,
                fill: r.active ? "#e8edf2" : "#8a929b"});
            lab.textContent = r.id;
            layer.appendChild(lab);
        }
        for (const [id, h] of this._hist)
            if (now - h[h.length - 1].t > HIST_S) this._hist.delete(id);
        this._draw_track(vis);
        this._draw_legend(vis, chains);
    }

    // Keep the last HIST_S of each satellite's positions; a repeat of the last one is dropped.
    _record(r, now) {
        let h = this._hist.get(r.id);
        if (!h) this._hist.set(r.id, h = []);
        const last = h[h.length - 1];
        if (last && last.az === r.az && last.el === r.el) return;
        h.push({t: now, az: r.az, el: r.el});
        while (now - h[0].t > HIST_S) h.shift();
    }

    // Unit direction of motion on the plot, from the position ARROW_LAG_S ago (or the oldest
    // one, if at least ARROW_MIN_S old) to (x, y); null while there is too little history or
    // the satellite has not moved.
    _heading(id, x, y, now) {
        const h = this._hist.get(id);
        if (!h) return null;
        let ref = null;
        for (const e of h) {
            if (now - e.t < ARROW_MIN_S) break;
            ref = e;
            if (now - e.t <= ARROW_LAG_S) break;
        }
        if (!ref) return null;
        const p = project(ref.az, ref.el);
        const dx = x - p.x, dy = y - p.y, d = Math.hypot(dx, dy);
        return d < ARROW_MIN_MOVE ? null : {x: dx / d, y: dy / d};
    }

    _on_move(e) {
        const ctm = this._svg.getScreenCTM();
        if (!ctm) return;
        const pt = this._svg.createSVGPoint();
        pt.x = e.clientX;
        pt.y = e.clientY;
        const p = pt.matrixTransform(ctm.inverse());
        let best = null, bd = HOVER_R;
        for (const d of this._drawn) {
            const dd = Math.hypot(d.x - p.x, d.y - p.y);
            if (dd < bd) { bd = dd; best = d; }
        }
        this._set_hover(best);
    }

    _set_hover(d) {
        const id = d ? d.id : null;
        if (id === this._hover) return;
        this._hover = id;
        if (d) this._fetch_track(d);
        if (this._last) this._draw_track(this._last.vis);
    }

    _fetch_track(d) {
        const c = this._tracks.get(d.id);
        if (c && (c.pending || Date.now() - c.t < TRACK_TTL_MS)) return;
        this._tracks.set(d.id, {t: Date.now(), pts: c ? c.pts : null, pending: true});
        fetch(`/gps_track?tag=${encodeURIComponent(d.tag)}&prn=${d.prn}`)
            .then(r => r.ok ? r.json() : null).catch(() => null)
            .then(j => {
                this._tracks.set(d.id, {t: Date.now(), pts: j && j.ok ? j.points : null,
                                        pending: false});
                if (this._hover === d.id && this._last) this._draw_track(this._last.vis);
            });
    }

    // The hovered satellite's path over +-1 h (/gps_track): the past dashed, the future solid
    // with a tick every 10 min, both cut at the horizon.
    _draw_track(vis) {
        const layer = this._track_layer;
        while (layer.firstChild) layer.removeChild(layer.firstChild);
        const d = this._drawn.find(x => x.id === this._hover);
        const c = d ? this._tracks.get(d.id) : null;
        if (!d || !vis[d.tag] || !c || !c.pts) return;
        const cc = chain_color(d.tag);
        let run = [];
        const flush = future => {
            if (run.length > 1)
                layer.appendChild(svg("polyline", {
                    points: run.map(p => p.x.toFixed(2) + "," + p.y.toFixed(2)).join(" "),
                    fill: "none", stroke: cc, "stroke-width": 0.45,
                    "stroke-dasharray": future ? "none" : "0.8 0.6",
                    opacity: future ? 0.9 : 0.55}));
            run = [];
        };
        let future = false;
        for (const [dt, az, el] of c.pts) {
            if (!future && dt >= 0) {
                // The dt = 0 point ends the past run and starts the future one.
                const p = el >= 0 ? project(az, el) : null;
                if (p) run.push(p);
                flush(false);
                future = true;
                if (p) run.push(p);
                continue;
            }
            if (el < 0) { flush(future); continue; }
            const p = project(az, el);
            run.push(p);
            if (future && dt > 0 && dt % 10 === 0)
                layer.appendChild(svg("circle", {cx: p.x, cy: p.y, r: 0.35, fill: cc}));
        }
        flush(future);
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
