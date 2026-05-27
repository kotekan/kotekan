// All-sky background image with a "looking at" arrow that rotates with the
// telescope's galactic longitude. Listens for ``state:pointing_updated``
// events from CCERAPointingPanel.

const SVG_NS = "http://www.w3.org/2000/svg";

export class GalaxyViewPanel {
    constructor({app, target, image_url}) {
        this.app = app;
        this.bus = app.bus;

        const wrapper = $("<div/>").appendTo($("#" + target))
            .width("100%")
            .css({position: "relative", float: "left"});

        const img = $("<img/>").appendTo(wrapper)
            .attr("src", image_url)
            .attr({width: "100%"})
            .css({filter: "invert(100%)", display: "block"});

        const sun_frac_loc = [0.5, 1 - 0.309];

        // Overlay SVG that sits on top of the image; scales with it via
        // ``width: 100%`` and a viewBox keyed to the fractional coords we
        // use for the sun position.
        const svg = document.createElementNS(SVG_NS, "svg");
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.setAttribute("preserveAspectRatio", "none");
        Object.assign(svg.style, {
            position: "absolute", left: "0", top: "0",
            width: "100%", height: "100%",
            "z-index": "100", "pointer-events": "none",
        });

        // Arrow head (a marker), so the line auto-terminates with one.
        const defs = document.createElementNS(SVG_NS, "defs");
        const marker = document.createElementNS(SVG_NS, "marker");
        marker.setAttribute("id", "galaxy-arrow-head");
        marker.setAttribute("viewBox", "0 0 10 10");
        marker.setAttribute("refX", "8");
        marker.setAttribute("refY", "5");
        marker.setAttribute("markerWidth", "6");
        marker.setAttribute("markerHeight", "6");
        marker.setAttribute("orient", "auto-start-reverse");
        const arrow_path = document.createElementNS(SVG_NS, "path");
        arrow_path.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
        arrow_path.setAttribute("fill", "#000");
        marker.appendChild(arrow_path);
        defs.appendChild(marker);
        svg.appendChild(defs);

        const line = document.createElementNS(SVG_NS, "line");
        line.setAttribute("x1", sun_frac_loc[0] * 100);
        line.setAttribute("y1", sun_frac_loc[1] * 100);
        line.setAttribute("x2", sun_frac_loc[0] * 100);
        line.setAttribute("y2", sun_frac_loc[1] * 100);
        line.setAttribute("stroke", "#000");
        line.setAttribute("stroke-width", "0.5");
        line.setAttribute("vector-effect", "non-scaling-stroke");
        line.setAttribute("marker-end", "url(#galaxy-arrow-head)");
        svg.appendChild(line);
        wrapper[0].appendChild(svg);

        this.bus.on("state:pointing_updated", ({gl}) => {
            if (gl == null) return;
            // Sun lives at the local (sun_frac_loc) anchor in viewBox space
            // (0..100). The arrow points outwards along the galactic
            // longitude direction, scaled to half the viewBox extent.
            const dx = sun_frac_loc[0] * 100 - 50 * Math.sin(gl / 360 * 2 * Math.PI);
            const dy = sun_frac_loc[1] * 100 - 50 * Math.cos(gl / 360 * 2 * Math.PI);
            line.setAttribute("x2", dx);
            line.setAttribute("y2", dy);
            line.setAttribute("stroke-width", "1");
        });
    }
}
