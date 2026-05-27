// CCERA telescope pointing readouts. Polls ``ccera_rest.py``'s endpoints on
// port 3000 every few seconds and updates an info panel. Emits
// ``state:pointing_updated`` so other panels (e.g. the galaxy view) can
// react.

export class CCERAPointingPanel {
    constructor({app, target, host, port, pointing_interval_ms, state_interval_ms}) {
        this.app = app;
        this.bus = app.bus;
        this.state = app.state;
        this._host = host || location.hostname;
        this._port = port || 3000;
        const pointing_ms = pointing_interval_ms || 5000;
        const state_ms    = state_interval_ms    || 1000;

        this.state.CCERA = {lat: null, lon: null, el: null,
                            alt: null, az: null,
                            ra: null, dec: null, gl: null, gb: null};

        const wrapper = $("<div/>").uniqueId().appendTo($("#" + target))
            .css({position: "relative", float: "left", width: "100%"});
        const ccerawrap = $("<div/>").uniqueId().css({width: "100%"}).appendTo(wrapper)
            .css({position: "relative", float: "left"})
            .css({"font-family": "sans-serif", "font-size": "10pt"});

        $("<p>").text("Telescope Info").css({"font-size": "14pt", "text-align": "center"}).appendTo(ccerawrap);
        const lcol = $("<div/>").css({width: "33%", float: "left"}).appendTo(ccerawrap);
        const mcol = $("<div/>").css({width: "33%", float: "left"}).appendTo(ccerawrap);
        const rcol = $("<div/>").css({width: "33%", height: "75px", position: "relative", float: "left"}).appendTo(ccerawrap);
        const row  = (parent, lbl) => $("<div/>").css({width: "100%"}).text(lbl).appendTo(parent);
        const alt = row(lcol, "Alt: ");
        const az  = row(lcol, "Az: ");
        const lat = row(lcol, "Lat: ");
        const lon = row(lcol, "Lon: ");
        const el  = row(lcol, "Elev: ");
        const ra  = row(mcol, "RA: ");
        const dec = row(mcol, "Dec: ");
        const gl  = row(mcol, "Gal. Lon: ");
        const gb  = row(mcol, "Gal. Lat: ");
        const state_div = $("<div/>")
            .css({width: "100%", height: "100%",
                  display: "flex", "justify-content": "center", "align-items": "center",
                  "white-space": "pre-line",
                  "text-align": "center", "font-size": "12pt"})
            .text("STATE")
            .css({border: "2px solid black", "border-radius": "5px"})
            .appendTo(rcol);

        const base = "http://" + this._host + ":" + this._port;
        const self = this;
        const update_pointing = function() {
            fetch(base + "/position")
                .then(r => r.json().then(data => {
                    Object.assign(self.state.CCERA, {lat: data.lat, lon: data.lon, el: data.el});
                    lat.text("Lat: "  + data.lat.toFixed(2) + " deg");
                    lon.text("Lon: "  + data.lon.toFixed(2) + " deg");
                    el .text("Elev: " + data.el .toFixed(2) + " m");
                    fetch(base + "/pointing")
                        .then(r => r.json().then(data => {
                            Object.assign(self.state.CCERA, {
                                alt: data.alt, az: data.az,
                                ra: data.ra, dec: data.dec, gl: data.gl, gb: data.gb,
                            });
                            alt.text("Alt: "      + data.alt.toFixed(2) + " deg");
                            az .text("Az: "       + data.az .toFixed(2) + " deg");
                            ra .text("RA: "       + data.ra .toFixed(2) + " deg");
                            dec.text("Dec: "      + data.dec.toFixed(2) + " deg");
                            gl .text("Gal. Lon: " + data.gl .toFixed(2) + " deg");
                            gb .text("Gal. Lat: " + data.gb .toFixed(2) + " deg");
                            self.bus.emit("state:pointing_updated", {...self.state.CCERA});
                        }));
                }));
        };
        update_pointing();
        setInterval(update_pointing, pointing_ms);

        const update_state = function() {
            fetch(base + "/state")
                .then(r => r.json().then(data => {
                    let text = data.state;
                    if (text.startsWith("on source")) {
                        text = "On Source\n" + text.split(",")[1].substring(1);
                        state_div.css({backgroundColor: "#90EE90"});
                    } else if (text.startsWith("slewing")) {
                        text = "Slewing";
                        state_div.css({backgroundColor: "#FFDBBB"});
                    } else {
                        state_div.css({backgroundColor: "white"});
                    }
                    if (data.expiry) {
                        const time_left = (data.expiry - data.now).toFixed(0);
                        state_div.text(text + "\nDwell: " + time_left + " sec");
                    } else {
                        state_div.text(text);
                    }
                }));
        };
        update_state();
        setInterval(update_state, state_ms);
    }
}
