// "Save Data" button that snapshots the current waterfall buffer and
// triggers a browser download. Replaces the old server-side record dance
// (``addRecordButton``) which is gone with the consolidation; if you want
// server-side recording back, instantiate a different panel here.

function download(bytes, fname) {
    const blob = new Blob(bytes, {type: "application/octet-stream"});
    const link = document.createElement("a");
    link.href = window.URL.createObjectURL(blob);
    link.download = fname;
    link.click();
    window.URL.revokeObjectURL(link.href);
}

export class LocalRecordPanel {
    constructor({app, target}) {
        this.app = app;
        this.state = app.state;

        const self = this;
        const marg = 10;
        // Flex row: filename input flexes to fill remaining width, button
        // keeps its natural width. ``min-width: 0`` on the input lets it
        // shrink below its preferred width without overflowing the card,
        // and ``flex-wrap: nowrap`` keeps everything on one line.
        const wrapper = $("<div/>")
            .css({width: "100%", margin: marg + "px 0",
                  padding: `0 ${marg}px`, "box-sizing": "border-box",
                  display: "flex", "flex-wrap": "nowrap",
                  "align-items": "center", gap: "8px"})
            .appendTo($("#" + target));

        this._fn_idx = 0;
        this._record_fn = $("<input type='text'/>")
            .css({flex: "1 1 0", "min-width": "0", "font-size": "14pt"})
            .val("output" + ("000" + this._fn_idx).slice(-4) + ".dat")
            .appendTo(wrapper);

        $("<button/>").appendTo(wrapper)
            .css({"flex-shrink": "0"})
            .button({label: "Save Data", icons: {primary: "ui-icon-disk"}})
            .click(function() {
                const oldmode = self.state.mode;
                self.state.mode = "idle";
                const ccera = self.state.CCERA || {alt: NaN, az: NaN};
                const file_data = [].concat(
                    new Int32Array([self.state.num_freqs, self.state.scroll_data.length]),
                    new Float32Array([ccera.alt, ccera.az]),
                    new Float32Array(self.state.freq_list),
                    new Float64Array(self.state.timearr),
                    new Float32Array(self.state.spectrum),
                    new Float32Array(self.state.spectrum_baseline),
                    self.state.scroll_data);
                download(file_data, self._record_fn.val());
                self._fn_idx += 1;
                self._record_fn.val("output" + ("000" + self._fn_idx).slice(-4) + ".dat");
                self.state.mode = oldmode;
            });
    }
}
