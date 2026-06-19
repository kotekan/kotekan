// Big Stop/Start button that pauses ingestion of new frames. Pause is local
// only (we keep the WebSocket open; the WaterfallView ignores frames when
// mode is "stopped").

export class StartStopPanel {
    constructor({app, target}) {
        this.app = app;
        this.state = app.state;
        this.bus = app.bus;

        const self = this;
        const stopped = (this.state.mode === "stopped");
        // Centered, slim layout: just enough padding to keep the button off
        // the card edges without leaving a wide empty gap.
        const wrapper = $("<div/>").appendTo($("#" + target))
            .css({padding: "8px", display: "flex", "justify-content": "center"});
        this._btn = $("<button/>").appendTo(wrapper)
            .button({
                label: stopped ? "Start" : "Stop",
                icons: {primary: stopped ? "ui-icon-play" : "ui-icon-stop"},
            })
            .css({border: "1px solid"})
            .click(function() {
                if (self.state.mode === "stopped") {
                    $(this).button("option", {label: "Stop", icons: {primary: "ui-icon-stop"}})
                        .css({border: "3px solid green"});
                    self.state.mode = "normal";
                    self.state.scroll_data = [];
                    self.app.socket.connect();
                    self.bus.emit("state:mode_changed", {mode: "normal"});
                } else {
                    $(this).button("option", {label: "Start", icons: {primary: "ui-icon-play"}})
                        .css({border: "1px solid"});
                    self.state.mode = "stopped";
                    self.app.socket.close();
                    self.bus.emit("state:mode_changed", {mode: "stopped"});
                }
            });
    }
}
