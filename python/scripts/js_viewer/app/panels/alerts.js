// Alerts panel: the stack's ALERT files and the jobs whose silence matters.
//
// Data is the viewer server's /alerts endpoint (livebeam_server.AlertsResource). A heartbeat
// is a file a job rewrites while it works; it turns red once older than its limit, since a job
// that has stopped writes no ALERT. Each ALERT file is shown with its age and first lines.
//
// Self-contained: fetches its own endpoint on a timer, independent of the GpsFeed.

function esc(s) {
    return String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
}

function age(s) {
    if (s == null) return "missing";
    if (s < 120) return `${s.toFixed(0)} s`;
    if (s < 7200) return `${(s / 60).toFixed(0)} min`;
    if (s < 172800) return `${(s / 3600).toFixed(1)} h`;
    return `${(s / 86400).toFixed(1)} d`;
}

function dot(color, title) {
    return `<span title="${esc(title)}" style="display:inline-block;width:9px;height:9px;`
         + `border-radius:50%;background:${color};vertical-align:middle"></span>`;
}

export class AlertsPanel {
    constructor({target, poll_ms}) {
        this.root = document.getElementById(target);
        this.poll_ms = poll_ms || 30000;
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
            const r = await fetch("alerts", {cache: "no-store"});
            data = await r.json();
        } catch (e) {
            if (this.root) this.root.innerHTML =
                `<div style="color:#8a8f98;padding:8px">alerts unavailable: ${esc(e)}</div>`;
            return;
        }
        this._render(data);
    }

    _render(data) {
        if (!this.root) return;
        const beats = (data && data.heartbeats) || [];
        const alerts = (data && data.alerts) || [];
        let html = `<div style="padding:4px 6px">`;
        for (const b of beats) {
            const title = `${b.path}: limit ${age(b.max_age_s)}`;
            html += `<span style="margin-right:14px">${dot(b.stale ? "#e5534b" : "#2ea043", title)} `
                  + `${esc(b.name)} <span style="color:#8a8f98">${age(b.age_s)}</span></span>`;
        }
        html += `</div>`;
        if (!alerts.length)
            html += `<div style="color:#8a8f98;padding:4px 6px">no ALERT files</div>`;
        for (const a of alerts) {
            html += `<div style="padding:4px 6px;border-top:1px solid #222">`
                  + `<span style="color:#d0a215;font-weight:600">${esc(a.name)}</span> `
                  + `<span style="color:#8a8f98" title="${esc(a.path)}">${age(a.age_s)} ago</span>`
                  + `<pre style="margin:2px 0 0;white-space:pre-wrap;color:#cbd5e1">`
                  + `${a.lines.map(esc).join("\n")}</pre></div>`;
        }
        this.root.innerHTML = html;
    }
}
