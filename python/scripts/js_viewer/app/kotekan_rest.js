// Thin adapter for kotekan's REST endpoints. Panels build URLs via
// ``rest.stageUrl(stage, "set_config")`` instead of concatenating
// ``http://${host}:${port}/${stage}/${endpoint}`` everywhere.
export class KotekanRest {
    constructor({host, port, airspy_stages, lag_align_stage}) {
        this.host = host;
        this.port = port;
        this.airspy_stages = airspy_stages || [];
        this.lag_align_stage = lag_align_stage || null;
    }

    get base() {
        return `http://${this.host}:${this.port}`;
    }

    stageUrl(stage, endpoint) {
        return `${this.base}/${this.resolveStage(stage)}/${endpoint}`;
    }

    // gps_* <-> bare stage-name aliasing (the JS twin of python/scripts/gnss_stages.py).
    // The tri-constellation config names the GPS chain gps_search / gps_track /
    // gps_combiner, matching gal_*/bds_*; the older single-constellation benches still
    // use search / track / combiner. Panels always ask for the gps_* spelling and this
    // maps it onto whatever the running pipeline actually registered. loadStages() fills
    // the table once at startup; until it lands (or if /endpoints fails) names pass
    // through unchanged, which is the correct behaviour for every non-GPS stage.
    loadStages() {
        return fetch(`${this.base}/endpoints`)
            .then(r => (r.ok ? r.json() : null))
            .then(eps => {
                if (!eps) return;
                this._stages = new Set();
                for (const method of ["GET", "POST"])
                    for (const path of eps[method] || []) {
                        const parts = path.replace(/^\//, "").split("/");
                        if (parts.length >= 2) this._stages.add(parts[0]);
                    }
            })
            .catch(() => { /* pipeline not up: pass names through */ });
    }

    resolveStage(stage) {
        if (!this._stages || this._stages.has(stage)) return stage;
        const alt = stage.startsWith("gps_") ? stage.slice(4) : `gps_${stage}`;
        return this._stages.has(alt) ? alt : stage;
    }

    stageGet(stage, endpoint) {
        return fetch(this.stageUrl(stage, endpoint));
    }

    stagePost(stage, endpoint, body) {
        // Real CORS request (not ``no-cors``) so the reply is readable and
        // failures surface. Only Content-Type is set -- adding Accept would put
        // ``accept`` in the preflight, which kotekan's allowed CORS headers
        // (x-prototype-version, x-requested-with, content-type) don't include.
        return fetch(this.stageUrl(stage, endpoint), {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(body),
        }).then((res) => {
            if (!res.ok)
                console.warn(`kotekan POST ${stage}/${endpoint} -> HTTP ${res.status}`);
            return res;
        });
    }
}
