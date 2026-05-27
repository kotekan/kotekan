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
        return `${this.base}/${stage}/${endpoint}`;
    }

    stageGet(stage, endpoint) {
        return fetch(this.stageUrl(stage, endpoint));
    }

    stagePost(stage, endpoint, body) {
        return fetch(this.stageUrl(stage, endpoint), {
            mode: "no-cors",
            method: "POST",
            headers: {"Accept": "application/json", "Content-Type": "application/json"},
            body: JSON.stringify(body),
        });
    }
}
