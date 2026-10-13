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
