#!/usr/bin/env python3
"""A synthetic fleet for the broker's equivalence gate (broker_equiv.py).

The gate needs a transcript. The best come from real chains, but none is available with the
F-engine down, so this serves a deterministic, self-contained fleet from which a transcript
can always be captured:

    broker_fakefleet.py --port 12777 &
    broker_equiv.py record /tmp/t.jsonl -- \
        --detectors http://localhost:12777/gps_search \
        --trackers http://localhost:12777/track \
        --combiner http://localhost:12777/combiner \
        --time0-endpoint telescope/time0_ns --rest-url http://localhost:12777 \
        --carrier-hz 1176.45e6 --chip-rate-hz 10.23e6 --code-length 10230 \
        --hops-per-sec 195312.5 --nh-overlay-len 20 --interval 0.05 --acquire-snr 12

It is not a simulator and makes no claim to be physical. It drives code paths: detections
that advance (so cp_hist fills, the rate fit runs, freshness stamps move), a status table
that reports locks (so the hold/watchdog/DLL/carrier paths execute), and a time anchor (so
the anchor-dependent arithmetic runs). It must never be static: a frozen fleet replays
perfectly and tests nothing.

With --chains it also serves a node's chain instances as the generator names them: on each GPU
half g of --halves, each chain c is a row gnss<g>_<c> with the per-chain paths
/gnss<g>_<c>_{inject,n2assemble,n2combine}/<verb>. With --hosted, the halves' hosts of those
stages (gnss<g>_inject, gnss<g>_records, gnss<g>_combine) serve the same verbs by chain, as
gnss::rest_register_rows does: GET ?chain=<c> is the row's own reply and a GET without chain
every row's under its chain; a POST takes exactly {"chain": <c>, "body": <request>}, and the
inject host's set_trim also {"chains": {<c>: [{prn, trim_chips}, ...]}}, applied to every
named chain or, on any refusal, to none. Refusals are the node's, word for word. Each row keeps
its own seeds, PRN map and trims, so a request that reaches the wrong row shows. --log writes
every request it serves, one JSON line each, for reading the POST stream.

Everything it emits is a deterministic function of the detection-poll index and the requests
posted to it, so two recordings of the same length are identical apart from wall-clock timing
(which the transcript pins anyway).
"""
import argparse
import json
import math
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=12777)
ap.add_argument(
    "--chains",
    default="",
    help="comma list of broker chains each GPU half serves as rows (gnss<g>_<chain>)",
)
ap.add_argument("--halves", type=int, default=2, help="GPU halves with rows")
ap.add_argument(
    "--hosted",
    default="",
    help="comma list of the stages (inject, n2assemble, n2combine) whose host serves the "
    "rows by chain on every half",
)
ap.add_argument("--log", help="append every request served here, one JSON line each")
a = ap.parse_args()

# The fleet broker_fake_l5.jsonl was recorded against. Fixed, not flags: a fleet that differs
# from the blessed fixture's records a transcript the fixture's digest says nothing about.
PRNS = [3, 19, 26, 32]
HOP0 = 114436200145
HOPS_PER_SEC = 195312.5
REVISIT_S = 1.0  # snapshot-hop advance between successive detection sets
CODE_LENGTH = 10230.0
# The anchor served at telescope/time0_ns: a real CHORD frame-0 time, so the dead-reckon
# arithmetic sees a plausible epoch.
TIME0 = 1786167610.000002870
_n = [0]  # detection-poll counter; with the posted requests, the only state
_lock = threading.Lock()
_seeds = [[]]  # last seed payload received on a path that names no row

# The rows' verbs by stage, GET or POST, as the node registers them.
VERBS = {
    "inject": {
        "set_seeds": "POST",
        "set_trim": "POST",
        "get_prns": "GET",
        "set_prns": "POST",
    },
    "n2assemble": {
        "get_spectrum": "GET",
        "set_sat_geometry": "POST",
        "get_elem_cal": "GET",
        "set_elem_sum_shared_ref": "POST",
        "get_lane_inst": "GET",
        "set_elem_gain": "POST",
        "set_elem_sum_adapt": "POST",
        "set_elem_sum_shared": "POST",
        "set_elem_proj": "POST",
        "set_reference_element": "POST",
    },
    "n2combine": {"get_status": "GET", "get_records": "GET", "get_elements": "GET"},
}
HOSTS = {"inject": "inject", "n2assemble": "records", "n2combine": "combine"}
CHAINS = [c for c in a.chains.split(",") if c]
HOSTED = {s for s in a.hosted.split(",") if s}
if HOSTED - set(VERBS):
    ap.error("--hosted takes %s" % ", ".join(VERBS))
# Per row (half, chain): seeds, the PRN map, trims and the other posted bodies by verb.
ROWS = {
    (g, c): {"seeds": [], "prns": list(PRNS), "trims": {}, "posts": {}}
    for g in range(a.halves)
    for c in CHAINS
}
# Requests counted at a row's own path while its host serves it by chain, as the node's
# kotekan_gnss_rest_legacy_hits_total.
_hits = {}
_ROW_PATH = re.compile(
    r"^/gnss(\d+)_([a-z0-9]+_[a-z0-9]+)_(inject|n2assemble|n2combine)/(\w+)$"
)
_HOST_PATH = re.compile(r"^/gnss(\d+)_(inject|records|combine)/(\w+)$")


def _detections():
    """One set per poll, at an advancing ref_hop. Doppler drifts linearly and code phase
    drifts in proportion to Doppler x time, so the rate fit has something to fit."""
    with _lock:
        k = _n[0]
        _n[0] += 1
    out = []
    dh = int(REVISIT_S * HOPS_PER_SEC)
    for i, prn in enumerate(PRNS):
        t = k * REVISIT_S
        dop = -1500.0 + 120.0 * i + 0.42 * t  # Hz, drifting like a real pass
        cp = (137.5 * (i + 1) + 0.0361 * dop * t) % CODE_LENGTH
        out.append(
            {
                "prn": prn,
                "snr": 34.0 + 3.0 * math.sin(0.7 * k + i),
                "doppler_hz": dop,
                "code_phase_chips": cp,
                "ref_hop": HOP0 + k * dh,
                "nh": (k + i) % 20,
                "code_phase_long_chips": cp + CODE_LENGTH * ((k + i) % 20),
                "code_phase_at_ref_chips": cp,
            }
        )
    return out


def _status(seeded, k):
    """Combiner rows for whatever is currently seeded: locked, with a live discriminator."""
    rows = []
    for i, s in enumerate(seeded):
        prn = int(s.get("prn", 0))
        # A small, changing E-L imbalance so the DLL has a real discriminator to act on,
        # and a deep that clears the floor so coherence_s > 0 certifies the lock.
        d = 0.04 * math.sin(0.31 * k + 0.9 * i)
        rows.append(
            {
                "prn": prn,
                "amplitude": 8.0 + i,
                "amp_snr": 21.0 + 2.0 * i,
                "deep_snr": 60.0 + 5.0 * i,
                "deep_amplitude": 40.0 + i,
                "coherence_s": 1.048576,
                "deep_records": 100,
                "deep_floor": 7.0,
                "unbiased_amplitude": 7.5 + i,
                "coh_amplitude": 30.0 + i,
                "doppler_hz": float(s.get("doppler_hz", 0.0)),
                "deep_rate_hz": 0.31 * math.cos(0.17 * k + i),
                "deep_rate_q": 3.4,
                "carrier_hz_resid": 0.5 * math.sin(0.11 * k + i),
                "e_pow": 100.0 * (1.0 - d),
                "p_pow": 140.0,
                "l_pow": 100.0 * (1.0 + d),
                "dll_disc": d,
                "records": 100,
                "n_chan": 7,
                "hop": HOP0 + k * int(REVISIT_S * HOPS_PER_SEC),
            }
        )
    return rows


def _seed_rows(body):
    """The broker POSTs a bare list of seed dicts; a {"seeds": [...]} wrapper is accepted too,
    so a switch to that form does not silently empty the seed table (an empty status table
    reads as "the broker stopped seeding")."""
    if isinstance(body, list):
        rows = body
    elif isinstance(body, dict):
        rows = body.get("seeds") or []
    else:
        rows = []
    return [r for r in rows if isinstance(r, dict)]


def _parse_trims(body, prns):
    """TrackChain::parse_trims: [(prn, trim)] of the row's PRNs, or the refusal's text."""
    try:
        out = []
        for s in body:
            prn, t = int(s["prn"]), float(s["trim_chips"])
            if not math.isfinite(t):
                return None, "trim_chips is not finite"
            if prn in prns:
                out.append((prn, t))
        return out, None
    except (TypeError, KeyError, ValueError) as e:
        return None, "%s" % e


def row_get(g, c, stage, verb, query):
    """(status, reply) of a row's GET."""
    with _lock:
        st, k = ROWS[(g, c)], _n[0]
        if verb == "get_status":
            rows = _status(st["seeds"], k)
            for r in rows:
                r["pow_hop"] = r["hop"]
            return 200, rows
        if verb == "get_records":
            hop = HOP0 + k * int(REVISIT_S * HOPS_PER_SEC)
            recs = []
            for i, s in enumerate(st["seeds"]):
                ph = 0.3 * k + 0.7 * i + 0.11 * g
                recs.append(
                    {
                        "prn": int(s.get("prn", 0)),
                        "records": [
                            [
                                hop - 2048 * j,
                                9.0 * math.cos(ph + 0.01 * j),
                                9.0 * math.sin(ph + 0.01 * j),
                                81.0,
                            ]
                            for j in range(4)
                        ],
                    }
                )
            return 200, recs
        if verb == "get_elements":
            return 200, []
        if verb == "get_prns":
            return (
                200,
                {"prns": list(st["prns"]), "pending": False, "last_hop": HOP0 + k},
            )
        if verb == "get_spectrum":
            # Windows that never finish: the fleet fit asks for the common one and drops
            # every instance, so both of its GETs run and nothing is fitted.
            if "window" in query:
                return 200, {"status": "not_yet", "available": [0, k]}
            return 200, {"available": [0, k]}
        return 200, {}


def row_post(g, c, stage, verb, body):
    """(status, reply) of a row's POST: a trim refusal is the node's 400."""
    with _lock:
        st = ROWS[(g, c)]
        if verb == "set_seeds":
            st["seeds"] = _seed_rows(body)
        elif verb == "set_trim":
            upd, err = _parse_trims(body, st["prns"])
            if err:
                return 400, "bad trim payload: " + err
            st["trims"].update(upd)
        elif verb == "set_prns" and isinstance(body, dict) and "prns" in body:
            st["prns"] = [int(x) for x in body["prns"]]
        else:
            st["posts"][verb] = body
    return 200, {}


def _chains_of(g, stage, verb):
    return [c for c in CHAINS if (g, c) in ROWS and verb in VERBS[stage]]


def host_get(g, stage, verb, path, query):
    chains = _chains_of(g, stage, verb)
    if "chain" not in query:
        out = {}
        for c in chains:
            code, r = row_get(g, c, stage, verb, query)
            if code != 200:
                return code, "%s: %s" % (c, r)
            out[c] = r
        return 200, out
    c = query.pop("chain")
    if c not in chains:
        return (
            400,
            "%s has no chain %s; its chains are %s" % (path, c, ", ".join(chains)),
        )
    return row_get(g, c, stage, verb, query)


def batch_trims(g, path, request):
    """The inject host's set_trim of several chains: every chain's request is checked before
    any is applied, and a refusal names every refused and every unknown chain."""
    chains = _chains_of(g, "inject", "set_trim")
    by_chain = request["chains"]
    if not isinstance(by_chain, dict):
        return (
            400,
            '%s takes {"chains": {<chain>: <request>, ...}}; its chains are %s'
            % (path, ", ".join(chains)),
        )
    unknown = [c for c in by_chain if c not in chains]
    if unknown:
        return (
            400,
            "%s has no chain %s; its chains are %s"
            % (path, ", ".join(unknown), ", ".join(chains),),
        )
    with _lock:
        upd, refused = {}, []
        for c in [c for c in chains if c in by_chain]:
            upd[c], err = _parse_trims(by_chain[c], ROWS[(g, c)]["prns"])
            if err:
                refused.append("%s: bad trim payload: %s" % (c, err))
        if refused:
            return 400, "; ".join(refused)
        for c, u in upd.items():
            ROWS[(g, c)]["trims"].update(u)
    return 200, {}


def host_post(g, stage, verb, path, request):
    chains = _chains_of(g, stage, verb)
    batch = stage == "inject" and verb == "set_trim"
    if batch and isinstance(request, dict) and list(request) == ["chains"]:
        return batch_trims(g, path, request)
    if not (
        isinstance(request, dict)
        and set(request) == {"chain", "body"}
        and isinstance(request["chain"], str)
    ):
        return (
            400,
            '%s takes {"chain": <chain>, "body": <request>}%s; its chains are %s'
            % (
                path,
                ' or {"chains": {<chain>: <request>, ...}}' if batch else "",
                ", ".join(chains),
            ),
        )
    c = request["chain"]
    if c not in chains:
        return (
            400,
            "%s has no chain %s; its chains are %s" % (path, c, ", ".join(chains)),
        )
    return row_post(g, c, stage, verb, request["body"])


def _metrics():
    lines = []
    for (g, c, stage), verbs in sorted(_hits.items()):
        for v, n in sorted(verbs.items()):
            name = "/gnss%d_%s_%s" % (g, c, stage)
            lines.append(
                'kotekan_gnss_rest_legacy_hits_total{stage_name="%s",path="%s/%s"} %d'
                % (name, name, v, n)
            )
    return "\n".join(lines) + "\n"


def route(method, raw_path, body):
    """(status, reply) of a request to the rows or their hosts, or None for any other path."""
    u = urllib.parse.urlsplit(raw_path)
    path = u.path.rstrip("/")
    query = dict(urllib.parse.parse_qsl(u.query, keep_blank_values=True))
    m = _ROW_PATH.match(path)
    if m and (int(m.group(1)), m.group(2)) in ROWS and m.group(4) in VERBS[m.group(3)]:
        g, c, stage, verb = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        if VERBS[stage][verb] != method:
            return 404, "no %s %s" % (method, path)
        if stage in HOSTED:
            with _lock:
                h = _hits.setdefault((g, c, stage), {})
                h[verb] = h.get(verb, 0) + 1
        if method == "GET":
            return row_get(g, c, stage, verb, query)
        return row_post(g, c, stage, verb, body)
    m = _HOST_PATH.match(path)
    if m:
        g, verb = int(m.group(1)), m.group(3)
        stage = {h: s for s, h in HOSTS.items()}[m.group(2)]
        if stage not in HOSTED or g >= a.halves or not CHAINS:
            return 404, "no %s %s" % (method, path)
        if verb == "chains" and method == "GET":
            return (
                200,
                [
                    {
                        "name": "gnss%d_%s" % (g, c),
                        "stage": "/gnss%d_%s_%s" % (g, c, stage),
                        "chain": c,
                    }
                    for c in CHAINS
                ],
            )
        if VERBS[stage].get(verb) != method:
            return 404, "no %s %s" % (method, path)
        if method == "GET":
            return host_get(g, stage, verb, path, query)
        return host_post(g, stage, verb, path, body)
    return None


class H(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, obj, code=200, text=None):
        body = text.encode() if text is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain" if text else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log(self, method, body, code):
        if a.log:
            with _lock, open(a.log, "a") as f:
                f.write(
                    json.dumps({"m": method, "p": self.path, "b": body, "s": code})
                    + "\n"
                )

    def _reply(self, method, r, body=None):
        code, obj = r
        self._log(method, body, code)
        if code != 200:
            return self._send(None, code, json.dumps({"code": code, "message": obj}))
        return self._send(obj)

    def do_GET(self):
        r = route("GET", self.path, None)
        if r is not None:
            return self._reply("GET", r)
        p = self.path.rstrip("/")
        self._log("GET", None, 200)
        if p == "/metrics":
            return self._send(None, text=_metrics())
        if p.endswith("get_detections"):
            return self._send(_detections())
        if p.endswith("get_status"):
            with _lock:
                seeded, k = list(_seeds[0]), _n[0]
            return self._send(_status(seeded, k))
        if "time0_ns" in p:
            return self._send({"time0_ns": TIME0 * 1e9})
        return self._send([])

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(n).decode()) if n else {}
        except Exception:
            body = {}
        r = route("POST", self.path, body)
        if r is not None:
            return self._reply("POST", r, body)
        self._log("POST", body, 200)
        if self.path.rstrip("/").endswith("set_seeds"):
            with _lock:
                _seeds[0] = _seed_rows(body)
        return self._send({"ok": True})


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    print(
        "fake fleet on :%d  prns=%s  rows %s x %d halves, hosted %s"
        % (
            a.port,
            PRNS,
            ",".join(CHAINS) or "-",
            a.halves,
            ",".join(sorted(HOSTED)) or "-",
        ),
        flush=True,
    )
    srv.serve_forever()
