#!/usr/bin/env python3
"""Stage-name resolution for the GNSS tools: speak gps_* everywhere, work anywhere.

The tri-constellation configs name their chains symmetrically -- gps_search / gps_track /
gps_combiner alongside gal_* and bds_* -- but ~25 single-constellation bench/replay configs
still carry the historical BARE names (search / track / combiner), and old recordings pair
with old configs. Rather than a flag day, the tools resolve stage names against the live
pipeline's /endpoints listing: ask for 'gps_search', get 'search' on a config that only has
that, and vice versa. Unknown names pass through untouched (a typo still fails loudly at
the REST call, where the error is legible).

    from gnss_stages import resolve_stage
    stage = resolve_stage("http://localhost:12048", "gps_combiner")   # -> "combiner" (old cfg)
"""
import json
import urllib.request

_CACHE = {}   # base url -> set of stage names registered on that pipeline


def _stages(base):
    """First path component of every registered REST endpoint (i.e. the stage names)."""
    if base in _CACHE:
        return _CACHE[base]
    names = set()
    try:
        with urllib.request.urlopen("%s/endpoints" % base.rstrip("/"), timeout=5) as r:
            eps = json.loads(r.read().decode())
        for method in ("GET", "POST"):
            for path in eps.get(method, []) or []:
                parts = path.strip("/").split("/")
                if len(parts) >= 2:          # "/<stage>/<endpoint>"
                    names.add(parts[0])
    except Exception:
        names = set()                        # pipeline not up yet: pass names through
    if names:
        _CACHE[base] = names                 # only cache a real answer
    return names


def resolve_stage(base, name):
    """Map a requested stage name onto whatever this pipeline actually registered.

    gps_X <-> X in both directions; everything else (gal_*, bds_*, airspy_in, ...) is
    already unambiguous and passes straight through."""
    have = _stages(base)
    if not have or name in have:
        return name
    alt = name[4:] if name.startswith("gps_") else "gps_" + name
    return alt if alt in have else name


def _find_key(node, key):
    """First value of `key` anywhere in the config tree (dicts and lists alike)."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def capture_clock(base, adc_stage="airspy_in"):
    """Convert a record/emit UTC stamp into TRUE unix UTC. Returns f(rec_utc) -> unix.

    The pipeline stamps records on a CAPTURE clock -- `capture_utc0` (a small config
    constant) plus samples-since-sample-0 / sample_rate -- because that is the timescale the
    data actually lives on: exact, monotonic, and gap-honest, where wall-clock-at-emit is
    none of those (pipeline latency, bursty drains). What it is NOT is unix time. The airspy
    supplies the missing anchor: utc0_sample0, the wall UTC it stamped on its first sample.

        unix = utc0_sample0 + (rec_utc - capture_utc0)

    Every tool that compares a pipeline observable against the outside world (an ephemeris,
    another band, another receiver) needs this, and needs it identically -- so it lives here
    once rather than being re-derived, subtly differently, in each of them.

    Falls back to the identity if the anchor isn't up yet (the caller sees raw capture time
    rather than a plausible-looking wrong answer)."""
    base = base.rstrip("/")
    try:
        with urllib.request.urlopen("%s/%s/adcstat" % (base, resolve_stage(base, adc_stage)),
                                    timeout=5) as r:
            utc0 = float(json.loads(r.read().decode()).get("utc0_sample0", 0.0))
        with urllib.request.urlopen("%s/config" % base, timeout=5) as r:
            off = _find_key(json.loads(r.read().decode()), "capture_utc0")
        off = float(off) if off is not None else 0.0
    except Exception:
        return lambda t: t
    if utc0 <= 0.0:
        return lambda t: t
    return lambda t: utc0 + (t - off)
