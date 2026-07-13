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
