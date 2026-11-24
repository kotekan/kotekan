"""The receiver's cache of decoded broadcast ephemerides: its own orbit feed, beside the BRDC
dailies and the GuoWang OMM.

Every decoder that reads a satellite's own message (nav_decode.py and the bootstrap's) writes
what it decodes here; the broker and the offline tools (gnss_fix, gnss_skytime) read it through
decoded_eph.load_files. Each satellite's issues are kept for days, so after a restart with no
network the receiver still has the orbits it last decoded, as the BRDC cache keeps its dailies.

Layout: one JSON file per writer, <dir>/<writer>.json, in nav_decode's --eph-out schema
({"schema", "t_unix", "eph": [...]}). A writer owns its file, merges into it what it read back
at start, and replaces it atomically, so writers never contend and a restart keeps the history.
Readers merge every *.json in the directory. Each entry is {sys, prn, signal, toe_field, eph,
toe_gpst, week_from, t_decoded}: toe_gpst is the issue's reference time as continuous GPST, placed
from the sky's own week (week_from "sky") or else the host clock's ("host"), and an issue is
kept once per (sys, prn, signal, toe_gpst), its newest decode. The almanacs a decoder reads
(nav_almanac.py) are kept the same way in a writer of their own, <writer>.alm, with t_oa as
toe_gpst; only decoded_eph.Reader(almanac=True) reads them.

Retention, against the writer's newest toe_gpst rather than any clock: every issue within
RECENT_S, one an hour per satellite and signal within HOURLY_S, one a day within KEEP_S.

    python3 eph_cache.py [DIR]          # what the cache holds, by writer and constellation
"""

import collections
import glob
import json
import os
import sys
import time

DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps", "decoded")
RECENT_S = 6 * 3600.0
HOURLY_S = 3 * 86400.0
KEEP_S = 30 * 86400.0
_KEYS = ("sys", "prn", "signal", "toe_field", "eph", "toe_gpst")


def _key(e):
    return (e["sys"], int(e["prn"]), e["signal"], round(float(e["toe_gpst"])))


def read(path):
    """The well-formed entries of one cache file; none if it cannot be read."""
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return []
    return [
        e
        for e in doc.get("eph", [])
        if isinstance(e, dict) and all(k in e for k in _KEYS)
    ]


def load(cache_dir=DEFAULT_DIR):
    """Every writer's entries."""
    out = []
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        out += read(path)
    return out


def thin(entries):
    """The entries kept: see the module's retention."""
    if not entries:
        return []
    # The newest from the sky's week where any is: one issue a wrong host clock put weeks ahead
    # would otherwise prune the rest.
    sky = [e for e in entries if e.get("week_from") == "sky"] or entries
    newest = max(float(e["toe_gpst"]) for e in sky)
    kept, slots = [], set()
    for e in sorted(entries, key=lambda e: -float(e["toe_gpst"])):
        age = newest - float(e["toe_gpst"])
        if age <= RECENT_S:
            kept.append(e)
            continue
        if age > KEEP_S:
            continue
        span = 3600.0 if age <= HOURLY_S else 86400.0
        slot = (
            e["sys"],
            int(e["prn"]),
            e["signal"],
            span,
            float(e["toe_gpst"]) // span,
        )
        if slot not in slots:
            slots.add(slot)
            kept.append(e)
    return kept


class Writer:
    """One writer's file in the cache. Not shared: a second process under the same name would
    overwrite the first's history."""

    def __init__(self, name, cache_dir=DEFAULT_DIR):
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, "%s.json" % name)
        self.entries = {_key(e): e for e in read(self.path)}

    def put(self, entries, t_unix=None):
        """Add decoded issues (dicts with the entry keys but t_decoded) and rewrite the file if
        anything is new. Returns how many were new."""
        t_unix = time.time() if t_unix is None else t_unix
        new = 0
        for e in entries:
            k = _key(e)
            if k not in self.entries or self.entries[k]["eph"] != e["eph"]:
                new += 1
            self.entries[k] = dict(e, t_decoded=round(t_unix, 2))
        if new:
            self.entries = {_key(e): e for e in thin(list(self.entries.values()))}
            doc = {"schema": 1, "t_unix": round(t_unix, 2)}
            doc["eph"] = sorted(self.entries.values(), key=_key)
            tmp = "%s.%d.tmp" % (self.path, os.getpid())
            with open(tmp, "w") as fh:
                json.dump(doc, fh, default=_jsonable)
            os.replace(tmp, self.path)
        return new


def _jsonable(o):
    """json.dump's fallback for the decoders' numpy scalars and containers."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


def main(argv=None):
    cache_dir = (argv or sys.argv[1:] or [DEFAULT_DIR])[0]
    gps0 = 315964800.0 - 18.0  # GPST epoch less the leap seconds, for a readable date
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        ents = read(path)
        print("%s: %d issues" % (os.path.basename(path), len(ents)))
        by = collections.defaultdict(list)
        for e in ents:
            by[(e["sys"], e["signal"])].append(e)
        for (s, sig), v in sorted(by.items()):
            t = [float(e["toe_gpst"]) for e in v]
            print(
                "  %s %-16s %3d satellites, %4d issues, toe %s .. %s, week from %s"
                % (
                    s,
                    sig,
                    len({e["prn"] for e in v}),
                    len(v),
                    time.strftime("%m-%d %H:%M", time.gmtime(min(t) + gps0)),
                    time.strftime("%m-%d %H:%M", time.gmtime(max(t) + gps0)),
                    "/".join(sorted({e.get("week_from", "?") for e in v})),
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
