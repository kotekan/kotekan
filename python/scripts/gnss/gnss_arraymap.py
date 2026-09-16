#!/usr/bin/env python3
"""Which physical feed a correlator element was, at a given time -- and what may be co-added.

    epochs                  list the intervals
    show      [--at T]      the whole array as it stood at T
    element   IDX [--at T]  one element (--cube to index the 32-wide beam-cube axis)
    diff      T1 T2         what changed between two times -- the question that motivates this
    masters   CUBE.npz...   preflight: may these masters be co-added?
    check                   validate the table against the shipped config
    stamp     [--at T]      the {array_epoch, array_epoch_key} an artifact should carry

WHY THIS EXISTS. The element -> physical feed map is NOT a constant. On 2026-09-15 the site
team rewired three dishes, so element 9 before that date and element 9 after it are different
polarisations of the same dish; re-pointing moves the boresight every few months; and the live
element set has grown before and will grow again. Anything that co-adds across time -- a beam
map summed over days, most of all -- has to know whether the thing it is summing stayed the
same thing. Today that knowledge is a table in config/; the intent is a database. Everything
here goes through `Backend`, so that swap is one class.

TWO NUMBERINGS, AND THEY ARE DIFFERENT AXES. The correlator element index runs 0..127 and is
`pol*64 + dish`; the beam cube packs a 32-wide axis, `pol*16 + dish`, in `cube_order`. Crossing
them silently mislabels every polarisation, which is why `element()` takes one or the other
explicitly and there is no arithmetic anywhere else.

THE RESOLVER NEVER EXTRAPOLATES. A time outside every recorded interval raises `NoEpoch`. That
includes the deliberate gap over the day of a recabling whose hour nobody wrote down: refusing
to answer is the correct answer there, and it is the whole value of the hook.

@author Keith Vanderlinde
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))            # python/scripts/gnss
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
CONFIG = os.path.join(_ROOT, "config", "chord_array_epochs.json")
NODE_CONFIG = os.path.join(os.path.dirname(CONFIG), "chord_gnss_node.yaml")


class Refused(Exception):
    """Base for every refusal. Callers asking "is this one array?" catch this.

    The two subclasses answer different questions -- "nothing is recorded here" vs "too much
    is" -- but a caller about to co-add wants both, and which one fires depends on where the
    unrecorded gaps happen to fall. Catching only one is a bug waiting for a boundary to move.
    """


class NoEpoch(Refused):
    """No recorded array configuration covers this time. Never guess past this."""


class Straddle(Refused):
    """The span asked about crosses a configuration change, so it is not one array."""


# ── time ───────────────────────────────────────────────────────────────────────────────────
def as_utc(t):
    """datetime | unix seconds | 'YYYYMMDD' | ISO-8601 -> aware UTC datetime.

    A bare 'YYYYMMDD' means the START of that UTC day. A day is the unit the cube builds in, so
    a day string that lands inside an epoch gap is a real question about that build, not a
    rounding accident -- see `day_span`.
    """
    if isinstance(t, datetime):
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    if isinstance(t, (int, float)):
        return datetime.fromtimestamp(float(t), timezone.utc)
    s = str(t).strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def day_span(daystr):
    """'YYYYMMDD' -> (start, end) of that UTC day, the interval a one-day master covers."""
    t0 = as_utc(daystr)
    return t0, datetime.fromtimestamp(t0.timestamp() + 86400.0, timezone.utc)


# ── records ────────────────────────────────────────────────────────────────────────────────
class Element(object):
    """One correlator element as it stood in one epoch."""

    __slots__ = ("index", "dish", "dish_name", "slot", "plane", "plane_confidence",
                 "enu_m", "position_confidence")

    def __init__(self, index, d):
        self.index = int(index)
        self.dish = d["dish"]
        self.dish_name = d["dish_name"]
        self.slot = d["slot"]
        self.plane = d.get("plane")
        self.plane_confidence = d.get("plane_confidence", "unknown")
        self.enu_m = tuple(d.get("enu_m") or ())
        self.position_confidence = d.get("position_confidence", "nominal")

    @property
    def known(self):
        """True when the feed plane is pinned. `plane is None` is a real state, not a default."""
        return self.plane is not None and self.plane_confidence in ("measured", "reported")

    def label(self):
        return "%s slot%d %s" % (self.dish_name, self.slot,
                                 "plane %s" % self.plane if self.plane else "plane ?")

    def __repr__(self):
        return "<Element %d %s>" % (self.index, self.label())


class Pointing(object):
    __slots__ = ("name", "dec_deg", "bore_az_deg", "bore_el_deg", "source")

    def __init__(self, name, d):
        self.name = name
        self.dec_deg = d["dec_deg"]
        self.bore_az_deg = d["bore_az_deg"]
        self.bore_el_deg = d["bore_el_deg"]
        self.source = d.get("source", "")

    def boresight(self):
        return self.bore_az_deg, self.bore_el_deg

    def __repr__(self):
        return "<Pointing %s dec %+.2f, boresight az %.1f el %.2f>" % (
            self.name, self.dec_deg, self.bore_az_deg, self.bore_el_deg)


class ArrayEpoch(object):
    """The array's identity over one interval [valid_from, valid_to)."""

    def __init__(self, d, pointings):
        self.name = d["name"]
        self.valid_from = as_utc(d["valid_from"])
        self.valid_to = as_utc(d["valid_to"]) if d.get("valid_to") else None
        self.pointing = pointings[d["pointing"]]
        self.cube_order = list(d["cube_order"])
        self.verified = bool(d.get("verified", False))
        self.provenance = d.get("provenance", "")
        self.notes = list(d.get("notes", []))
        self.elements = {int(k): Element(k, v) for k, v in d["elements"].items()}
        self._raw = d

    # -- identity ---------------------------------------------------------------------------
    def key(self):
        """Stable short id: the name plus a digest of what the table actually SAYS.

        The name alone is not enough to stamp an artifact with. If the table is corrected in
        place -- a plane pinned, a position re-measured -- artifacts built against the old
        content must stop matching rather than silently claim agreement.
        """
        body = json.dumps({"cube_order": self.cube_order, "pointing": self.pointing.name,
                           "elements": {str(k): [e.dish, e.slot, e.plane, e.plane_confidence,
                                                 list(e.enu_m)]
                                        for k, e in sorted(self.elements.items())}},
                          sort_keys=True).encode()
        return "%s:%s" % (self.name, hashlib.sha1(body).hexdigest()[:8])

    def contains(self, t):
        t = as_utc(t)
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)

    # -- lookup -----------------------------------------------------------------------------
    def element(self, index=None, cube=None):
        """One element, by correlator index OR by beam-cube index. Exactly one of the two."""
        if (index is None) == (cube is None):
            raise ValueError("give exactly one of index= (correlator 0..127) or cube= (0..31)")
        if cube is not None:
            if not 0 <= cube < len(self.cube_order):
                raise KeyError("cube index %r outside the %d-wide axis of epoch %s"
                               % (cube, len(self.cube_order), self.name))
            index = self.cube_order[cube]
        if index not in self.elements:
            raise KeyError("correlator element %r is not live in epoch %s" % (index, self.name))
        return self.elements[index]

    def cube_elements(self):
        """The cube axis in order: [(cube_index, Element), ...]."""
        return [(i, self.elements[c]) for i, c in enumerate(self.cube_order)]

    def dish_elements(self, dish):
        return sorted((e for e in self.elements.values() if e.dish == dish),
                      key=lambda e: e.slot)

    def stamp(self):
        """What an artifact built from this epoch should carry."""
        return {"array_epoch": self.name, "array_epoch_key": self.key(),
                "pointing": self.pointing.name}

    def __repr__(self):
        return "<ArrayEpoch %s %s..%s %s>" % (
            self.name, self.valid_from.date(),
            self.valid_to.date() if self.valid_to else "open", self.pointing.name)


# ── backends ───────────────────────────────────────────────────────────────────────────────
class Backend(object):
    """Where the table comes from. One method, so the database swap is one class.

    Implementations return epochs sorted by valid_from, non-overlapping. Gaps are allowed and
    meaningful: a gap is "we do not know", and the resolver raises inside one.
    """

    def epochs(self):
        raise NotImplementedError


class JsonBackend(Backend):
    """The shipped table, config/chord_array_epochs.json."""

    def __init__(self, path=CONFIG):
        self.path = path
        self._cache = None

    def epochs(self):
        if self._cache is None:
            with open(self.path) as fh:
                d = json.load(fh)
            if int(d.get("schema", 0)) != 1:
                raise ValueError("%s: schema %r, this module speaks 1" % (self.path, d.get("schema")))
            pts = {k: Pointing(k, v) for k, v in d["pointings"].items()}
            eps = [ArrayEpoch(e, pts) for e in d["epochs"]]
            self._cache = sorted(eps, key=lambda e: e.valid_from)
        return self._cache


class DbBackend(Backend):
    """The intended source once the array database exists. NOT IMPLEMENTED.

    It is here to fix the shape of the question before someone answers it in a hurry. What this
    module needs from a database is one query, and the awkward parts are stated so they are
    designed in rather than discovered:

      * intervals, not snapshots -- 'what was element 9 on 2026-09-08' must be answerable long
        after the fact, so rows carry [valid_from, valid_to) and are never updated in place;
      * a gap must be representable, and must come back as a gap. 'No row' has to mean 'not
        recorded', which the caller turns into NoEpoch. Silently returning the nearest row is
        the one failure mode this whole module exists to prevent;
      * the element set itself is versioned, not just its contents: live_element_ranges changes
        when dishes are added, so the row set per epoch is part of the answer;
      * a content digest travels with the answer, so an artifact stamped with `array_epoch_key`
        can be told apart from one built before a correction landed.

    Until then, keep the JSON authoritative and let the database mirror it -- not the reverse.
    """

    def __init__(self, dsn):
        self.dsn = dsn

    def epochs(self):
        raise NotImplementedError(
            "the array database backend is not written yet; unset GNSS_ARRAYMAP to use %s"
            % CONFIG)


def backend():
    """The backend named by $GNSS_ARRAYMAP ('json:PATH' | 'db:DSN'), else the shipped table."""
    spec = os.environ.get("GNSS_ARRAYMAP")
    if not spec:
        return JsonBackend()
    kind, _, rest = spec.partition(":")
    if kind == "json":
        return JsonBackend(rest or CONFIG)
    if kind == "db":
        return DbBackend(rest)
    raise ValueError("GNSS_ARRAYMAP=%r: expected 'json:PATH' or 'db:DSN'" % spec)


# ── resolution ─────────────────────────────────────────────────────────────────────────────
def epochs(be=None):
    return (be or backend()).epochs()


def at(t, be=None):
    """The epoch covering t. Raises NoEpoch rather than guessing -- read the module docstring."""
    t = as_utc(t)
    for e in epochs(be):
        if e.contains(t):
            return e
    known = epochs(be)
    raise NoEpoch("no recorded array configuration covers %s (known: %s)"
                  % (t.isoformat(), ", ".join("%s %s..%s" % (
                      e.name, e.valid_from.date(),
                      e.valid_to.date() if e.valid_to else "open") for e in known)))


def span(t0, t1, be=None):
    """Every epoch the half-open interval [t0, t1) touches, in order. Empty spans raise."""
    t0, t1 = as_utc(t0), as_utc(t1)
    if t1 <= t0:
        raise ValueError("span end %s is not after start %s" % (t1, t0))
    hit = [e for e in epochs(be)
           if e.valid_from < t1 and (e.valid_to is None or t0 < e.valid_to)]
    if not hit:
        raise NoEpoch("no recorded array configuration covers %s..%s"
                      % (t0.isoformat(), t1.isoformat()))
    # A span that reaches into a gap is not covered even though epochs on both sides are hit.
    cursor = t0
    for e in hit:
        if e.valid_from > cursor:
            raise NoEpoch("%s..%s is not recorded (gap before epoch %s)"
                          % (cursor.isoformat(), e.valid_from.isoformat(), e.name))
        cursor = max(cursor, e.valid_to) if e.valid_to else t1
    if cursor < t1:
        raise NoEpoch("%s..%s is not recorded (past the last epoch)"
                      % (cursor.isoformat(), t1.isoformat()))
    return hit


def require_single(t0, t1, be=None):
    """The one epoch covering [t0, t1). Raises Straddle if the interval crosses a change.

    This is the guard for anything that co-adds: a build over a day, a map summed over days.
    """
    hit = span(t0, t1, be)
    if len(hit) > 1:
        raise Straddle(
            "%s..%s crosses %d array configurations (%s) -- these are not one array and must "
            "not be co-added" % (as_utc(t0).isoformat(), as_utc(t1).isoformat(), len(hit),
                                 ", ".join(e.name for e in hit)))
    return hit[0]


def for_day(daystr, be=None):
    """The single epoch covering a whole UTC day, the unit the cube builds in."""
    t0, t1 = day_span(daystr)
    return require_single(t0, t1, be)


def stamp_for_day(daystr, be=None):
    return for_day(daystr, be).stamp()


# ── co-add compatibility ───────────────────────────────────────────────────────────────────
def assert_compatible(metas, be=None):
    """Refuse a co-add of artifacts that are not the same array. `metas` are master meta dicts.

    Accepts anything carrying `day`, and uses `array_epoch_key` when present. Masters built
    before this module existed carry no stamp; those are resolved from their `day` instead and
    reported as such, because silently trusting an unstamped artifact is how a mixed sum
    happens. Returns (epoch, [notes]).
    """
    seen, notes, unstamped = {}, [], []
    for m in metas:
        who = m.get("path") or m.get("day") or "<unnamed>"
        key = m.get("array_epoch_key")
        if key:
            ep = next((e for e in epochs(be) if e.key() == key), None)
            if ep is None:
                raise Straddle(
                    "%s is stamped %s, which is not in the current table -- it was built "
                    "against a configuration that has since been corrected; rebuild it" % (who, key))
        else:
            if not m.get("day"):
                raise Straddle("%s carries neither array_epoch_key nor day; cannot place it" % who)
            ep = for_day(m["day"], be)
            unstamped.append(who)
        seen.setdefault(ep.key(), (ep, []))[1].append(who)
        pt = m.get("pointing")
        if pt and pt != ep.pointing.name:
            raise Straddle("%s says pointing %s but epoch %s is %s"
                           % (who, pt, ep.name, ep.pointing.name))
    if len(seen) > 1:
        raise Straddle("these artifacts span %d array configurations and must not be "
                       "co-added:\n  %s" % (len(seen), "\n  ".join(
                           "%s: %s" % (ep.name, ", ".join(map(str, w)))
                           for ep, w in seen.values())))
    ep = list(seen.values())[0][0]
    if unstamped:
        notes.append("%d artifact(s) carry no array_epoch stamp; placed by day into %s: %s"
                     % (len(unstamped), ep.name, ", ".join(map(str, unstamped))))
    if not ep.verified:
        notes.append("epoch %s is NOT verified on sky (%s)" % (ep.name, ep.provenance))
    return ep, notes


# ── validation ─────────────────────────────────────────────────────────────────────────────
def check(be=None, node_config=NODE_CONFIG):
    """Validate the table. Returns a list of problems; empty means good."""
    bad = []
    eps = epochs(be)
    if not eps:
        return ["no epochs at all"]
    for a, b in zip(eps, eps[1:]):
        if a.valid_to is None:
            bad.append("epoch %s is open-ended but %s follows it" % (a.name, b.name))
        elif a.valid_to > b.valid_from:
            bad.append("epochs %s and %s overlap" % (a.name, b.name))
    for e in eps:
        if e.valid_to is not None and e.valid_to <= e.valid_from:
            bad.append("epoch %s ends before it starts" % e.name)
        if len(set(e.cube_order)) != len(e.cube_order):
            bad.append("epoch %s repeats an element in cube_order" % e.name)
        for c in e.cube_order:
            if c not in e.elements:
                bad.append("epoch %s: cube_order names element %d, which has no record" % (e.name, c))
        for idx, el in e.elements.items():
            if el.index != idx:
                bad.append("epoch %s: element %d carries index %d" % (e.name, idx, el.index))
            if el.slot * 64 + el.dish != idx:
                bad.append("epoch %s: element %d is not pol*64+dish for dish %d slot %d"
                           % (e.name, idx, el.dish, el.slot))
            if el.plane not in (None, "A", "B"):
                bad.append("epoch %s: element %d has plane %r" % (e.name, idx, el.plane))
        # Both slots of a dish must carry different planes, or neither must be pinned: a dish
        # whose two feeds read as the same plane is a transcription error, not a measurement.
        for dish in sorted({el.dish for el in e.elements.values()}):
            pl = [el.plane for el in e.dish_elements(dish)]
            if len(pl) == 2 and pl[0] is not None and pl[0] == pl[1]:
                bad.append("epoch %s: dish %d has plane %s in both slots" % (e.name, dish, pl[0]))
    # The live epoch's element set must match what the fleet is actually configured to read.
    live = [e for e in eps if e.valid_to is None]
    if live and os.path.exists(node_config):
        want = _live_ranges(node_config)
        if want is not None and want != live[-1].cube_order:
            bad.append("epoch %s cube_order %s != array.live_element_ranges in %s -> %s"
                       % (live[-1].name, live[-1].cube_order, os.path.basename(node_config), want))
    return bad


def _live_ranges(path):
    """array.live_element_ranges from the node yaml, flattened. None if it cannot be read.

    Deliberately a small regex rather than a yaml import: this module is imported by tools that
    run under venv-ft, where pulling in a yaml parser for one list is not worth the dependency.
    """
    import re
    try:
        with open(path) as fh:
            txt = fh.read()
    except IOError:
        return None
    m = re.search(r"^\s*live_element_ranges:\s*(\[.*?\])\s*$", txt, re.M)
    if not m:
        return None
    try:
        out = []
        for lo, hi in json.loads(m.group(1)):
            out.extend(range(int(lo), int(hi) + 1))
        return out
    except Exception:
        return None


# ── CLI ────────────────────────────────────────────────────────────────────────────────────
def _fmt_element(e):
    pos = ("%7.3f %7.3f %5.3f" % e.enu_m) if e.enu_m else " " * 21
    return ("  elem %3d  cube %-4s  %-4s slot%d  plane %-5s %-10s  ENU %s  (%s)"
            % (e.index, "-", e.dish_name, e.slot, e.plane or "?", e.plane_confidence,
               pos, e.position_confidence))


def cmd_epochs(args):
    for e in epochs():
        print("%-22s %s .. %-10s  %-14s  %d elements  %s"
              % (e.name, e.valid_from.date(),
                 e.valid_to.date() if e.valid_to else "open", e.pointing.name,
                 len(e.elements), "verified" if e.verified else "UNVERIFIED"))
        print("    %s" % e.provenance)
    return 0


def cmd_show(args):
    e = at(args.at) if args.at else epochs()[-1]
    print("%s   %s .. %s   %s" % (e.name, e.valid_from.isoformat(),
                                  e.valid_to.isoformat() if e.valid_to else "open", e.pointing))
    print("  %s   key %s" % ("verified" if e.verified else "NOT VERIFIED ON SKY", e.key()))
    print("  %s" % e.provenance)
    for n in e.notes:
        print("    - %s" % n)
    print("  cube axis is %d wide; cube i -> correlator cube_order[i]" % len(e.cube_order))
    print("   cube  elem  dish  slot  plane   confidence   ENU east   north      up   pos")
    for ci, el in e.cube_elements():
        print("   %4d  %4d  %-4s   %d    %-5s   %-10s  %7.3f %7.3f %7.3f  %s"
              % (ci, el.index, el.dish_name, el.slot, el.plane or "?", el.plane_confidence,
                 el.enu_m[0], el.enu_m[1], el.enu_m[2], el.position_confidence))
    return 0


def cmd_element(args):
    e = at(args.at) if args.at else epochs()[-1]
    el = e.element(cube=args.index) if args.cube else e.element(index=args.index)
    print("epoch %s (%s .. %s)" % (e.name, e.valid_from.date(),
                                   e.valid_to.date() if e.valid_to else "open"))
    print("  correlator element %d%s" % (el.index, "  (cube index %d)" % args.index if args.cube
                                         else ""))
    print("  dish %s (index %d), slot %d, feed plane %s (%s)"
          % (el.dish_name, el.dish, el.slot, el.plane or "unknown", el.plane_confidence))
    if len(el.enu_m) == 3:
        print("  ENU %.3f %.3f %.3f m (%s)" % (el.enu_m + (el.position_confidence,)))
    else:
        print("  no position")
    other = [o for o in e.dish_elements(el.dish) if o.index != el.index]
    if other:
        print("  the other feed on this dish: element %d, plane %s"
              % (other[0].index, other[0].plane or "unknown"))
    return 0


def cmd_diff(args):
    a, b = at(args.t1), at(args.t2)
    print("%s  ->  %s" % (a.name, b.name))
    if a.key() == b.key():
        print("  identical: same key %s" % a.key())
        return 0
    if a.pointing.name != b.pointing.name:
        print("  POINTING CHANGED: %s -> %s -- maps across this boundary must never be co-added"
              % (a.pointing, b.pointing))
    if a.cube_order != b.cube_order:
        print("  cube axis changed: %d -> %d elements" % (len(a.cube_order), len(b.cube_order)))
    n = 0
    for idx in sorted(set(a.elements) | set(b.elements)):
        ea, eb = a.elements.get(idx), b.elements.get(idx)
        if ea is None or eb is None:
            print("  elem %3d  %s" % (idx, "added" if ea is None else "removed"))
            n += 1
            continue
        if (ea.dish, ea.slot, ea.plane) != (eb.dish, eb.slot, eb.plane):
            ci = (a.cube_order.index(idx) if idx in a.cube_order else None)
            print("  elem %3d%s  %s  ->  %s" % (
                idx, "  (cube %d)" % ci if ci is not None else "", ea.label(), eb.label()))
            n += 1
    print("  %d element(s) changed identity" % n)
    if n:
        print("  => anything summed across this boundary mixes them. Split the sum.")
    return 0


def cmd_masters(args):
    """Preflight a co-add of beam-cube masters. Reads only the meta blob, never the arrays."""
    import numpy as np                       # only this subcommand needs it
    metas = []
    for path in args.masters:
        z = np.load(path, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        m["path"] = os.path.basename(path)
        metas.append(m)
    ep, notes = assert_compatible(metas)
    print("OK  %d master(s) are one array: %s  (%s)"
          % (len(metas), ep.name, ep.pointing.name))
    print("    key %s" % ep.key())
    for n in notes:
        print("    note: %s" % n)
    return 0


def cmd_check(args):
    bad = check()
    for b in bad:
        print("FAIL  %s" % b)
    if not bad:
        eps = epochs()
        print("OK  %d epoch(s), %s .. %s, no overlaps; live cube_order matches "
              "array.live_element_ranges" % (len(eps), eps[0].valid_from.date(),
                                             eps[-1].valid_to.date() if eps[-1].valid_to else "open"))
        for e in eps:
            if not e.verified:
                print("note  epoch %s is not verified on sky" % e.name)
    return 1 if bad else 0


def cmd_stamp(args):
    e = at(args.at) if args.at else epochs()[-1]
    print(json.dumps(e.stamp(), indent=1))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("epochs").set_defaults(fn=cmd_epochs)
    s = sub.add_parser("show"); s.add_argument("--at", default=None); s.set_defaults(fn=cmd_show)
    s = sub.add_parser("element")
    s.add_argument("index", type=int)
    s.add_argument("--cube", action="store_true", help="index the 32-wide beam-cube axis")
    s.add_argument("--at", default=None)
    s.set_defaults(fn=cmd_element)
    s = sub.add_parser("diff"); s.add_argument("t1"); s.add_argument("t2"); s.set_defaults(fn=cmd_diff)
    s = sub.add_parser("masters")
    s.add_argument("masters", nargs="+", help="cube_<day>_nside<N>.npz")
    s.set_defaults(fn=cmd_masters)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    s = sub.add_parser("stamp"); s.add_argument("--at", default=None); s.set_defaults(fn=cmd_stamp)
    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        p.print_help()
        return 2
    try:
        return a.fn(a)
    except Refused as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
