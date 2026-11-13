#!/usr/bin/env python3
"""The fleet's tracker chains, one row each: `chains:` in config/gnss_fleet_chord.yaml.

A row is the one place a chain's properties are written. The node generator
(gen_chord_gnss_config.py), gen_fleet.py, config/gnss_fleet.py (the broker's endpoints and the
lane relay's chains), the broker's dll-spacing (gnss_fleet.load_chains) and the gnss-obs@ unit
list all read them from here.

    python3 config/gnss_chain_table.py list                  # the rows, validated
    python3 config/gnss_chain_table.py obs-units [--strict]  # gnss-obs@ units against
                                                             # gnss-stack.target

Python 3 standard library + PyYAML only, as config/gnss_fleet.py; validate() imports the
interface golden and chord_band_plan lazily.
"""
import argparse
import collections
import os
import re
import sys

import yaml

CONF = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(CONF, "gnss_fleet_chord.yaml")
STACK_TARGET = os.path.join(
    os.path.dirname(CONF), "scripts", "gnss", "site", "systemd", "gnss-stack.target"
)

ChainRow = collections.namedtuple(
    "ChainRow",
    "id key descriptor prns primary dll_spacing_chips record_cp_mod_chips telem_rows "
    "proj_sys shared_ref_band",
)
#: Row keys in the manifest, and their defaults (None: required).
KEYS = {
    "id": None,
    "key": None,
    "descriptor": None,
    "prns": None,
    "primary": False,
    "dll-spacing-chips": None,
    "record-cp-mod-chips": None,
    "telem-rows": None,
    "proj-sys": None,
    "shared-ref-band": None,
}
OPTIONAL = ("primary", "record-cp-mod-chips")
PROJ_SYS = ("G", "E", "C", "R")


class TableError(ValueError):
    """The chains table is malformed: say which row and what."""


def rows(man):
    """The ChainRows of a parsed manifest, in its order. Raises TableError on a row that is not
    a mapping, lacks a required key or carries an unknown one."""
    out = []
    for i, ent in enumerate((man or {}).get("chains") or []):
        if not isinstance(ent, dict):
            raise TableError("chains[%d]: not a mapping" % i)
        unknown = sorted(set(ent) - set(KEYS))
        missing = sorted(k for k in KEYS if k not in ent and k not in OPTIONAL)
        if unknown or missing:
            raise TableError(
                "chains[%d] (%s): %s"
                % (
                    i,
                    ent.get("key"),
                    "; ".join(
                        ["unknown key %s" % k for k in unknown]
                        + ["no %s" % k for k in missing]
                    ),
                )
            )
        v = {k: ent.get(k, d) for k, d in KEYS.items()}
        out.append(
            ChainRow(
                id=v["id"],
                key=v["key"],
                descriptor=v["descriptor"],
                prns=str(v["prns"]),
                primary=v["primary"] is True,
                dll_spacing_chips=v["dll-spacing-chips"],
                record_cp_mod_chips=v["record-cp-mod-chips"],
                telem_rows=v["telem-rows"],
                proj_sys=v["proj-sys"],
                shared_ref_band=v["shared-ref-band"],
            )
        )
    return out


def load(path=None):
    """The rows of the manifest at `path` (default: config/gnss_fleet_chord.yaml)."""
    with open(path or MANIFEST) as f:
        return rows(yaml.safe_load(f))


def chain_key(descriptor):
    """The chain key a descriptor's stages, endpoints and broker chain are named by
    (gnss_fleet.broker_chain_name): GAL_E5A_Q_CS -> gal_e5a."""
    return "_".join(descriptor.split("_")[:2]).lower()


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(table, descriptors=None):
    """Problems with the rows, as lines (empty = none). `descriptors` is the set of names
    gnssSignal.hpp's signal_by_name resolves (default: the interface golden's signals section,
    which lists ALL_SIGNALS, the list it searches)."""
    if CONF not in sys.path:
        sys.path.insert(0, CONF)
    from chord_band_plan import iface, parse_slots

    if descriptors is None:
        descriptors = set(iface.golden().sections["signals"])

    bad = []
    if not table:
        return ["no chains"]
    for field in ("id", "key", "descriptor"):
        seen = collections.Counter(getattr(r, field) for r in table)
        bad += ["%s %s is on %d rows" % (field, v, n) for v, n in seen.items() if n > 1]
    prim = [r.key for r in table if r.primary]
    if prim != [table[0].key]:
        bad.append(
            "the first row, and only it, is the primary (primary: %s)"
            % (", ".join(prim) or "none")
        )
    for r in table:
        at = "chain %s (id %s)" % (r.key, r.id)
        if not isinstance(r.id, int) or isinstance(r.id, bool) or r.id < 1:
            bad.append("%s: id must be an integer >= 1" % at)
        if r.descriptor not in descriptors:
            bad.append(
                "%s: descriptor %s is not one signal_by_name resolves (gnssSignal.hpp "
                "ALL_SIGNALS, as the interface golden lists it)" % (at, r.descriptor)
            )
        if r.key != chain_key(str(r.descriptor)):
            bad.append(
                "%s: key must be %s, the name its descriptor's stages and endpoints take"
                % (at, chain_key(str(r.descriptor)))
            )
        try:
            prns, _ = parse_slots(r.prns)
        except ValueError as e:
            bad.append("%s: prns: %s" % (at, e))
        else:
            if not prns or len(set(prns)) != len(prns):
                bad.append("%s: prns must be a non-empty list without repeats" % at)
        if not (_num(r.dll_spacing_chips) and 0 < r.dll_spacing_chips <= 1):
            bad.append("%s: dll-spacing-chips must be in (0, 1]" % at)
        if r.record_cp_mod_chips is not None and not (
            isinstance(r.record_cp_mod_chips, int) and r.record_cp_mod_chips > 0
        ):
            bad.append("%s: record-cp-mod-chips must be a positive integer" % at)
        if not (isinstance(r.telem_rows, int) and r.telem_rows > 0):
            bad.append("%s: telem-rows must be a positive integer" % at)
        if r.proj_sys not in PROJ_SYS:
            bad.append("%s: proj-sys must be one of %s" % (at, " ".join(PROJ_SYS)))
        if not (isinstance(r.shared_ref_band, str) and r.shared_ref_band):
            bad.append("%s: shared-ref-band must be a band name" % at)
    return bad


def dll_spacing_drift(table, have, where):
    """Lines for every chain in `have` ({chain key: dll spacing}) whose value is not its row's:
    the node taps and the broker's trim law must use one spacing (gen_chord_gnss_config.py
    puts the row's on the taps, gnss_fleet.load_chains the row's on the broker)."""
    want = {r.key: r.dll_spacing_chips for r in table}
    return [
        "%s: %s dll spacing %s, the chains table says %s" % (where, c, v, want.get(c))
        for c, v in sorted(have.items())
        if want.get(c) != v
    ]


def obs_units(table):
    """The gnss-obs@ writer units the broker's chains need, one per row."""
    return ["gnss-obs@%s.service" % r.key for r in table]


def stack_obs_units(path=STACK_TARGET):
    """The gnss-obs@ units a stack target's Wants= lines start, in order."""
    with open(path) as f:
        text = f.read()
    return re.findall(
        r"gnss-obs@[\w.-]+\.service",
        "\n".join(ln for ln in text.splitlines() if ln.startswith("Wants=")),
    )


def obs_unit_check(table, path=STACK_TARGET):
    """(missing, extra): the table's obs units the target does not start, and the target's
    that no row needs."""
    have, want = stack_obs_units(path), obs_units(table)
    return [u for u in want if u not in have], [u for u in have if u not in want]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default=MANIFEST)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the rows, validated")
    o = sub.add_parser(
        "obs-units", help="the gnss-obs@ units the rows need against the stack target"
    )
    o.add_argument("--target", default=STACK_TARGET)
    o.add_argument(
        "--strict", action="store_true", help="exit 1 when the two lists differ"
    )
    a = ap.parse_args(argv)
    try:
        table = load(a.manifest)
    except TableError as e:
        sys.exit("%s: %s" % (a.manifest, e))
    if a.cmd == "list":
        for r in table:
            print(
                "%3d %-9s %-14s dll %-4s telem %2d %s %-5s%s%s"
                % (
                    r.id,
                    r.key,
                    r.descriptor,
                    r.dll_spacing_chips,
                    r.telem_rows,
                    r.proj_sys,
                    r.shared_ref_band,
                    " cp mod %d" % r.record_cp_mod_chips
                    if r.record_cp_mod_chips
                    else "",
                    "  primary" if r.primary else "",
                )
            )
        bad = validate(table)
        for line in bad:
            print("INVALID  %s" % line)
        return 1 if bad else 0
    missing, extra = obs_unit_check(table, a.target)
    print(
        "%d chains, %d gnss-obs@ units in %s"
        % (len(table), len(stack_obs_units(a.target)), a.target)
    )
    for u in missing:
        print("missing  %s" % u)
    for u in extra:
        print("extra    %s" % u)
    return 1 if a.strict and (missing or extra) else 0


if __name__ == "__main__":
    sys.exit(main())
