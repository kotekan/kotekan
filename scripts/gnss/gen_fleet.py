#!/usr/bin/env python3
"""Generate (or verify) every CHORD GNSS node config from one versioned manifest.

    scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml            # write them
    scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml --check    # verify, write nothing
    scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml --node cx19 --print-cmd

The manifest holds every node's config/gen_chord_gnss_config.py flags, so each config can be
reproduced from a committed file; a node's config is regenerated, never hand-edited. Its
`nodes:` is the fleet list (config/gnss_fleet.py): the nodes written are its hosts, each node's
search-port-base comes from its slot, and the search aggregator's config (the manifest's
`aggregator:` block) is written from the same list, so a node joining or leaving is one line
there and one run of this.

`--check` is the gate to run before a fleet restart. It regenerates every node into memory,
compares the result byte for byte with the file on disk (the live EOP and bad-input values
aside; the node is regenerated with the bad-input block its file carries, since the GNSS
element lists are derived from it) and checks the file against the stock render
(stock_parity.py), so a hand-edit, a changed base or a generator whose output moved all fail.
It does the same for the aggregator's config, and checks the chains yaml against the fleet:
every node endpoint key must be `fleet` (or null), and the chain instances the broker will
derive must be exactly the chains each regenerated config builds. It prints a diff per
mismatching node
and exits 1 if any node differs, 2 if nothing could be compared (a missing base, a generator
that cannot run, a crash here).

`--check-prns` is a separate, network-dependent check of the manifest's PRN lists against live
BRDC.
"""
import argparse
import difflib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import traceback
import urllib.parse

import yaml

import stock_parity

K = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(K, "config", "gen_chord_gnss_config.py")
OUTDIR = os.path.join(K, "config", "generated")
sys.path.append(os.path.join(K, "python", "scripts", "gnss"))
sys.path.insert(0, os.path.join(K, "config"))
import kotekan_src  # noqa: E402
import gnss_fleet  # noqa: E402

CHAINS = os.path.join(K, "config", "gnss_chains_chord.yaml")


def _split_header(text):
    """(leading '#' comment lines, the YAML body). The generator writes its provenance header,
    then yaml.safe_dump(sort_keys=True), so the body holds no comments."""
    lines = text.splitlines(True)
    n = 0
    while n < len(lines) and lines[n].startswith("#"):
        n += 1
    return "".join(lines[:n]), "".join(lines[n:])


# The generator's interface stamp (gnss_broker/iface.py stamp_line): the golden the frames were
# sized from. --check leaves it out of the comparison: it configures nothing (the deploy
# preflight checks it against the binary), and a config written before the stamp has none,
# which must not read as a difference.
_STAMP = "# gnss_iface: "


def _stamp(text):
    """The config's gnss_iface stamp line, or None."""
    for line in _split_header(text)[0].splitlines():
        if line.startswith(_STAMP):
            return line
    return None


def _drop_stamp(text):
    head, body = _split_header(text)
    return (
        "".join(ln for ln in head.splitlines(True) if not ln.startswith(_STAMP)) + body
    )


def _drop_live(text):
    """Blank the live values in a rendered config: the Earth-orientation table and bffs's
    bad-input block. Both are fetched at generation time and move on their own (the EOP window
    rolls daily, the bad-input list changes whenever bffs flags a feed), so --check must not
    report them as drift.

    Parsed, then re-dumped exactly as the generator dumps, so the round trip is byte-exact and
    everything outside those two values still compares byte for byte.
    """
    head, body = _split_header(text)
    cfg = yaml.load(body, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    if isinstance(cfg, dict):
        if isinstance(cfg.get("earth_rotation_data"), dict):
            cfg["earth_rotation_data"]["earth_orientation_parameter_table"] = None
        if (
            isinstance(cfg.get("updatable_config"), dict)
            and "bad_inputs" in cfg["updatable_config"]
        ):
            cfg["updatable_config"]["bad_inputs"] = None
    return head + yaml.dump(
        cfg,
        Dumper=getattr(yaml, "CSafeDumper", yaml.SafeDumper),
        default_flow_style=False,
        sort_keys=True,
    )


def _bad_input_block_of(text):
    """The updatable_config/bad_inputs block a config carries, or None."""
    cfg = yaml.load(
        _split_header(text)[1], Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    )
    blk = ((cfg or {}).get("updatable_config") or {}).get("bad_inputs")
    return blk if isinstance(blk, dict) and "bad_inputs" in blk else None


def _live_inputs_missing(text):
    """Why a generated config must not be written, or None. Both live values must have come from
    a live source: the stock template's EOP table is empty, and the stock telescope block sets
    fatal_eop_out_of_range, so a node without a current table shuts down; the template's
    bad-input block flags no feed at all, and the mask rides to recv1 with the N^2."""
    cfg = yaml.load(
        _split_header(text)[1], Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    )
    why = []
    if not (cfg.get("earth_rotation_data") or {}).get(
        "earth_orientation_parameter_table"
    ):
        why.append("no EOP table (choco and the fleet both unreachable)")
    bi = (cfg.get("updatable_config") or {}).get("bad_inputs") or {}
    if bi.get("update_id") in (None, "initial_flags"):
        why.append(
            "bad_inputs is the template's default (bffs and the stock nodes unreachable)"
        )
    return "; ".join(why) or None


def _relay_notes(node, stderr):
    """Echo the generator's SKIPPED and WARNING lines for one node to stderr, each once. The
    generator writes its summary there and gen_fleet keeps it quiet on success, but a chain
    skipped on a node (no covering channel) or a warning is not something a fleet regeneration
    may hide."""
    seen = set()
    for line in stderr.splitlines():
        line = line.strip()
        if ("SKIPPED" in line or "WARNING" in line) and line not in seen:
            seen.add(line)
            sys.stderr.write("%s: %s\n" % (node, line))


def flags_from(mapping):
    """Manifest keys -> generator flags, same convention as gnss_chains_chord.yaml:
    true -> a bare flag, false/None -> omitted, anything else -> --key value.

    A list repeats the flag, as argparse's action="append" expects:
    `extra-signal: [A, B]` -> `--extra-signal A --extra-signal B`.
    """
    out = []
    for k, v in mapping.items():
        if v is False or v is None:
            continue
        for item in v if isinstance(v, (list, tuple)) else [v]:
            out.append("--" + k)
            if item is not True:
                out.append(str(item))
    return out


def common_flags(man):
    """The manifest's `common:` as generator flags, less the keys only site services read."""
    common = man.get("common") or {}
    return flags_from({k: v for k, v in common.items() if k not in gnss_fleet.SITE_KEYS})


# ---- PRN content gate (--check-prns) -------------------------------------------------------
# --check compares the generated files and is offline and deterministic. --check-prns asks a
# different question: are the manifest's hand-written PRN lists, the slots each node boots
# with, still true of the sky? They drift as the constellation changes, and nothing else
# checks the manifest against live BRDC (the broker's PRN map only re-assigns running slots).
#
# The two findings are not equally bad, and the exit code says so:
#   EXCLUDED (active, capable, rises here, no slot)  -> a satellite missing from the list: fails.
#   DEAD     (slot, no active satellite)             -> wasted capacity: warns.
#   DARK     (slot, active, cannot broadcast it)     -> a noise probe on a chain that sets
#                                                       probe-dark: noted; elsewhere DEAD.
SYS_OF = {"GPS": "G", "GAL": "E", "BDS": "C", "GLO": "R", "SBAS": "S"}
LAT, LON, ALT = 49.32075144444, -119.62081125, 545.0
# Capability, not just activity: BeiDou B2a/B2b are BDS-3 signals the BDS-2 satellites do not
# broadcast, and L2C/L5 are carried only by the newer GPS blocks, so leaving those satellites
# out of a list is correct and must not be reported. The rule is the IGS registry's block types
# (prnmap.signal_incapable_prns), the source the broker filters on; a PRN floor is no proxy,
# since BeiDou re-uses low PRNs for BDS-3 satellites and GLONASS's L3OC-capable generations are
# interspersed across the slots. A list that mirrors a sibling's (B3I follows B2a/B2b so
# tau_band sees the same rays) is judged by the sibling's capability.
MIRRORS = {"BDS_B3I_NH": "BDS_B2A_P_CS"}
# Degrees: only a satellite whose peak elevation here exceeds this counts as excluded; one that
# merely grazes the horizon has no claim on a boot slot. The manifest's BeiDou lists are cut at
# the same threshold.
RISE_DEG = 10.0


def check_prns(man, rise_deg=RISE_DEG, hours=24.0, step_min=10.0):
    """Compare every manifest PRN list against live BRDC + a 24 h visibility sweep."""
    import time
    from datetime import datetime, timezone

    sys.path.insert(0, os.path.join(K, "python", "scripts", "gnss"))
    from gnss_ephemeris import (
        fetch_brdc,
        glonass_freq_channels,
        parse_rinex_nav,
        predict_all,
    )
    from gnss_broker.prnmap import ensure_registry, signal_incapable_prns
    from chord_band_plan import parse_slots

    ensure_registry()

    lists = {}
    fdma_k = {}  # FDMA chains (PRN@k): {name: {slot: k}}
    for ent in man.get("common", {}).get("extra-signal") or []:
        name, _, prn_s = str(ent).partition(":")
        prns, ks = parse_slots(prn_s)
        lists[name] = sorted(prns)
        if ks:
            fdma_k[name] = dict(zip(prns, ks))
    if not lists:
        print("no extra-signal PRN lists in the manifest -- nothing to check")
        return 0

    # Chains whose broker block sets probe-dark (config/gnss_chains_chord.yaml): only there is a
    # slot for a satellite that cannot broadcast the signal a noise probe rather than waste.
    # Keyed like the broker's chains: GLO_L2OC_P_OC -> glo_l2oc.
    try:
        _chains = yaml.safe_load(
            open(os.path.join(K, "config", "gnss_chains_chord.yaml"))
        ).get("chains", {})
    except (OSError, yaml.YAMLError, AttributeError):
        _chains = {}
    probe_dark = {
        c for c, spec in _chains.items() if (spec or {}).get("probe-dark") is True
    }

    eph = parse_rinex_nav(fetch_brdc(datetime.now(timezone.utc)))
    # Peak elevation over the sweep: a satellite that never rises here has no claim on a slot,
    # however active it is.
    now = time.time()
    peak = {}
    for k in range(int(hours * 60 / step_min)):
        pd = predict_all(
            eph,
            LAT,
            LON,
            ALT,
            now + k * step_min * 60.0,
            mask_deg=-90.0,
            max_age=86400.0,
        )
        for key, v in pd.items():
            if v["el"] > peak.get(key, -90.0):
                peak[key] = v["el"]

    bad = 0
    for name in sorted(lists):
        sysid = SYS_OF.get(name.split("_")[0])
        if sysid is None:
            print("%-16s UNKNOWN constellation prefix -- skipped" % name)
            continue
        cfg = set(lists[name])
        incap = signal_incapable_prns(MIRRORS.get(name, name))
        active = {p for (s, p) in peak if s == sysid and p not in incap}
        rises = {p for p in active if peak[(sysid, p)] > rise_deg}
        # A slot for an active satellite that cannot broadcast the signal: signal-free at any
        # elevation, the chain's noise probe under --probe-dark (glo_l2oc's GLO-M+ slots). On a
        # chain without it the broker never seeds such a slot, so it stays DEAD.
        chain = "_".join(name.split("_")[:2]).lower()
        dark = (
            sorted(p for p in cfg - active if (sysid, p) in peak and p in incap)
            if chain in probe_dark
            else []
        )
        dead = sorted(cfg - active - set(dark))
        excluded = sorted((active & rises) - cfg)
        if not active:
            # An empty sky excludes nothing, so without this a stale or missing ephemeris
            # would pass.
            bad += 1
            print(
                "%-16s FAIL  no active satellite in the sky: stale or empty ephemeris"
                % name
            )
            continue
        print(
            "%-16s slots %2d | capable+active %2d | rises >%.0f deg %2d%s"
            % (
                name,
                len(cfg),
                len(active),
                rise_deg,
                len(rises),
                ("  [capability: %d excluded by the IGS registry]" % len(incap))
                if incap
                else "",
            )
        )
        if dark:
            print(
                "      note  %d dark slot(s) (active, cannot broadcast this signal: noise "
                "probes under --probe-dark): %s" % (len(dark), dark)
            )
        if dead:
            print(
                "      WARN  %d dead slot(s) (configured, no active satellite): %s"
                % (len(dead), dead)
            )
        if excluded:
            bad += 1
            print(
                "      FAIL  %d ACTIVE, CAPABLE and VISIBLE but EXCLUDED: %s  (peak el %s)"
                % (
                    len(excluded),
                    excluded,
                    ", ".join("%.0f" % peak[(sysid, p)] for p in excluded),
                )
            )
            if dead:
                print(
                    "            -> %d dead slot(s) available; swap rather than resize."
                    % len(dead)
                )
        if name in fdma_k:
            # an FDMA slot at the wrong frequency channel despreads another carrier: no error,
            # no signal
            brdc_k = glonass_freq_channels()
            wrong = sorted(
                (p, k, brdc_k.get(p))
                for p, k in fdma_k[name].items()
                if p in brdc_k and brdc_k[p] != k
            )
            if wrong:
                bad += 1
                print(
                    "      FAIL  %d slot(s) at a frequency channel BRDC no longer gives "
                    "(slot, manifest k, BRDC k): %s" % (len(wrong), wrong)
                )
        if not dead and not excluded:
            print("      ok")
    print(
        "\nPRN CONTENT GATE %s"
        % ("RED (a visible satellite has no slot)" if bad else "GREEN")
    )
    return 1 if bad else 0


_COMBINE = re.compile(r"^gnss(\d+)(_[a-z0-9]+)?_n2combine$")


def built_instances(text):
    """{tag: {gpu}} of the chains a generated node config builds: one n2combine stage per
    chain instance (gnss<g><tag>_n2combine; the primary's tag is '')."""
    cfg = yaml.load(
        _split_header(text)[1], Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    )
    out = {}
    for k in cfg or {}:
        m = _COMBINE.match(str(k))
        if m:
            out.setdefault(m.group(2) or "", set()).add(int(m.group(1)))
    return out


def derived_instances(fleet, host):
    """{tag: {gpu}} of the chains the fleet derivation puts on this host."""
    out = {}
    for chain, (_name, tag) in fleet.tracker_signals().items():
        gpus = {g for h, g in fleet.instances(chain) if h == host}
        if gpus:
            out[tag] = gpus
    return out


def check_chains(fleet, path=CHAINS):
    """Problems with the chains yaml against the fleet list, as lines (empty = none): a node
    endpoint key written by hand rather than `fleet`, or an expanded URL on a host outside
    the fleet."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    keys = set(gnss_fleet.URL_KEYS) | set(gnss_fleet.SINGLE_KEYS)
    bad = []
    common = raw.get("common") or {}
    for where, block in [("common", common)] + [
        ("chains." + c, v or {}) for c, v in (raw.get("chains") or {}).items()
    ]:
        for k in sorted(keys & set(block)):
            v = block[k]
            if v is None or v == gnss_fleet.FLEET:
                continue
            bad.append(
                "%s: %s is written by hand (%s); set it to `fleet` so it follows the "
                "fleet list" % (where, k, str(v)[:60])
            )
    try:
        cfg = gnss_fleet.expand_chains(raw, fleet)
    except gnss_fleet.FleetError as e:
        return bad + [str(e)]
    hosts = set(fleet.hosts)
    for c, spec in (cfg.get("chains") or {}).items():
        merged = dict(cfg.get("common") or {})
        merged.update(spec or {})
        for k in sorted(keys):
            v = merged.get(k)
            if not isinstance(v, str):
                continue
            for url in v.split(","):
                host = urllib.parse.urlsplit(url.strip()).hostname
                if (
                    host
                    and host not in hosts
                    and host not in ("localhost", "127.0.0.1")
                ):
                    bad.append(
                        "chains.%s: %s names %s, which is not in the fleet list (%s)"
                        % (c, k, host, " ".join(fleet.hosts))
                    )
    return bad


def check_instances(fleet, regen):
    """Lines for every node whose regenerated config builds other chain instances than the
    fleet derivation lists for it."""
    out = []
    for node, text in sorted(regen.items()):
        have, want = built_instances(text), derived_instances(fleet, node)
        if have != want:
            out.append(
                "%s: the generator builds %s, the fleet derivation lists %s"
                % (
                    node,
                    sorted((t, sorted(g)) for t, g in have.items()),
                    sorted((t, sorted(g)) for t, g in want.items()),
                )
            )
    return out


def check_combs(fleet, hosts):
    """Lines for every host whose frames name other combs than the node table; a host that
    does not answer is noted on stderr, not failed (a node down for work)."""
    out = []
    for host, tab, live in gnss_fleet.comb_report(fleet, hosts):
        if live is None:
            sys.stderr.write(
                "combs: %s did not answer; its table entry is unverified\n" % host
            )
        elif live != tab:
            out.append(
                "%s: the node table says combs %s, its frames say %s (python3 "
                "config/gnss_fleet.py combs --write, then regenerate)"
                % (host, tab, live)
            )
    return out


def aggregator_cmd(man, fleet, base_dir):
    """(generator argv, output path) for the manifest's aggregator: block, or None."""
    agg = man.get("aggregator")
    if not agg:
        return None
    agg = dict(agg)
    base = agg.pop("base")
    base = base if os.path.isabs(base) else os.path.join(K, "config", base)
    out = os.path.join(base_dir, agg.pop("out"))
    feeds = ["%s:%d" % (h, fleet.search_port_base(h)) for h in fleet.hosts]
    cmd = (
        [sys.executable, GEN, "--base", base, "--node", fleet.hosts[0]]
        + ["--aggregator-instance"]
        + feeds
        + flags_from(agg.pop("flags", None) or {})
    )
    if agg:
        raise SystemExit("aggregator: unknown keys %s" % ", ".join(sorted(agg)))
    return cmd, out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("manifest")
    ap.add_argument(
        "--check",
        action="store_true",
        help="regenerate each node into memory, compare it byte for byte with the file on "
        "disk (the live EOP table and bad-input list aside) and check the file against "
        "the stock render (stock_parity.py); write nothing. Exit 1 if any node differs, "
        "2 if nothing could be compared",
    )
    ap.add_argument(
        "--node",
        action="append",
        default=None,
        help="restrict to these nodes (repeatable); default is every node",
    )
    ap.add_argument(
        "--print-cmd",
        action="store_true",
        help="print the full generator command line instead of running it",
    )
    ap.add_argument(
        "--preview-dir",
        default=None,
        metavar="DIR",
        help="write the configs under DIR instead of config/generated/, to look at a "
        "regeneration without touching the files the nodes start from",
    )
    ap.add_argument(
        "--print-path",
        action="store_true",
        help="print the config path this manifest owns for each node and exit, so a "
        "caller (node_up.sh) can tell exactly whether a file is one the manifest owns "
        "rather than guess from the node name: the suffix lives in the manifest.",
    )
    ap.add_argument(
        "--check-prns",
        action="store_true",
        help="check the manifest's extra-signal PRN lists against live BRDC and a 24 h "
        "visibility sweep, and exit: reports dead slots (no active satellite; warn), "
        "dark slots on a probe-dark chain (noted), and active, capable satellites that "
        "rise above 10 deg here without a slot (fail, exit 1). Not part of --check, "
        "which is offline and deterministic; this one needs the network, and its "
        "answer changes as the constellation does.",
    )
    a = ap.parse_args()

    man_path = a.manifest if os.path.isabs(a.manifest) else os.path.join(K, a.manifest)
    with open(man_path) as f:
        man = yaml.safe_load(f)

    if a.check_prns:
        return check_prns(man)

    try:
        fleet = gnss_fleet.Fleet(man)
    except gnss_fleet.FleetError as e:
        sys.exit("%s: %s" % (man_path, e))

    if a.print_path:
        sfx = man.get("suffix", "")
        for n in sorted(fleet.hosts):
            if not a.node or n in a.node:
                print(os.path.join(OUTDIR, "chord_gnss_%s%s.yaml" % (n, sfx)))
        return

    base = man["base"]
    base = base if os.path.isabs(base) else kotekan_src.src("config", base)
    if not os.path.exists(base):
        # Exit 2: nothing was compared (see the generator failure below).
        sys.stderr.write(
            "base config %s does not exist. It is a real INPUT -- the generator injects\n"
            "into it -- so it must be committed, not re-captured ad hoc.\n" % base
        )
        raise SystemExit(2)
    suffix = man.get("suffix", "")
    common = common_flags(man)
    # Every config, written or checked, is diffed against the stock render (stock_parity.py);
    # an undeclared difference stops a write and fails --check. Rendered once here.
    stock_ref = stock_parity.render_stock(base)

    def parity(text):
        cfg = yaml.load(
            _split_header(text)[1], Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        )
        return stock_parity.classify(stock_parity.differences(stock_ref, cfg))[1]

    nodes = sorted(fleet.hosts)
    if a.node:
        missing = [n for n in a.node if n not in nodes]
        if missing:
            sys.exit(
                "not in the fleet: %s (the fleet is %s%s)"
                % (
                    ", ".join(missing),
                    ", ".join(nodes),
                    "; role off: %s"
                    % ", ".join(h for h in fleet.slots if fleet.roles[h] == "off")
                    if len(fleet.slots) > len(fleet.hosts)
                    else "",
                )
            )
        nodes = [n for n in nodes if n in a.node]

    # The fleet checks need this checkout's chains yaml beside the manifest; a manifest
    # elsewhere (a test's) has none and skips them.
    chains = os.path.join(os.path.dirname(man_path), os.path.basename(CHAINS))
    fleet_checks = os.path.exists(chains)
    # The combs the configs are built for, against the frames. A config built for the wrong
    # comb despreads channels its replicas are not for (#194), so it is not written.
    comb_problems = (
        check_combs(fleet, nodes) if fleet_checks and not a.print_cmd else []
    )
    if comb_problems and not a.check:
        sys.exit(
            "NOT WRITTEN: the node table's combs disagree with the frames:\n  %s"
            % "\n  ".join(comb_problems)
        )

    bad, written, regen = [], [], {}
    for node in nodes:
        out = os.path.join(
            a.preview_dir or OUTDIR, "chord_gnss_%s%s.yaml" % (node, suffix)
        )
        # The node's port is the fleet's (its slot); gnss_fleet refuses a search-port-base
        # anywhere in the manifest, so nothing after it overrides it.
        cmd = (
            [sys.executable, GEN, "--base", base, "--node", node]
            + common
            + ["--search-port-base", str(fleet.search_port_base(node))]
            + flags_from(fleet.gen_flags(node))
        )

        if a.print_cmd:
            print(" ".join(shlex.quote(c) for c in cmd + ["--out", out]))
            continue

        # --check regenerates with the bad-input block the file on disk carries. The block is
        # live data (blanked by _drop_live), but the GNSS branch's element lists are derived
        # from it at generation (stations, channel-noise inputs, shared-ref zeros), so with
        # bffs's current list instead every bffs change would read as drift: --check would
        # fail fleet-wide and eop_cron.sh would stop refreshing the on-disk EOP tables.
        tmp_bi = None
        if a.check and os.path.exists(out):
            blk = _bad_input_block_of(open(out).read())
            if blk is not None:
                fd, tmp_bi = tempfile.mkstemp(
                    prefix="gen_fleet_bad_inputs_", suffix=".json"
                )
                with os.fdopen(fd, "w") as f:
                    json.dump(blk, f)
                cmd = cmd + ["--bad-inputs-from", tmp_bi]

        # No --out: take the config on stdout so --check never touches the tree. The
        # generator writes its human summary to stderr, so stdout is the file exactly.
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=K)
        finally:
            if tmp_bi:
                os.unlink(tmp_bi)
        if p.returncode != 0:
            # Exit 2, not 1: exit 1 means the config differs from the manifest, and
            # node_up.sh's preflight reads it that way. A generator that failed to run has
            # said nothing about the config.
            sys.stderr.write(p.stderr[-3000:] + "\n")
            sys.stderr.write(
                "generator failed for %s (exit %d)\n" % (node, p.returncode)
            )
            raise SystemExit(2)
        _relay_notes(node, p.stderr)
        text = p.stdout
        regen[node] = text

        if not a.check:
            undeclared = parity(text)
            missing = _live_inputs_missing(text)
            if undeclared or missing:
                for kind, path, detail, _ in undeclared:
                    sys.stderr.write("  !!  %s %s  %s\n" % (kind, path, detail))
                sys.exit(
                    "NOT WRITTEN: %s -- %s"
                    % (
                        node,
                        missing
                        or "%d undeclared difference(s) from the stock render "
                        "(scripts/gnss/stock_parity.py; declare one there only with a reason)"
                        % len(undeclared),
                    )
                )
            # Atomic (a sibling .tmp, then os.replace): a concurrent reader (node_up.sh's
            # preflight, another --check, kotekan starting) must never see a truncated config,
            # which reads as a difference or fails to load.
            tmp = out + ".tmp"
            with open(tmp, "w") as f:
                f.write(text)
            os.replace(tmp, out)
            written.append(os.path.relpath(out, K))
            continue

        if not os.path.exists(out):
            print("MISSING  %s" % os.path.relpath(out, K))
            bad.append(node)
            continue
        have = open(out).read()

        undeclared = parity(have)
        if undeclared:
            bad.append(node + ":stock")
            print(
                "STOCK    %s  %d undeclared difference(s) from the stock render:"
                % (os.path.relpath(out, K), len(undeclared))
            )
            for kind, path, detail, _ in undeclared[:12]:
                print("    !!  %s %s  %s" % (kind, path, detail))
        # The EOP table and bad-input block are live data that move on their own: blank them on
        # both sides (_drop_live), leave the interface stamp out (_drop_stamp), and keep
        # everything else byte-exact.
        have_s, text_s = _drop_stamp(have), _drop_stamp(text)
        have_c, text_c = _drop_live(have_s), _drop_live(text_s)
        if have_c == text_c and have != text:
            why = []
            if _stamp(have) != _stamp(text):
                why.append(
                    "the gnss_iface stamp%s"
                    % (" (none on disk)" if _stamp(have) is None else "")
                )
            if have_s != text_s:
                why.append("live data: EOP table / bad inputs")
            print(
                "ok*      %s  (differs only in %s)"
                % (os.path.relpath(out, K), "; ".join(why))
            )
            continue
        if have == text:
            print("ok       %s" % os.path.relpath(out, K))
        else:
            bad.append(node)
            print("DIFFERS  %s" % os.path.relpath(out, K))
            if _stamp(have) != _stamp(text):
                print("    (the gnss_iface stamp differs too; not in the diff below)")
            # The diff of what the verdict compared: without the stamp, which an unstamped
            # config on disk would otherwise open with and count in its line total.
            d = list(
                difflib.unified_diff(
                    have_s.splitlines(),
                    text_s.splitlines(),
                    "committed",
                    "regenerated",
                    lineterm="",
                    n=1,
                )
            )
            print("\n".join("    " + l for l in d[:40]))
            if len(d) > 40:
                print("    ... %d more diff lines" % (len(d) - 40))

    # The search aggregator, from the same fleet list. Not a node config: no stock parity and
    # no live values, so it compares byte for byte (the interface stamp aside).
    agg = aggregator_cmd(man, fleet, a.preview_dir or OUTDIR) if not a.node else None
    if agg:
        cmd, out = agg
        if a.print_cmd:
            print(" ".join(shlex.quote(c) for c in cmd + ["--out", out]))
        else:
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=K)
            if p.returncode != 0:
                sys.stderr.write(p.stderr[-3000:] + "\n")
                sys.stderr.write(
                    "generator failed for the aggregator (exit %d)\n" % p.returncode
                )
                raise SystemExit(2)
            _relay_notes("aggregator", p.stderr)
            text = p.stdout
            rel = os.path.relpath(out, K)
            if not a.check:
                tmp = out + ".tmp"
                with open(tmp, "w") as f:
                    f.write(text)
                os.replace(tmp, out)
                written.append(rel)
            elif not os.path.exists(out):
                print("MISSING  %s" % rel)
                bad.append("aggregator")
            else:
                # The body only: the aggregator's header is a description (its flags live in
                # the manifest), and a header-only change must not hold eop_cron.sh's regen,
                # which waits for a passing check.
                have_b = _split_header(open(out).read())[1]
                text_b = _split_header(text)[1]
                if have_b == text_b:
                    print("ok       %s" % rel)
                else:
                    bad.append("aggregator")
                    print("DIFFERS  %s" % rel)
                    d = list(
                        difflib.unified_diff(
                            have_b.splitlines(),
                            text_b.splitlines(),
                            "committed",
                            "regenerated",
                            lineterm="",
                            n=1,
                        )
                    )
                    print("\n".join("    " + l for l in d[:40]))

    if a.print_cmd:
        return

    # The chains yaml against the fleet, and the derivation against what the generator built:
    # the broker's endpoint lists come from gnss_fleet's covering rule, the stages from the
    # generator's, and a GPU on one list and not the other is a chain polled at a 404 or a
    # chain nobody reads.
    problems = list(comb_problems)
    if fleet_checks:
        problems += check_chains(fleet, chains) + check_instances(fleet, regen)
    else:
        sys.stderr.write("no %s beside the manifest: fleet checks skipped\n" % chains)
    # A role-off node's config goes, so nothing that lists the generated configs (node_up.sh,
    # elem_shared_ref.py, cubearch_up.sh) still finds it; --check names one left behind.
    if not a.node:
        for host in fleet.slots:
            if fleet.roles[host] != "off":
                continue
            stale = os.path.join(
                a.preview_dir or OUTDIR, "chord_gnss_%s%s.yaml" % (host, suffix)
            )
            if not os.path.exists(stale):
                continue
            if a.check:
                problems.append(
                    "%s is role off but %s is still there (a write removes it)"
                    % (host, os.path.relpath(stale, K))
                )
            else:
                os.remove(stale)
                written.append("removed %s (role off)" % os.path.relpath(stale, K))
    for line in problems:
        print("FLEET    %s" % line)
    if problems:
        bad.append("fleet")

    if a.check:
        if bad:
            sys.exit(
                "\n%d of %d node config(s) do not match the manifest: %s\n"
                "Either the file was hand-edited (regenerate: drop --check), or the\n"
                "generator's output moved (regenerate and review the diff), or the base\n"
                "changed underneath. All three are things to decide, not to ignore."
                % (len(bad), len(nodes), ", ".join(bad))
            )
        print(
            "\nALL %d node configs%s match the manifest; the chains yaml follows the fleet."
            % (len(nodes), " and the aggregator's" if agg else "")
        )
    else:
        print("\nwrote %d config(s):\n  %s" % (len(written), "\n  ".join(written)))
        if problems:
            sys.exit(
                "\nthe chains yaml does not follow the fleet list (above): the configs are "
                "written, but the broker would not poll what they build"
            )
        print(
            "\nNothing restarts on its own -- `scripts/gnss/site/node_up.sh <node> restart`."
        )


if __name__ == "__main__":
    # Exit 1 means a config differs from the manifest (node_up.sh's preflight reads it that
    # way), so a crash here, or no kotekan tree to render the base from, which has compared
    # nothing, exits 2.
    try:
        sys.exit(main())
    except kotekan_src.NotFound as e:
        sys.stderr.write("%s\n" % e)
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
