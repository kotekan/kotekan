#!/usr/bin/env python3
"""The GNSS fleet: which hosts run the GNSS branch, in one list, and everything derived from it.

The list is `nodes:` in config/gnss_fleet_chord.yaml. Its ORDER is each host's slot: the
search aggregator listens for slot k's GPU g on SEARCH_PORT_BASE + 2k + g, and that host's
config sends there. A node that leaves stays in the list with `role: off`, so it keeps its slot
and no other node's port moves; a node that joins is appended. The site facts about every host,
its two combs (gpu_offsets_mod16), live in config/chord_gnss_node.yaml `nodes:`, which also lists
hosts outside the fleet: those run production's stock kotekan and are the stock references the
generator asks for the EOP table and the bad-input block (a role-off node is given back to
production, so it is one too).

Derived from the list and nothing else:
  * each node's search-port-base (gen_fleet.py passes it to the generator);
  * the search aggregator's feeds (gen_chord_gnss_config.py --aggregator-instance);
  * every chain's node endpoints in config/gnss_chains_chord.yaml, where a key whose value is
    `fleet` expands at load (expand_chains: broker_multi.py, iface check-fleet) to one URL per
    node GPU that holds a covering channel of the chain's signal: the same covering rule the
    generator builds the chain with (chord_band_plan.covering_channels), so a GPU without the
    chain is never listed (L2C, L2OC) and gen_fleet.py --check compares the two;
  * the site scripts' host lists (`gnss_fleet.py hosts`, or hosts() from Python).

    python3 config/gnss_fleet.py hosts            # the fleet's hosts, space-separated
    python3 config/gnss_fleet.py hosts --stock    # the stock reference hosts
    python3 config/gnss_fleet.py table            # slot, port, combs and chains per host
    python3 config/gnss_fleet.py expand [CHAINS]  # the chains yaml's `fleet` keys, expanded

Python 3 standard library + PyYAML only, so the site scripts can use it from /usr/bin/python3;
the covering-channel computation (instances, expand_chains) imports chord_band_plan lazily.
"""
import argparse
import os
import sys

import yaml

CONF = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(CONF, "gnss_fleet_chord.yaml")
NODE_FILE = os.path.join(CONF, "chord_gnss_node.yaml")
CHAINS_YAML = os.path.join(CONF, "gnss_chains_chord.yaml")

#: The aggregator's first feed port: slot k's GPU g sends to SEARCH_PORT_BASE + 2k + g.
SEARCH_PORT_BASE = 11040
#: Manifest node keys that are fleet facts, not generator flags (gen_fleet strips them).
FLEET_KEYS = ("role",)
#: role values. A node without `role` is "gnss".
ROLES = ("gnss", "off")
#: The value that asks for a derived endpoint list in the chains yaml.
FLEET = "fleet"
#: Chains-yaml keys that may be `fleet`, and the node stage each one names. rest-url and
#: combiner are single values; the rest are lists over the chain's instances.
URL_KEYS = {
    "trackers": "inject",
    "dll-combiners": "n2combine",
    "spectrum-endpoints": "n2assemble",
    "rf-stats-endpoints": "srch_tap",
}
SINGLE_KEYS = ("rest-url", "combiner")


class FleetError(ValueError):
    """The fleet list is inconsistent: say what and where."""


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def broker_chain_name(name):
    """gnssSignal.hpp name -> the broker's chain key. GPS_L5_Q -> gps_l5, GAL_E5A_Q_CS -> gal_e5a.

    config/gnss_chains_chord.yaml keys its chains the same way (gps_l5, gal_e5a, bds_b2a, ...),
    so the key is derived rather than carried in a table. It goes on the telemetry wire
    (gnssTelem.hpp) as a name, not an integer id: an id would need identical tables in the
    generator, the C++ and the broker, and a drifted one would attribute one constellation's
    data to another with nothing to say so.
    """
    return "_".join(name.split("_")[:2]).lower()


def signal_tag(name):
    """Short buffer/endpoint suffix for an extra signal chain, e.g. GAL_E5A_Q_CS -> _e5a.

    Must be distinct across the chains on one node: every buffer, stage and REST endpoint of
    the chain is named from it, and kotekan keys its per-consumer bookkeeping by stage name.
    """
    parts = name.split("_")
    body = parts[1] if len(parts) > 1 else parts[0]
    return "_" + body.lower()


class Fleet:
    """The fleet list (manifest `nodes:`) joined with the site's node table.

    hosts      the GNSS hosts, in slot order (role off left out)
    slots      {host: slot} for every manifest node, off ones included
    stock      site hosts not in the fleet (role off included): production's nodes, the
               stock references
    """

    def __init__(self, manifest=None, node_file=None):
        self.manifest_path = manifest or MANIFEST
        self.node_file = node_file or NODE_FILE
        self.man = (
            manifest if isinstance(manifest, dict) else _load_yaml(self.manifest_path)
        )
        self.site = (
            node_file if isinstance(node_file, dict) else _load_yaml(self.node_file)
        )
        if isinstance(manifest, dict):
            self.manifest_path = "<manifest>"
        site_nodes = self.site.get("nodes") or {}
        self.site_hosts = [h for h, v in site_nodes.items() if isinstance(v, dict)]
        if "search-port-base" in (self.man.get("common") or {}):
            raise FleetError(
                "%s: common sets search-port-base; each node's is derived from its slot "
                "in nodes: (%d + 2 x slot), so remove it"
                % (self.manifest_path, SEARCH_PORT_BASE)
            )
        nodes = self.man.get("nodes") or {}
        if not isinstance(nodes, dict) or not nodes:
            raise FleetError("%s: no nodes" % self.manifest_path)
        self.slots, self.roles, self.hosts = {}, {}, []
        for k, (host, spec) in enumerate(nodes.items()):
            spec = spec or {}
            role = spec.get("role", "gnss")
            if role not in ROLES:
                raise FleetError(
                    "%s: %s has role %r; one of %s"
                    % (self.manifest_path, host, role, ", ".join(ROLES))
                )
            if host not in self.site_hosts:
                raise FleetError(
                    "%s: %s is not in the site's node table (%s nodes:), which holds each "
                    "host's combs; add it there first"
                    % (self.manifest_path, host, self.node_file)
                )
            # The port is derived from the slot. A hand-written value would be a second
            # copy that can disagree, and a wrong one reads one node's comb as another's.
            if "search-port-base" in spec:
                raise FleetError(
                    "%s: %s sets search-port-base; it is derived from the node's position "
                    "in nodes: (%d + 2 x slot), so remove it"
                    % (self.manifest_path, host, SEARCH_PORT_BASE)
                )
            self.slots[host] = k
            self.roles[host] = role
            if role != "off":
                self.hosts.append(host)
        if not self.hosts:
            raise FleetError("%s: every node is role off" % self.manifest_path)
        self.stock = [h for h in self.site_hosts if h not in self.hosts]
        self.rest_port = int(self.site["runtime"]["rest_port"])

    # ---- per host --------------------------------------------------------------------------
    def search_port_base(self, host):
        return SEARCH_PORT_BASE + 2 * self.slots[host]

    def gen_flags(self, host):
        """The manifest's flags for one node, without the fleet keys."""
        spec = dict((self.man.get("nodes") or {}).get(host) or {})
        for k in FLEET_KEYS:
            spec.pop(k, None)
        return spec

    def rest_url(self, host):
        return "http://%s:%d" % (host, self.rest_port)

    def offsets(self, host):
        return list(self.site["nodes"][host]["gpu_offsets_mod16"])

    # ---- chains ----------------------------------------------------------------------------
    def tracker_signals(self):
        """{broker chain: (descriptor, tag)} for every chain the nodes build: the primary
        (node table signals.tracker, tag '') and each manifest extra-signal."""
        sig = self.site["signals"]
        out = {
            broker_chain_name(sig["primary"]): (sig.get("tracker", sig["primary"]), "")
        }
        for ent in (self.man.get("common") or {}).get("extra-signal") or []:
            name = str(ent).partition(":")[0].strip()
            out[broker_chain_name(name)] = (name, signal_tag(name))
        return out

    def instances(self, chain):
        """[(host, gpu)] of the chain's instances, slot-major: every fleet GPU holding at least
        one of the chain's covering channels, by the rule the generator builds it with."""
        sigs = self.tracker_signals()
        if chain not in sigs:
            raise FleetError(
                "chain %s: no node builds it (the primary is %s; extra-signal in %s: %s)"
                % (
                    chain,
                    self.site["signals"]["primary"],
                    self.manifest_path,
                    ", ".join(sorted(c for c in sigs if sigs[c][1])),
                )
            )
        name, tag = sigs[chain]
        out = []
        for host in self.hosts:
            for gpu in self._chain_gpus(host, name, primary=not tag):
                out.append((host, gpu))
        return out

    def _chain_gpus(self, host, name, primary):
        # Lazy: chord_band_plan imports the interface golden (gnss_broker.iface).
        if CONF not in sys.path:
            sys.path.insert(0, CONF)
        from chord_band_plan import covering_channels, node_channels, signal_table

        sig = self.site["signals"]
        doppler = float(sig["max_doppler_hz"])
        chans = node_channels(self.site, host)
        if primary:
            # As gen_chord_gnss_config.py: the node table's carrier and chip rate.
            cov = covering_channels(
                chans, float(sig["carrier_hz"]), float(sig["chip_rate_hz"]), doppler
            )
        else:
            s = signal_table()[name]
            cov = covering_channels(
                chans,
                s["carrier_hz"],
                s["chip_rate_hz"],
                doppler,
                modulation=s["modulation"],
                boc_m=s["boc_m"],
                boc_n=s["boc_n"],
            )
        offs = self.offsets(host)
        return sorted({offs.index(f % 16) for f in cov})

    def stage_url(self, host, gpu, chain, stage):
        tag = "" if stage == "srch_tap" else self.tracker_signals()[chain][1]
        return "%s/gnss%d%s_%s" % (self.rest_url(host), gpu, tag, stage)

    def expand_chain(self, chain, merged):
        """{key: value} for every `fleet` key of one chain's merged block (common overlaid by
        its own), expanded. Keys not set to `fleet` are left to the caller as written."""
        out = {}
        if any(
            merged.get(k) == FLEET for k in URL_KEYS if k != "rf-stats-endpoints"
        ) or (merged.get("combiner") == FLEET):
            if not self.instances(chain):
                raise FleetError(
                    "chain %s: no GPU of the fleet (%s) holds a covering channel of its "
                    "signal, so it would poll nothing; take the chain out of the chains yaml "
                    "or add a node whose combs hold it" % (chain, " ".join(self.hosts))
                )
        rest_host = self.hosts[0]
        if merged.get("rest-url") == FLEET:
            out["rest-url"] = self.rest_url(rest_host)
        for key, stage in URL_KEYS.items():
            if merged.get(key) != FLEET:
                continue
            if stage == "srch_tap":
                # The search taps: one per GPU holding the primary's covering channels.
                prim = broker_chain_name(self.site["signals"]["primary"])
                inst = self.instances(prim)
            else:
                inst = self.instances(chain)
            out[key] = ",".join(self.stage_url(h, g, chain, stage) for h, g in inst)
        if merged.get("combiner") == FLEET:
            # Bare on rest-url when that host holds the chain (the first choice the broker
            # tries for bare names), else the first instance's URL.
            inst = self.instances(chain)
            tag = self.tracker_signals()[chain][1]
            on_rest = [g for h, g in inst if h == rest_host]
            out["combiner"] = (
                "gnss%d%s_n2combine" % (on_rest[0], tag)
                if on_rest
                else self.stage_url(inst[0][0], inst[0][1], chain, "n2combine")
            )
        return out


def expand_chains(cfg, fleet=None):
    """A parsed chains yaml with every `fleet` value replaced (the input is not modified).

    A key may say `fleet` in `common:` or in a chain's block; either way the expansion is per
    chain and lands in the chain's block (rest-url, chain-independent, also in common), so a
    reader that overlays common with the chain's block, as every reader does, sees URLs."""
    cfg = dict(cfg or {})
    common = dict(cfg.get("common") or {})
    chains = {k: dict(v or {}) for k, v in (cfg.get("chains") or {}).items()}
    keys = set(URL_KEYS) | set(SINGLE_KEYS)

    def wants(d):
        return any(d.get(k) == FLEET for k in keys)

    if not (wants(common) or any(wants(c) for c in chains.values())):
        return cfg
    if fleet is None:
        fleet = Fleet()
    elif callable(fleet):
        fleet = fleet()
    if common.get("rest-url") == FLEET:
        common["rest-url"] = fleet.rest_url(fleet.hosts[0])
    for name, spec in chains.items():
        merged = dict(common)
        merged.update(spec)
        if not wants(merged):
            continue
        sig = merged.get("signal", name)
        spec.update(fleet.expand_chain(str(sig), merged))
    cfg["common"] = common
    cfg["chains"] = chains
    return cfg


def load_chains(path=None, fleet=None):
    """The chains yaml at `path`, parsed and expanded. Without `fleet`, the fleet list and the
    node table are the ones beside the chains file (a checkout's config/), else this module's."""
    path = path or CHAINS_YAML
    cfg = _load_yaml(path)
    if fleet is None:
        d = os.path.dirname(os.path.abspath(path))
        man = os.path.join(d, os.path.basename(MANIFEST))
        nodes = os.path.join(d, os.path.basename(NODE_FILE))
        if os.path.exists(man) and os.path.exists(nodes):
            fleet = lambda: Fleet(man, nodes)  # noqa: E731  (built only if needed)
    return expand_chains(cfg, fleet)


def live_comb(host, port, science_band, timeout=4.0):
    """[GPU0's comb, GPU1's comb] (freq_id mod 16) as the host's frames label them, or None.

    Read from the newest frame of each GPU's n2_buffer_<g> (/buffer_frame), whose chord
    metadata carries the frame's freq_id: every node has those buffers, production's stock
    nodes and ours alike, so a host can be read before it joins the fleet, and the label holds
    while the F-engine sends zeros. None when the node does not answer or a frame carries no
    science-band freq_id (a node still starting)."""
    import json
    import urllib.request

    out = []
    for g in (0, 1):
        url = "http://%s:%d/buffer_frame?name=n2_buffer_%d" % (host, port, g)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                fid = int(json.load(r)["metadata"]["freq_id"])
        except Exception:
            return None
        if not science_band["min_freq_id"] <= fid <= science_band["max_freq_id"]:
            return None
        out.append(fid % 16)
    return out


def comb_report(fleet, hosts, timeout=4.0):
    """[(host, table comb or None, live comb or None)] for each host."""
    site = fleet.site
    port = int(site["runtime"]["rest_port"])
    out = []
    for h in hosts:
        tab = (site["nodes"].get(h) or {}).get("gpu_offsets_mod16")
        out.append(
            (
                h,
                list(tab) if tab else None,
                live_comb(h, port, site["science_band"], timeout),
            )
        )
    return out


def write_combs(node_file, combs):
    """Set `gpu_offsets_mod16` for each {host: [c0, c1]} in the node table file, rewriting only
    that host's line, so the comments and layout stay. A host with no line gets one after the
    last host line of `nodes:`."""
    lines = open(node_file).read().split("\n")
    start = lines.index("nodes:")
    at = {}  # host -> line index, for the `    host: {...}` lines of nodes:
    for i in range(start + 1, len(lines)):
        if lines[i] and not lines[i].startswith(" "):
            break
        key, sep, val = lines[i].strip().partition(":")
        if sep and not key.startswith("#") and val.strip().startswith("{"):
            at[key] = i
    entry = "    %s: {gpu_offsets_mod16: [%d, %d]}"
    for host, comb in combs.items():
        if host in at:
            lines[at[host]] = entry % (host, comb[0], comb[1])
    last = max(at.values()) if at else start
    for host in sorted(set(combs) - set(at)):
        last += 1
        lines.insert(last, entry % (host, combs[host][0], combs[host][1]))
    tmp = node_file + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
    os.replace(tmp, node_file)


def hosts(manifest=None, node_file=None):
    """The fleet's GNSS hosts in slot order (the site scripts' default node list)."""
    return Fleet(manifest, node_file).hosts


def stock_hosts(manifest=None, node_file=None):
    """The site's hosts outside the fleet: production's stock nodes."""
    return Fleet(manifest, node_file).stock


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--node-file", default=NODE_FILE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hosts", help="the fleet's hosts, space-separated, slot order")
    g = h.add_mutually_exclusive_group()
    g.add_argument("--stock", action="store_true", help="the stock reference hosts")
    g.add_argument("--off", action="store_true", help="the role-off hosts")
    sub.add_parser("table", help="slot, search port, combs and chains per host")
    c = sub.add_parser(
        "combs",
        help="read each host's combs from its frames and compare them with the node table",
    )
    c.add_argument(
        "hosts", nargs="*", help="default: every host in the node table and the fleet"
    )
    c.add_argument(
        "--write",
        action="store_true",
        help="set the node table's gpu_offsets_mod16 to what the frames say, for every "
        "host read (a host not in the table is added)",
    )
    e = sub.add_parser("expand", help="print the chains yaml with `fleet` expanded")
    e.add_argument("chains", nargs="?", default=CHAINS_YAML)
    a = ap.parse_args(argv)
    try:
        fl = Fleet(a.manifest, a.node_file)
    except FleetError as err:
        sys.exit(str(err))
    if a.cmd == "hosts":
        if a.stock:
            print(" ".join(fl.stock))
        elif a.off:
            print(" ".join(h for h in fl.slots if fl.roles[h] == "off"))
        else:
            print(" ".join(fl.hosts))
        return 0
    if a.cmd == "table":
        chains = sorted(fl.tracker_signals())
        per = {c: set(fl.instances(c)) for c in chains}
        print(
            "%-6s %4s %6s %-9s %s" % ("host", "slot", "port", "combs", "chains (GPUs)")
        )
        for host in fl.slots:
            if fl.roles[host] == "off":
                print(
                    "%-6s %4d %6d %-9s off"
                    % (
                        host,
                        fl.slots[host],
                        fl.search_port_base(host),
                        fl.offsets(host),
                    )
                )
                continue
            row = []
            for c in chains:
                gpus = sorted(g for hh, g in per[c] if hh == host)
                if gpus != [0, 1]:
                    row.append("%s(%s)" % (c, ",".join(map(str, gpus)) or "-"))
            print(
                "%-6s %4d %6d %-9s all %d%s"
                % (
                    host,
                    fl.slots[host],
                    fl.search_port_base(host),
                    fl.offsets(host),
                    len(chains),
                    (" except " + " ".join(row)) if row else "",
                )
            )
        print("stock: %s" % (" ".join(fl.stock) or "(none)"))
        return 0
    if a.cmd == "combs":
        want = a.hosts or list(dict.fromkeys(fl.site_hosts + list(fl.slots)))
        rows = comb_report(fl, want)
        print("%-6s %-10s %-10s" % ("host", "table", "frames"))
        fix, bad = {}, 0
        for h, tab, live in rows:
            verdict = (
                "unreachable"
                if live is None
                else "ok"
                if live == tab
                else "NOT IN TABLE"
                if tab is None
                else "MISMATCH"
            )
            print("%-6s %-10s %-10s %s" % (h, tab or "-", live or "-", verdict))
            if live is not None and live != tab:
                fix[h] = live
                bad += 1
        if a.write and fix:
            write_combs(a.node_file, fix)
            print(
                "wrote %d host(s) to %s; regenerate (scripts/gnss/gen_fleet.py) and restart "
                "those nodes, the aggregator and the broker" % (len(fix), a.node_file)
            )
            return 0
        return 1 if bad else 0
    if a.cmd == "expand":
        yaml.safe_dump(
            load_chains(a.chains, fl), sys.stdout, sort_keys=False, width=100000
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
