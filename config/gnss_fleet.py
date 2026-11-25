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

The chains are the chains table, `chains:` in the same manifest (config/gnss_chain_table.py).

Derived from the list and nothing else:
  * each node's search-port-base (gen_fleet.py passes it to the generator);
  * the search aggregator's feeds (gen_chord_gnss_config.py --aggregator-instance);
  * every chain's node endpoints in config/gnss_chains_chord.yaml, where a key whose value is
    `fleet` expands at load (expand_chains: broker_multi.py, iface check-fleet) to one URL per
    node GPU that holds a covering channel of the chain's signal: the same covering rule the
    generator builds the chain with (chord_band_plan.covering_channels), so a GPU without the
    chain is never listed (L2C, L2OC) and gen_fleet.py --check compares the two;
  * the site scripts' host lists (`gnss_fleet.py hosts`, or hosts() from Python);
  * with the manifest's host flags, the GPU halves' hosts that serve their rows by chain
    (host_bases), the chains yaml's `node-hosts: fleet`;
  * a chains-table row's `dll-spacing-chips: fleet` (L2C's): the E/L spacing from the number
    of the chain's channels the fleet holds, by the same covering rule (Fleet.dll_spacing),
    one value for the node taps (gen_chord_gnss_config.py) and the broker's trim law
    (load_chains).

    python3 config/gnss_fleet.py hosts            # the fleet's hosts, space-separated
    python3 config/gnss_fleet.py hosts --stock    # the stock reference hosts
    python3 config/gnss_fleet.py table            # slot, port, combs and chains per host
    python3 config/gnss_fleet.py expand [CHAINS]  # the chains yaml's `fleet` keys, expanded

Python 3 standard library + PyYAML only, so the site scripts can use it from /usr/bin/python3;
the covering-channel computation (instances, expand_chains) and stage_chain import
chord_band_plan lazily.
"""
import argparse
import functools
import operator
import os
import re
import sys

import yaml

import gnss_chain_table

CONF = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(CONF, "gnss_fleet_chord.yaml")
NODE_FILE = os.path.join(CONF, "chord_gnss_node.yaml")
CHAINS_YAML = os.path.join(CONF, "gnss_chains_chord.yaml")

#: The aggregator's first feed port: slot k's GPU g sends to SEARCH_PORT_BASE + 2k + g.
SEARCH_PORT_BASE = 11040
#: Manifest node keys that are fleet facts, not generator flags (gen_fleet strips them).
FLEET_KEYS = ("role",)
#: Manifest `common:` keys read by site services, not the generator (gen_fleet strips them):
#: the lane relay's transfer tables and gate (scripts/gnss/site/lane_relay.py).
SITE_KEYS = (
    "fleet-lane-transfer",
    "fleet-lane-transfer-bright",
    "fleet-lane-transfer-bright-db",
)
#: role values. A node without `role` is "gnss".
ROLES = ("gnss", "off")
#: The value that asks for a derived endpoint list in the chains yaml (and, in a chains-table
#: row, a derived dll-spacing-chips: gnss_chain_table.FLEET, the same word).
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
#: The chains-yaml key that may be `fleet` in `common:`, expanded to the bases of every GPU half's
#: hosts that serve their rows' verbs by chain (Fleet.host_bases), for the broker's --node-hosts.
HOSTS_KEY = "node-hosts"
#: The generator flags that host a per-chain stage's rows in one stage per GPU half, the stage,
#: and the host's name, gnss<g>_<host> (gen_chord_gnss_config.py inject_host_form,
#: record_host_form, combine_host_form). A host serves its rows' verbs by chain at
#: /gnss<g>_<host>/<verb>.
HOST_FLAGS = {
    "inject-host": ("inject", "inject"),
    "record-host": ("n2assemble", "records"),
    "combine-host": ("n2combine", "combine"),
}


#: Where a derived (`fleet`) DLL spacing puts the E/L taps on the fleet's coherent sum, as a
#: fraction of its first null. Each instance despreads one channel of width B; the fleet loop
#: sums the N contiguous channels the fleet holds coherently, an N-channel Dirichlet whose first
#: null is 1/(N B) of delay, chip_rate / (N B) chips. The fraction is the one the 09-08 e2e sweep
#: validated on L2C: 0.4 CM chips on five 195.3125 kHz channels (q ~6 on peak, S-curve of the
#: right sign from -0.83 to +0.67 chips, no false lock; |R| ~0.29 of the peak), 0.764 of the
#: null. Taps nearer the null read a q far above ~6 and lock falsely off peak.
DIRICHLET_TAP_FRACTION = 0.4 * 5 * 195312.5 / 511.5e3


def dirichlet_spacing(n, chip_rate_hz, channel_hz):
    """The E/L spacing, in chips at chip_rate_hz, that puts the taps at DIRICHLET_TAP_FRACTION
    of the first null of n contiguous channels of width channel_hz summed coherently, rounded
    to 0.01 chip: 0.33 for L2C's six, 0.4 for five, 0.5 for four."""
    return round(DIRICHLET_TAP_FRACTION * chip_rate_hz / (n * channel_hz), 2)


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
    """The infix of a chain's buffer, stage and endpoint names: "_" and its broker chain key,
    e.g. GAL_E5A_Q_CS -> _gal_e5a (gnss0_gal_e5a_n2assemble).

    Must be distinct across the chains on one node: every buffer, stage and REST endpoint of
    the chain is named from it, and kotekan keys its per-consumer bookkeeping by stage name.
    """
    return "_" + broker_chain_name(name)


_STAGE_CHAINS = {}  # tag -> broker chain, filled by stage_chain's first call
# Chains named by their broker key from the start: no band-tagged name is theirs.
_NO_BAND_TAG = {"qzss_l5"}

#: The per-chain stages whose untagged name, gnss<g>_<stage>, was the primary's before every
#: chain was tagged. No host takes one of these names; a host's (gnss<g>_inject, _records,
#: _combine, _telem, _telem_send, _cube_send) names no chain.
UNTAGGED_STAGES = ("n2assemble", "n2combine", "n2ctl", "telem_pack")


def stage_chain(name, stage):
    """The broker chain of a per-chain stage `gnss<g><tag>_<stage>`, from its name alone or
    as it sits in a REST prefix or lane owner (`/<name>`), a cube sender (`<host>//<name>`) or
    a cube file (`<host>_<name>`): gnss0_gal_e5a_n2assemble -> gal_e5a. None when the name is
    not that stage's or names no chain.

    The tag is a broker chain key. Names generated before it was are read through two
    allowlists: a band tag is the other chain on that band (gnss0_e5a_n2assemble -> gal_e5a,
    gnss0_l5_n2assemble -> sbas_l5), and no tag is the primary (the node table's
    signals.primary), for the stages in UNTAGGED_STAGES only.
    """
    m = re.match(
        r"(?:/|[a-z][a-z0-9.-]*//?|[a-z][a-z0-9.-]*_)?gnss\d+(?:_([a-z0-9_]+))?_%s$"
        % re.escape(stage),
        name,
    )
    if not m or (m.group(1) is None and stage not in UNTAGGED_STAGES):
        return None
    if not _STAGE_CHAINS:
        # Lazy: chord_band_plan imports the interface golden (gnss_broker.iface).
        if CONF not in sys.path:
            sys.path.insert(0, CONF)
        from chord_band_plan import signal_table

        prim = broker_chain_name(_load_yaml(NODE_FILE)["signals"]["primary"])
        chains = {broker_chain_name(n) for n in signal_table()}
        bands = {}
        for c in chains - {prim} - _NO_BAND_TAG:
            bands.setdefault(c.partition("_")[2], []).append(c)
        _STAGE_CHAINS.update({b: cs[0] for b, cs in bands.items() if len(cs) == 1})
        _STAGE_CHAINS.update({c: c for c in chains})
        _STAGE_CHAINS[""] = prim
    return _STAGE_CHAINS.get(m.group(1) or "")


def host_rows(cfg, kotekan_stage, suffix):
    """[(name, block)] of the stages of type kotekan_stage in a parsed node config, by name. A
    top-level block is one, named by its key, unless it hosts rows (`chains:`): then each row is
    one, named `<row key><suffix>` as the host names it, which is the stage it stands for. A
    row's block is the host's keys overlaid with its own, with its worker's cpu_affinity where
    the host has `workers`: the block of that stage."""
    out = []
    for k, v in (cfg or {}).items():
        if not isinstance(v, dict) or v.get("kotekan_stage") != kotekan_stage:
            continue
        if "chains" not in v:
            out.append((k, v))
            continue
        host = {h: x for h, x in v.items() if h not in ("chains", "workers")}
        cpus = {r: w["cpu_affinity"] for w in v.get("workers") or [] for r in w["rows"]}
        for key, row in v["chains"].items():
            block = dict(host, **row)
            if key in cpus:
                block["cpu_affinity"] = cpus[key]
            out.append((key + suffix, block))
    return sorted(out, key=operator.itemgetter(0))


def assembler_rows(cfg):
    """host_rows of the record assemblers (GnssGpuRecordAssemble), a row of gnss<g>_records
    (gen_chord_gnss_config.py --record-host) named `<row key>_n2assemble`, its REST prefix, lane
    owner and cube sender."""
    return host_rows(cfg, "GnssGpuRecordAssemble", "_n2assemble")


def config_host_bases(cfg, origin):
    """<origin>/<host> of the hosts in a parsed node config that serve their rows' verbs by
    chain: gnss<g>_records and gnss<g>_combine, and the gnss<g>_inject of each GPU half whose
    cudaGnssInject hosts the half's injectors (--inject-host), as the manifest's host flags
    give them (Fleet.host_bases)."""
    out = set()
    for k, v in (cfg or {}).items():
        if not isinstance(v, dict):
            continue
        if "chains" in v and v.get("kotekan_stage") in (
            "GnssGpuRecordAssemble",
            "GnssCoherentCombiner",
        ):
            out.add("%s/%s" % (origin, k))
        for c in v.get("commands") or []:
            if (
                isinstance(c, dict)
                and c.get("name") == "cudaGnssInject"
                and "chains" in c
            ):
                for r in c["chains"]:
                    out.add("%s/%s_inject" % (origin, r.split("_", 1)[0]))
    return sorted(out)


def combiner_rows(cfg):
    """host_rows of the combiners (GnssCoherentCombiner), a row of gnss<g>_combine
    (gen_chord_gnss_config.py --combine-host) named `<row key>_n2combine`, its REST prefix."""
    return host_rows(cfg, "GnssCoherentCombiner", "_n2combine")


def telem_rows(cfg):
    """host_rows of the telemetry packs (GnssTelemPack), a row of gnss<g>_telem
    (gen_chord_gnss_config.py --telem-pack-host) named `<row key>_telem_pack`."""
    return host_rows(cfg, "GnssTelemPack", "_telem_pack")


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
        try:
            self.chains = gnss_chain_table.rows(self.man)
        except gnss_chain_table.TableError as e:
            raise FleetError("%s: %s" % (self.manifest_path, e))
        if not self.chains:
            raise FleetError("%s: no chains" % self.manifest_path)
        prim = broker_chain_name(self.site["signals"]["primary"])
        if self.chains[0].key != prim:
            raise FleetError(
                "%s: the first chains row is %s, but the node table's primary is chain %s"
                % (self.manifest_path, self.chains[0].key, prim)
            )
        if "extra-signal" in (self.man.get("common") or {}):
            raise FleetError(
                "%s: common sets extra-signal; the chains are the rows of chains: "
                "(config/gnss_chain_table.py), so move it there" % self.manifest_path
            )
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

    def hosted(self, host):
        """The stages whose rows one stage per GPU half hosts on this node, as the generator
        resolves the manifest: a host flag set true in `common:` or the node's own entry (a
        store_true flag, so neither can unset the other)."""
        common = self.man.get("common") or {}
        node = self.gen_flags(host)
        return sorted(
            stage
            for flag, (stage, _) in HOST_FLAGS.items()
            if common.get(flag) is True or node.get(flag) is True
        )

    def host_bases(self):
        """http://<node>:<port>/gnss<g>_<host> of every host the fleet's nodes run: per node, each
        GPU half that builds a chain and each stage hosted() names."""
        names = {stage: name for stage, name in HOST_FLAGS.values()}
        out = []
        for host in self.hosts:
            stages = self.hosted(host)
            if not stages:
                continue
            gpus = sorted(
                {
                    g
                    for c in self.tracker_signals()
                    for h, g in self.instances(c)
                    if h == host
                }
            )
            out += [
                "%s/gnss%d_%s" % (self.rest_url(host), g, names[s])
                for g in gpus
                for s in stages
            ]
        return out

    # ---- chains ----------------------------------------------------------------------------
    def tracker_signals(self):
        """{broker chain: (descriptor, tag)} for every chain the nodes build: the rows of the
        chains table, the primary first."""
        return {r.key: (r.descriptor, signal_tag(r.descriptor)) for r in self.chains}

    def instances(self, chain):
        """[(host, gpu)] of the chain's instances, slot-major: every fleet GPU holding at least
        one of the chain's covering channels, by the rule the generator builds it with."""
        row = next((r for r in self.chains if r.key == chain), None)
        if row is None:
            raise FleetError(
                "chain %s: no node builds it (chains: in %s: %s)"
                % (chain, self.manifest_path, ", ".join(r.key for r in self.chains))
            )
        out = []
        for host in self.hosts:
            for gpu in self._chain_gpus(host, row):
                out.append((host, gpu))
        return out

    def _chain_gpus(self, host, row):
        offs = self.offsets(host)
        return sorted({offs.index(f % 16) for f in self._chain_channels(host, row)})

    def _chain_channels(self, host, row):
        """The freq_ids of the chain's covering channels the host holds, by the rule the
        generator builds the chain with."""
        # Lazy: chord_band_plan imports the interface golden (gnss_broker.iface).
        if CONF not in sys.path:
            sys.path.insert(0, CONF)
        from chord_band_plan import (
            chain_covering_channels,
            covering_channels,
            node_channels,
            parse_slots,
            signal_table,
        )

        sig = self.site["signals"]
        doppler = float(sig["max_doppler_hz"])
        chans = node_channels(self.site, host)
        if row.primary:
            # As gen_chord_gnss_config.py: the node table's carrier and chip rate.
            cov = covering_channels(
                chans, float(sig["carrier_hz"]), float(sig["chip_rate_hz"]), doppler
            )
        else:
            # As gen_chord_gnss_config.py: an FDMA chain covers the union of its slots' lobes.
            ks = parse_slots(row.prns)[1]
            cov = chain_covering_channels(
                chans, signal_table()[row.descriptor], doppler, ks
            )
        return cov

    def chain_channels(self, row):
        """The freq_ids of the chain's covering channels the fleet's hosts hold, ascending."""
        return sorted({f for h in self.hosts for f in self._chain_channels(h, row)})

    def dll_spacing(self, row):
        """The chain's E/L tap spacing in its own chips: its row's number, or for a row that
        says `fleet` the spacing derived from the fleet, dirichlet_spacing over the chain's
        channels the fleet holds. A derived spacing needs those channels contiguous (a gap
        makes the sum something other than a Dirichlet) and at least two of them, and is not
        for an FDMA chain, whose slots each see their own set; each is refused here."""
        if row.dll_spacing_chips != gnss_chain_table.FLEET:
            return row.dll_spacing_chips
        if CONF not in sys.path:
            sys.path.insert(0, CONF)
        from chord_band_plan import parse_slots, signal_table

        at = "chains table row %s: dll-spacing-chips %s" % (row.key, FLEET)
        if parse_slots(row.prns)[1]:
            raise FleetError(
                "%s: an FDMA chain's slots each sum their own channels, so no one count "
                "sets its spacing; write a number" % at
            )
        fids = self.chain_channels(row)
        if len(fids) < 2:
            raise FleetError(
                "%s: the fleet (%s) holds %s of its channels, and the fleet loop needs two"
                % (at, " ".join(self.hosts), len(fids) or "none")
            )
        if fids != list(range(fids[0], fids[-1] + 1)):
            raise FleetError(
                "%s: the fleet's channels of it, %s, are not contiguous, so their sum is no "
                "Dirichlet to derive a spacing from; write a number (an e2e sweep picks it) "
                "until the gap closes" % (at, " ".join(map(str, fids)))
            )
        return dirichlet_spacing(
            len(fids),
            signal_table()[row.descriptor]["chip_rate_hz"],
            float(self.site["fengine"]["channel_width_hz"]),
        )

    def table(self):
        """The chains table with every `fleet` dll-spacing-chips resolved (dll_spacing): the
        rows the node taps and the broker's trim law take their spacing from. Raises
        FleetError when a `fleet` row cannot be derived from this node list."""
        return [r._replace(dll_spacing_chips=self.dll_spacing(r)) for r in self.chains]

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
                inst = self.instances(self.chains[0].key)
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

    if not (
        wants(common)
        or common.get(HOSTS_KEY) == FLEET
        or any(wants(c) for c in chains.values())
    ):
        return cfg
    if fleet is None:
        fleet = Fleet()
    elif callable(fleet):
        fleet = fleet()
    if common.get("rest-url") == FLEET:
        common["rest-url"] = fleet.rest_url(fleet.hosts[0])
    if common.get(HOSTS_KEY) == FLEET:
        # None, so the broker gets no --node-hosts, when no node hosts a stage.
        common[HOSTS_KEY] = ",".join(fleet.host_bases()) or None
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


def table_keys(cfg, table, where="the chains table", fleet=None):
    """A parsed chains yaml with each chain's dll-spacing set from its row of the chains table
    (the input is not modified): the spacing the node generator puts on the chain's taps, so
    the trim law and the taps cannot disagree. A row that says `fleet` takes the spacing
    `fleet` (a Fleet, or a callable that builds one when a chain needs it) derives
    (Fleet.dll_spacing), as the generator does; without `fleet` it is refused. A dll-spacing
    written in the yaml is refused, as is a chain without a row (no node builds it)."""
    cfg = dict(cfg or {})
    if "dll-spacing" in (cfg.get("common") or {}):
        raise FleetError(
            "common sets dll-spacing; each chain's comes from its row in %s" % where
        )
    rows = {r.key: r for r in table}
    common_signal = (cfg.get("common") or {}).get("signal")
    chains = {}
    for name, spec in (cfg.get("chains") or {}).items():
        spec = dict(spec or {})
        if "dll-spacing" in spec:
            raise FleetError(
                "chains.%s sets dll-spacing; it comes from the chain's row in %s"
                % (name, where)
            )
        key = str(spec.get("signal", common_signal or name))
        if key not in rows:
            raise FleetError("chains.%s: %s has no row in %s" % (name, key, where))
        spacing = rows[key].dll_spacing_chips
        if spacing == gnss_chain_table.FLEET:
            if fleet is None:
                raise FleetError(
                    "chains.%s: %s's row in %s says dll-spacing-chips %s, and no fleet list "
                    "was given to derive it from" % (name, key, where, FLEET)
                )
            if not isinstance(fleet, Fleet):
                fleet = fleet()
            spacing = fleet.dll_spacing(rows[key])
        spec["dll-spacing"] = spacing
        chains[name] = spec
    cfg["chains"] = chains
    return cfg


def load_chains(path=None, fleet=None):
    """The chains yaml at `path`, parsed, expanded, and each chain's dll-spacing taken from the
    chains table (table_keys), a `fleet` row's derived from the fleet (Fleet.dll_spacing).
    Without `fleet`, the fleet list, the chains table and the node table are the ones beside
    the chains file (a checkout's config/), else this module's."""
    path = path or CHAINS_YAML
    cfg = _load_yaml(path)
    d = os.path.dirname(os.path.abspath(path))
    man = os.path.join(d, os.path.basename(MANIFEST))
    if fleet is None:
        nodes = os.path.join(d, os.path.basename(NODE_FILE))
        if os.path.exists(man) and os.path.exists(nodes):
            fleet = lambda: Fleet(man, nodes)  # noqa: E731  (built only if needed)
    if isinstance(fleet, Fleet):
        table, man = fleet.chains, fleet.manifest_path
    else:
        man = man if os.path.exists(man) else MANIFEST
        try:
            table = gnss_chain_table.load(man)
        except gnss_chain_table.TableError as e:
            raise FleetError("%s: %s" % (man, e))
    cfg = expand_chains(cfg, fleet)
    if fleet is None:
        # A `fleet` row's spacing: from the fleet list the table was read from.
        fleet = functools.partial(Fleet, man)
    return table_keys(cfg, table, "%s chains:" % man, fleet)


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
        for r in fl.chains:
            if r.dll_spacing_chips != gnss_chain_table.FLEET:
                continue
            try:
                d = fl.dll_spacing(r)
            except FleetError as err:
                print("dll spacing %s: %s" % (r.key, err))
                continue
            fids = fl.chain_channels(r)
            print(
                "dll spacing %s: %s chips, from the fleet's %d channels %d-%d"
                % (r.key, d, len(fids), fids[0], fids[-1])
            )
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
