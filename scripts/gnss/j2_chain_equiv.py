#!/usr/bin/env python3
"""Prove the j2 GNSS include reproduces the generator's per-chain blocks, byte for byte.

Jim Mertens' suggestion (2026-08-18): keep the variable GNSS data in its own file and
include it from the main j2, condensing the repetition with loops. This is the gate for
that restructure -- not a demo. It:

  1. extracts the VARIABLE data (per node, per GPU, per chain) from a config the current
     Python generator emitted,
  2. writes it as a j2 vars fragment (config/gnss/gnss_vars_<node>.j2),
  3. renders config/gnss/gnss_chain.j2 against it with kotekan's own jinja settings,
  4. compares every per-chain GNSS block against the generator's, field by field.

WHAT A PASS PROVES, AND WHAT IT DOES NOT. It proves the template carries the invariant
structure correctly and that the vars schema is SUFFICIENT -- if any field I assumed
constant actually varies per chain, the comparison fails and names it. It does NOT prove
the vars themselves are right, because in this bridge they are extracted from generator
output rather than computed independently; that half comes when the generator emits the
vars natively from the node table. Step 1 is deliberately a bridge and is labelled as one
in the emitted file, so nobody mistakes it for the source of truth.

    scripts/gnss/j2_chain_equiv.py config/generated/chord_gnss_cx19_multi.yaml
"""
import argparse
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # scripts/gnss -> repo root
GNSS_DIR = os.path.join(ROOT, "config", "gnss")

# The per-chain block suffixes this include owns. The per-GPU singletons (srch_*, the
# primary telem_*) are a separate group and are not claimed here -- see the README.
SUFFIXES = ("n2dual", "n2assemble_tiles", "n2assemble", "n2combine", "n2sink",
            "telem_pack", "telem_send", "n2ctl_buf", "n2tiles_buf", "n2epl_buf",
            "n2rec_buf", "n2cmb_buf", "telem_buf")
# The PRIMARY chain carries no tag -- gnss0_n2combine rather than gnss0_e5a_n2combine --
# and is otherwise structurally IDENTICAL to a tagged one (verified field by field: same
# keys, same command list, differing only in signal and data). So it is not a special
# case, it is a chain whose tag is the empty string, and the same loop body renders it.
CHAIN_RE = re.compile(r"^gnss([01])_(e5a_|e5b_|b2a_|b2b_)?(.+)$")
# Per-GPU, NOT per-chain: one voltage tap per GPU feeds acquisition for every signal on it.
SEARCH_SUFFIXES = ("srch_tap", "srch_buf", "srch_send")
# ⚠️ ORPHANS, deliberately NOT rendered: gnss{0,1}_cmb_buf are defined in every deployed
# node config and referenced by NOTHING (checked against every string in the config).
# Templating them would launder dead config into the new structure; they are reported
# instead, for the generator to stop emitting.
ORPHANS = ("cmb_buf",)


def extract(cfg, node):
    """Pull the variable data out of a generated config (the bridge; see the docstring)."""
    gpus = {}
    for key, blk in cfg.items():
        m = CHAIN_RE.match(key)
        if not m or m.group(3) not in SUFFIXES:
            continue
        gpu, tag = int(m.group(1)), ("_" + m.group(2)[:-1] if m.group(2) else "")
        g = gpus.setdefault(gpu, {"gpu": gpu, "chains": {}})
        g["chains"].setdefault(tag, {"tag": tag})
    for gpu, g in gpus.items():
        tap = cfg.get("gnss%d_srch_tap" % gpu)
        if tap is not None:
            g["search"] = {"tap_core": tap["cpu_affinity"][0],
                           "send_core": cfg["gnss%d_srch_send" % gpu]["cpu_affinity"][0],
                           "chan_ids": tap["chan_ids"],
                           "element_offset": tap["element_offset"]}
        for tag, c in g["chains"].items():
            pre = "gnss%d%s_" % (gpu, tag)
            dual = cfg[pre + "n2dual"]
            inj = next(x for x in dual["commands"] if x.get("name") == "cudaGnssInject")
            pack = cfg[pre + "telem_pack"]
            c.update(
                chain=pack["chain"], signal=inj["signal"],
                channel_ids=inj["channel_ids"], local_channels=inj["gnss_local_channels"],
                prns=inj["prns"], f_offset_hz=inj["f_offset_hz"],
                cores={"dual": dual["cpu_affinity"][0],
                       "assemble": cfg[pre + "n2assemble"]["cpu_affinity"][0],
                       "tiles": cfg[pre + "n2assemble_tiles"]["cpu_affinity"][0],
                       "combine": cfg[pre + "n2combine"]["cpu_affinity"][0],
                       "sink": cfg[pre + "n2sink"]["cpu_affinity"][0],
                       "telem": pack["cpu_affinity"][0],
                       "send": cfg[pre + "telem_send"]["cpu_affinity"][0]},
                sizes={"ctl": cfg[pre + "n2ctl_buf"]["frame_size"],
                       "tiles": cfg[pre + "n2tiles_buf"]["frame_size"],
                       "epl": cfg[pre + "n2epl_buf"]["frame_size"],
                       "rec": cfg[pre + "n2rec_buf"]["frame_size"],
                       "cmb": cfg[pre + "n2cmb_buf"]["frame_size"],
                       "telem": cfg[pre + "telem_buf"]["frame_size"]})
    # receiver-wide constants, read from the first chain that has them
    any_gpu = gpus[min(gpus)]
    any_c = any_gpu["chains"][min(any_gpu["chains"])]
    pre0 = "gnss%d%s_" % (any_gpu["gpu"], any_c["tag"])
    dual0 = cfg[pre0 + "n2dual"]
    inj0 = next(x for x in dual0["commands"] if x.get("name") == "cudaGnssInject")
    corr0 = next(x for x in dual0["commands"] if x.get("name") == "cudaCorrelatorDual")
    asm0, cmb0, pk0, snk0 = (cfg[pre0 + "n2assemble"], cfg[pre0 + "n2combine"],
                             cfg[pre0 + "telem_pack"], cfg[pre0 + "n2sink"])
    g = {
        "node": node,
        "frame0_utc": repr(inj0["frame0_utc"]),
        "sample_rate_hz": repr(inj0["sample_rate"]),
        "sample_rate_mhz": asm0["sample_rate"],
        "fft_len": inj0["fft_length"], "hops_per_record": inj0["hops_per_record"],
        "num_synth": inj0["num_synth"], "trim_ttl_s": inj0["trim_ttl_s"],
        "carrier_phase_from_ref": str(inj0["carrier_phase_from_ref"]).lower(),
        "carrier_phase_mode": inj0["carrier_phase_mode"],
        "n_live_elements": corr0["num_live_elements"],
        "num_elements": cfg[pre0 + "n2assemble_tiles"]["num_elements"],
        "reference_element": asm0["reference_element"],
        "spectrum_ring_depth": asm0["spectrum_ring_depth"],
        "spectrum_window_samples": asm0["spectrum_window_samples"],
        "integration_length": cmb0["integration_length"],
        "record_export": cmb0["record_export"],
        "deep_rate_min_q": cmb0["deep_rate_min_q"],
        "sky_deep": str(cmb0["sky_deep"]).lower(),
        "max_prn": pk0["max_prn"], "records_per_frame": pk0["records_per_frame"],
        "telem_host": cfg[pre0 + "telem_send"]["server_ip"],
        "telem_port": cfg[pre0 + "telem_send"]["server_port"],
        "sink_dir": snk0["base_dir"],
        "buffer_depth": cfg[pre0 + "n2ctl_buf"]["num_frames"],
        "telem_frames": cfg[pre0 + "telem_buf"]["num_frames"],
        "search_host": cfg["gnss0_srch_send"]["server_ip"],
        "search_port_base": cfg["gnss0_srch_send"]["server_port"],
        "pool_objects": cfg["gnss_pool"]["num_metadata_objects"],
        "gpus": [{"gpu": k, "chains": [v["chains"][t] for t in sorted(v["chains"])],
                  "search": v.get("search")}
                 for k, v in sorted(gpus.items())],
    }
    return g


def write_vars(g, path):
    def lit(v):
        # ⚠️ "is it a number" must be float(), not a digits-and-dots regex: that regex
        # accepted the telemetry host 10.222.3.6 and emitted it unquoted, which jinja
        # parsed as 10.222 subscripted by .3 -- "float object has no element 3".
        if isinstance(v, str) and not v.startswith(("[", "{")):
            try:
                float(v)
            except ValueError:
                return '"%s"' % v
            return v
        return v
    out = ["{#", "  GENERATED BRIDGE FILE -- the variable GNSS data for one node.",
           "",
           "  ⚠️ NOT YET THE SOURCE OF TRUTH. Today this is extracted from the config the",
           "  Python generator emits, so the restructure can be gated (scripts/gnss/",
           "  j2_chain_equiv.py) before the generator is changed. The end state is the",
           "  generator emitting this directly from the node table, at which point the",
           "  captured production base goes away and the GNSS branch is included INTO",
           "  chord_pathfinder.j2 rather than injected into a frozen copy of it.",
           "#}", "{% set gnss = {"]
    for k, v in g.items():
        if k == "gpus":
            continue
        out.append('    "%s": %s,' % (k, lit(v)))
    out.append('    "gpus": [')
    for gp in g["gpus"]:
        out.append('        {"gpu": %d, "search": %r, "chains": ['
                   % (gp["gpu"], gp["search"]))
        for c in gp["chains"]:
            out.append('            {"tag": "%s", "chain": "%s", "signal": "%s",'
                       % (c["tag"], c["chain"], c["signal"]))
            out.append('             "channel_ids": %s,' % c["channel_ids"])
            out.append('             "local_channels": %s,' % c["local_channels"])
            out.append('             "prns": %s,' % c["prns"])
            out.append('             "f_offset_hz": %r,' % c["f_offset_hz"])
            out.append('             "cores": %r,' % c["cores"])
            out.append('             "sizes": %r},' % c["sizes"])
        out.append('        ]},')
    out.append("    ],")
    out.append("} %}")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def render(vars_name):
    import jinja2
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(GNSS_DIR))
    # ⚠️ IMPORT the vars, INCLUDE the structure. Jinja passes the parent context DOWN into
    # an {% include %} but does not export the include's {% set %} names back up, so a
    # vars file that is included is invisible to everything after it ("gnss is
    # undefined"). {% import %} does export top-level assignments. This is the one
    # mechanical constraint on the split and it belongs in the top-level file.
    tmpl = env.from_string(
        '{%% import "%s" as gv %%}{%% set gnss = gv.gnss %%}'
        '{%% include "gnss_chain.j2" %%}' % os.path.basename(vars_name))
    return yaml.safe_load(tmpl.render())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("generated", help="a config emitted by gen_chord_gnss_config.py")
    ap.add_argument("--keep", action="store_true", help="keep the emitted vars file")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.generated))
    node = re.search(r"chord_gnss_(\w+?)_", os.path.basename(a.generated)).group(1)
    g = extract(cfg, node)
    os.makedirs(GNSS_DIR, exist_ok=True)
    # ⚠️ NEVER WRITE OVER THE COMMITTED VARS unless asked. Those files are now real
    # artifacts (the generator emits them with --emit-j2-vars and chord_pathfinder.j2
    # includes them), and this checker used to write one and then DELETE it -- which was
    # harmless when nothing was committed there and destroyed six tracked files the first
    # time it ran afterwards. A check that mutates its inputs is not a check.
    vars_path = os.path.join(GNSS_DIR, ("gnss_vars_%s.j2" if a.keep
                                        else ".gnss_vars_%s.check.j2") % node)
    write_vars(g, vars_path)

    got = render(vars_path)
    want = {k: v for k, v in cfg.items()
            if (CHAIN_RE.match(k) and CHAIN_RE.match(k).group(3) in SUFFIXES)
            or re.match(r"^gnss[01]_(%s)$" % "|".join(SEARCH_SUFFIXES), k)
            or k == "gnss_pool"}
    orphans = sorted(k for k in cfg
                     if re.match(r"^gnss[01]_(%s)$" % "|".join(ORPHANS), k))

    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    bad = []
    for k in sorted(set(want) & set(got)):
        if got[k] != want[k]:
            wf, gf = want[k], got[k]
            diff = [f for f in sorted(set(wf) | set(gf)) if wf.get(f) != gf.get(f)]
            bad.append((k, diff, {f: (wf.get(f), gf.get(f)) for f in diff[:3]}))

    print("node %s: %d GNSS blocks expected, %d rendered" % (node, len(want), len(got)))
    if orphans:
        print("  NOT RENDERED (orphans in the generator's output, nothing references "
              "them): %s" % ", ".join(orphans))
    if missing:
        print("  MISSING from the template : %s" % ", ".join(missing[:8]))
    if extra:
        print("  EXTRA from the template   : %s" % ", ".join(extra[:8]))
    for k, diff, sample in bad[:6]:
        print("  FIELD MISMATCH %-28s %s" % (k, ",".join(diff)[:60]))
        for f, (w, gv) in sample.items():
            print("        %-22s generator=%.60r  j2=%.60r" % (f, w, gv))
    if not a.keep:
        os.remove(vars_path)          # only ever the dot-prefixed check copy
    ok = not (missing or extra or bad)
    print("\n%s" % ("EQUIVALENT -- the j2 include reproduces every GNSS block."
                    if ok else "*** NOT EQUIVALENT (%d missing, %d extra, %d differing)"
                               % (len(missing), len(extra), len(bad))))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
