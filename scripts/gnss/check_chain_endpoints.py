#!/usr/bin/env python3
"""Verify every endpoint in the broker's chain file exists on the node configs it names.

WHY THIS EXISTS. fleet_coherent treats an unreachable instance as "skipped, never fatal" and
the seed POST logs one rate-limited line, so a chain pointed at a stage that WAS NEVER BUILT
degrades silently: the broker runs on 10 instances while the file says 12, or seeds a tracker
that does not exist and the PRN simply never appears. Nothing turns red. The concrete trap is
--combine-gpus: on cx19 and cx51 path A's GPU-1 combiner is not instantiated at all, so the
natural-looking `gnss{0..1}_combine` silently loses 2 of 12 endpoints.

Endpoints come in TWO KINDS and only one is a top-level stage:
  * combiners / assemblers -- top-level keys in the node config (gnss0_combine, ...).
  * trackers -- NOT stage keys. They are REST paths a cudaCommand registers from its own
    `seed_endpoint` ("/gnss0_track/set_seeds"), declared inside a GPU stage's `commands`
    list. Checking these against top-level keys reports every one of them missing, which is
    how the first version of this script cried wolf 48 times.

  scripts/gnss/check_chain_endpoints.py [chains.yaml] [--configs config/generated]
"""
import argparse, os, re, sys, glob
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("chains", nargs="?", default="config/gnss_chains_chord.yaml")
ap.add_argument("--configs", default="config/generated")
ap.add_argument("--suffix", default="_multi")
a = ap.parse_args()

cfgs = {}
for p in glob.glob(os.path.join(a.configs, "chord_gnss_*%s.yaml" % a.suffix)):
    node = os.path.basename(p)[len("chord_gnss_"):-len("%s.yaml" % a.suffix)]
    cfgs[node] = yaml.safe_load(open(p))

def endpoints_of(cfg):
    """Every REST name this node config can answer to: stage keys + declared *_endpoint paths."""
    names = set(k for k in cfg if isinstance(cfg.get(k), dict))
    for k, v in cfg.items():
        if isinstance(v, dict):
            for c in v.get("commands") or []:
                for key in ("seed_endpoint", "trim_endpoint"):
                    m = re.match(r"/([^/]+)/", str(c.get(key, "")))
                    if m:
                        names.add(m.group(1))
    return names

have = {n: endpoints_of(c) for n, c in cfgs.items()}

def expand(s):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.search(r"\{(\d+)\.\.(\d+)\}", part)
        if m:
            out += [part[:m.start()] + str(i) + part[m.end():]
                    for i in range(int(m.group(1)), int(m.group(2)) + 1)]
        else:
            out.append(part)
    return out

d = yaml.safe_load(open(a.chains))
bad, summary = 0, []
for cname, ch in (d.get("chains") or {}).items():
    counts = {}
    for key in ("trackers", "dll-combiners", "n2-combiners", "spectrum-endpoints", "detectors"):
        if not ch.get(key):
            continue
        urls = expand(ch[key])
        counts[key] = len(urls)
        for u in urls:
            m = re.match(r"https?://([^:/]+):\d+/(\S+)", u)
            if not m:
                print("  UNPARSED  %-10s %-18s %s" % (cname, key, u)); bad += 1; continue
            node, stage = m.group(1), m.group(2).rstrip("/")
            if node not in have:          # localhost/aggregator: not a node, not ours to check
                continue
            if stage not in have[node]:
                print("  MISSING   %-10s %-18s -> %s/%s" % (cname, key, node, stage)); bad += 1
    cmb = ch.get("combiner")
    if cmb and not any(cmb in s for s in have.values()):
        print("  MISSING   %-10s %-18s -> %s (on any node)" % (cname, "combiner", cmb)); bad += 1
    summary.append((cname, cmb, counts))

print()
for cname, cmb, counts in summary:
    print("  %-9s published=%-24s %s" % (
        cname, cmb, "  ".join("%s=%d" % (k.replace("-combiners", ""), v)
                              for k, v in sorted(counts.items()))))
print("\n%d node config(s) checked -- %s" % (
    len(cfgs), "ALL ENDPOINTS RESOLVE" if not bad else "%d BAD ENDPOINT(S)" % bad))
sys.exit(1 if bad else 0)
