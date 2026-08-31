#!/usr/bin/env python3
"""Check every generated node config's CPU placement -- the invariants that broke silently.

THREE RULES, and all three were being violated on 2026-08-31 with nothing to say so:

  1. NO TWO cudaProcess STAGES SHARE A LOGICAL CPU. The generator has said for months that
     "two cudaProcess stages must not [share a core]" -- two heavy GPU-submission loops on one
     cpu -- but the rule lived in a comment while the arithmetic
     (`cores[(5 - 2*gpu + 3*ordv) % 10]`) drew BOTH GPUs from ONE ten-core pool. Fine at 5
     chains; at 7 it is unsatisfiable, and four pairs collided. Nothing noticed, because
     nothing checked.
     ⚠️ HT/SMT sharing is explicitly ALLOWED (KV, 2026-08-31): sibling threads of one physical
     core are fine given the buffering in front. It is the LOGICAL cpu that must be exclusive.

  2. EACH GPU'S STAGES SIT ON THAT GPU'S NUMA NODE. `nvidia-smi topo -m` on cx19: GPU0 -> node
     0 (cpus 0-15, 32-47), GPU1 -> node 1 (16-31, 48-63). The single pool was entirely node 1,
     so every gnss0_* stage drove GPU0 across the interconnect -- and node 1 carried all 14
     cudaProcess instances while node 0 carried none of ours.

  3. NEVER A DPDK PHYSICAL CORE. lcores 5-7 and 21-23 are poll-mode spinners at 0% idle, so
     their SMT siblings 37-39 and 53-55 are just as unusable. Sharing one starves packet
     capture, which surfaces far away as ring drops and dead workers.

    scripts/gnss/core_placement_gate.py [generated.yaml ...]
"""
import glob
import os
import sys
from collections import defaultdict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUMA = {0: set(range(0, 16)) | set(range(32, 48)), 1: set(range(16, 32)) | set(range(48, 64))}
# lcores and their SMT siblings (+32 on this 2x16x2 topology).
DPDK = {5, 6, 7, 21, 22, 23} | {37, 38, 39, 53, 54, 55}


def check(path):
    cfg = yaml.safe_load(open(path))
    owner = defaultdict(list)
    problems = []
    for name, st in cfg.items():
        if not isinstance(st, dict) or st.get("kotekan_stage") != "cudaProcess":
            continue
        gpu = st.get("gpu_id")
        for core in st.get("cpu_affinity") or []:
            owner[core].append(name)
            if core in DPDK:
                problems.append(f"{name}: core {core} is a DPDK poll core or its SMT sibling")
            if gpu in NUMA and core not in NUMA[gpu]:
                problems.append(f"{name}: core {core} is not on GPU{gpu}'s NUMA node")
    for core, names in sorted(owner.items()):
        if len(names) > 1:
            problems.append(f"core {core} shared by cudaProcess stages: {', '.join(names)}")
    return sorted(owner), problems


def main():
    paths = sys.argv[1:] or sorted(
        glob.glob(os.path.join(ROOT, "config", "generated", "chord_gnss_cx*_multi.yaml")))
    if not paths:
        print("no generated node configs found")
        return 1
    bad = 0
    for p in paths:
        cores, problems = check(p)
        name = os.path.basename(p)
        if problems:
            bad += 1
            print(f"*** {name}: {len(problems)} PROBLEM(S)")
            for x in problems:
                print(f"      {x}")
        else:
            print(f"ok   {name}  cudaProcess cores {cores}")
    print()
    if bad:
        print(f"*** {bad} config(s) violate CPU placement. Fix runtime.cpu_affinity_gpu{{0,1}} "
              "in config/chord_gnss_node.yaml, or dual_core() in the generator.")
        return 1
    print("ALL NODE CONFIGS: distinct logical cpus per cudaProcess, correct NUMA node, "
          "no DPDK cores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
