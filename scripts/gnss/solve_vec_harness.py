#!/usr/bin/env python3
"""Old-vs-new gate for fleet_coherent's _solve, driven through the `source` entry point.

fleet_coherent is fed `got = {inst: {prn: {hop: (A, energy)}}}` -- exactly what
TelemClient.coherent_source builds -- so the whole estimator is exercisable without a fleet,
a fixture file or a transport. Shapes deliberately production-sized: a vectorised inner loop
hides its bugs in the degenerate cases (one instance, one hop, an instance that drops out of
the cohort), not in the happy path.
"""
import sys, os, math, random, importlib.util, json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "/home/kvand/gnss/kotekan/python/scripts/gnss")

def load(path, name):
    # Loaded INSIDE the package so the module's relative imports resolve: this file is
    # gnss_broker.fleet in all but name, and stubbing its imports would test a different one.
    import gnss_broker  # noqa: F401  (ensures the package is importable/initialised)
    sp = importlib.util.spec_from_file_location("gnss_broker." + name, path)
    m = importlib.util.module_from_spec(sp)
    m.__package__ = "gnss_broker"
    sys.modules["gnss_broker." + name] = m
    sp.loader.exec_module(m)
    return m

def make_got(seed, n_inst=12, n_hop=128, prns=(1, 4, 9, 16, 27, 31), drop=True):
    rng = random.Random(seed)
    got = {}
    hop0 = 33_000_000_000
    for i in range(n_inst):
        u = "http://cx%02d:12048/gnss%d" % (19 + i // 2, i % 2)
        per = {}
        for prn in prns:
            amp = {1: 1.0, 4: 0.6, 9: 0.25, 16: 0.05, 27: 0.02, 31: 0.9}[prn]
            phi = rng.uniform(0, 2 * math.pi)          # per-instance carrier constant
            d = {}
            for k in range(n_hop):
                # ragged windows: instances start at different hops, some records missing
                if drop and rng.random() < 0.08:
                    continue
                hop = hop0 + (k + (i % 3)) * 2048
                s = amp * complex(math.cos(phi), math.sin(phi))
                n = complex(rng.gauss(0, 0.3), rng.gauss(0, 0.3))
                e = 40.0 + rng.random() * 6.0
                d[hop] = (s + n, e)
            if d:
                per[prn] = d
        got[u] = per
    return got

def run(mod, got, min_inst=2, min_rec=8):
    fleet_now = max(h for per in got.values() for d in per.values() for h in d)
    return mod.fleet_coherent([], min_inst, min_rec, log=None, seed=7,
                              hop_rate_hz=195312.5, source=(got, fleet_now))

def cmp_out(a, b, tol=1e-9):
    bad = []
    for prn in sorted(set(a) | set(b)):
        if prn not in a or prn not in b:
            bad.append("PRN %s present in only one arm" % prn); continue
        ra, rb = a[prn], b[prn]
        for k in sorted(set(ra) | set(rb)):
            va, vb = ra.get(k), rb.get(k)
            if isinstance(va, dict):
                for kk in sorted(set(va) | set(vb)):
                    x, y = va.get(kk), vb.get(kk)
                    if x is None or y is None or abs(x - y) > tol * max(1.0, abs(x), abs(y)):
                        bad.append("PRN %s %s[%s]: %r vs %r" % (prn, k, kk, x, y))
            elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                if abs(va - vb) > tol * max(1.0, abs(va), abs(vb)):
                    bad.append("PRN %s %s: %.17g vs %.17g" % (prn, k, va, vb))
            elif va != vb:
                bad.append("PRN %s %s: %r vs %r" % (prn, k, va, vb))
    return bad

if __name__ == "__main__":
    old = load(sys.argv[1], "fleet_old_solve")
    new = load(sys.argv[2], "fleet_new_solve")
    cases = [("production 12x128", dict(seed=1)),
             ("ragged, 3 instances", dict(seed=2, n_inst=3)),
             ("short window (16 hops)", dict(seed=3, n_hop=16)),
             ("no drops", dict(seed=4, drop=False)),
             ("2 instances (min)", dict(seed=5, n_inst=2)),
             ("single PRN", dict(seed=6, prns=(1,)))]
    fail = 0
    for name, kw in cases:
        got = make_got(**kw)
        a = run(old, got); b = run(new, got)
        bad = cmp_out(a, b)
        print("  %-24s %2d PRN | %s" % (name, len(a), "OK" if not bad else "MISMATCH"))
        for m in bad[:4]:
            print("        %s" % m)
        fail += bool(bad)
    print("HARNESS %s" % ("PASS" if not fail else "FAIL"))
    sys.exit(1 if fail else 0)
