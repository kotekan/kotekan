#!/usr/bin/env python3
"""PATH A vs PATH B, paired, on the statistic that actually moved (task #35).

WHAT THIS COMPARES AND WHY. The complaint is that GPS C/N0 got NOISIER, and
C/N0 = 20log10(deep_snr) - 10log10(coherence_s), so the thing to compare is a DISTRIBUTION,
not a value. Two spot checks agreeing tells you nothing about scatter -- that is how "path A
and path B are statistically identical" got asserted in the first place.

THE PAIRING IS THE POINT. Both paths are seeded by the same broker iteration with the same
payload (config/gnss_chains_chord.yaml, gps_l5 `trackers`), so at any instant they are
despreading the SAME sky with the SAME replica parameters. Polling them back-to-back in one
loop means every A sample has a B sample from the same second, so a real sky change (a satellite
crossing the beam, an ionospheric event) moves BOTH and cancels out of the comparison. An
unpaired A-now-vs-B-an-hour-ago comparison cannot separate those.

THREE LAYERS, because "noisier" could enter at any of them and they have different causes:
  1. PER-RECORD    -- |A| scatter and record supply straight from /get_records. Engine-local.
  2. FLEET COMBINE -- the real fleet_coherent over the same 12-instance geometry. This is where
                      the live system's scatter actually lives (measured 40-54% before path A
                      was restored, with n_src and n_rec both pinned -- so it is the per-record
                      PHASE ALIGNMENT, not the record supply).
  3. PUBLISHED     -- deep_snr / coherence_s / C/N0 as the viewer sees them.

⚠️ RUN AGAINST cx27/cx42/cx43/cx44 ONLY for layers 1-2. cx19 and cx51 were generated with
--combine-gpus, so their path-A combiner is one merged 14-channel instance against path B's
two 7-channel ones -- a different aperture, which is not the engine.
"""
import os, sys, time, math, json, statistics as st
# gnss_broker is a sibling package of this file -- derive it, never hardcode a checkout path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_broker.fleet import fleet_coherent
import urllib.request

UNIFORM = ["cx27", "cx42", "cx43", "cx44"]       # no --combine-gpus: 7ch vs 7ch, a fair pair
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ab_paths.jsonl"

def urls(stage, nodes=UNIFORM):
    return ["http://%s:12049/gnss%d_%s" % (n, g, stage) for n in nodes for g in (0, 1)]

A, B = urls("combine"), urls("n2combine")

def per_record(us):
    """Layer 1: per-record |A| scatter, record count and hop spacing, per PRN."""
    out = {}
    for u in us:
        try:
            with urllib.request.urlopen("%s/get_records" % u, timeout=5) as r:
                recs = json.load(r)
        except Exception:
            continue
        for rr in recs or []:
            prn = int(rr.get("prn", -1))
            rec = [x for x in (rr.get("records") or []) if float(x[3]) > 0]
            if prn <= 0 or len(rec) < 20:
                continue
            a = [math.hypot(x[1], x[2]) for x in rec]
            hops = [int(x[0]) for x in rec]
            dh = [hops[i+1]-hops[i] for i in range(len(hops)-1)]
            sr = sum(x[1] for x in rec); si = sum(x[2] for x in rec)
            out.setdefault(prn, []).append({
                "n": len(rec), "rel_sd": st.pstdev(a)/st.mean(a),
                "gaps": sum(1 for d in dh if d != 2048),
                "raw_coh": math.hypot(sr, si)/sum(a),
                "energy_rel_sd": st.pstdev([x[3] for x in rec])/st.mean([x[3] for x in rec])})
    return out

def per_hop(nodes=UNIFORM):
    """LAYER 0 -- THE DECISIVE ONE. Compare the two engines on the SAME HOPS.

    Every softer layer compares two DISTRIBUTIONS, and a distribution is confounded by
    whatever window each path happened to be holding and by the sky moving underneath it
    (satellites transit CHORD's beam fast enough that a PRN can move 10x in 45 minutes --
    which is what a whole afternoon of "the system regressed" turned out to be). `hop` is an
    exact integer index into the F-engine stream, so intersecting on it pins both engines to
    the identical input samples, on the identical GPU, at the identical sky instant. There is
    nothing left for a confounder to hide in.

      |B|/|A| per hop  -- a gain or normalisation difference is an offset; a quantisation
                          loss is a median below 1.
      complex corr     -- |<B conj A>| / sqrt(<|A|^2><|B|^2>). THIS is the number that settles
                          "statistically identical": both paths correlate the SAME antenna
                          voltages, so even a noise-dominated record must agree unless the two
                          engines used different REPLICAS. Path B's only intrinsic difference
                          is 4-bit replica quantisation, which predicts corr ~0.9995 flat in
                          signal strength -- so a corr that DROPS with strength is not
                          quantisation and means the seeds differed.
      dphase           -- a constant offset is a convention (path B stores synth on the row
                          side, so a conjugate is expected); SCATTER in it is not.
    """
    out = []
    for n in nodes:
        for g in (0, 1):
            def grab(stage):
                try:
                    with urllib.request.urlopen(
                            "http://%s:12049/gnss%d_%s/get_records" % (n, g, stage), timeout=6) as r:
                        return {rr["prn"]: {int(x[0]): complex(x[1], x[2])
                                            for x in rr["records"] if x[3] > 0}
                                for rr in json.load(r)}
                except Exception:
                    return {}
            A, B = grab("combine"), grab("n2combine")
            for prn in sorted(set(A) & set(B)):
                hops = sorted(set(A[prn]) & set(B[prn]))
                if len(hops) < 20:
                    continue
                a = [A[prn][h] for h in hops]; b = [B[prn][h] for h in hops]
                cr = sum(y * x.conjugate() for x, y in zip(a, b))
                na = math.sqrt(sum(abs(x) ** 2 for x in a))
                nb = math.sqrt(sum(abs(y) ** 2 for y in b))
                out.append((n, g, prn, len(hops),
                            st.median([abs(y) / abs(x) for x, y in zip(a, b) if abs(x) > 0]),
                            abs(cr) / (na * nb) if na * nb > 0 else 0.0,
                            __import__("cmath").phase(cr)))
    return out

print("=== LAYER 0: per-hop, same instance, same hops (the decisive comparison) ===")
print("%-6s %3s %3s | %5s | %7s | %8s | %8s" % ("node", "gpu", "prn", "nhop", "|B|/|A|", "corr", "dphase"))
for n, g, prn, nh, ratio, corr, dph in per_hop():
    print("%-6s %3d %3d | %5d | %7.3f | %8.4f | %+8.3f%s" % (
        n, g, prn, nh, ratio, corr, dph, "" if corr > 0.95 else "   <-- engines disagree"))
print()

hist = {"A": {}, "B": {}}
rec1 = {"A": {}, "B": {}}
with open(OUT, "w") as f:
    for i in range(N):
        for tag, us in (("A", A), ("B", B)):
            t = time.time()
            fc = fleet_coherent(us, 3, 32) or {}
            pr = per_record(us)
            for prn, v in fc.items():
                hist[tag].setdefault(prn, []).append(v)
            for prn, v in pr.items():
                rec1[tag].setdefault(prn, []).extend(v)
            f.write(json.dumps({"t": t, "path": tag, "iter": i,
                                "fcoh": {str(k): {kk: vv for kk, vv in v.items()
                                                  if kk in ("deep_snr","coh_frac","n_rec","n_src",
                                                            "floor","present","deep_amplitude")}
                                         for k, v in fc.items()},
                                "per_record": {str(k): v for k, v in pr.items()}}) + "\n")
            f.flush()
        time.sleep(2)

print("=== LAYER 2: fleet_coherent deep_snr, PAIRED (%d polls, cx27/42/43/44) ===" % N)
print("PRN |        PATH A  (mean    sd   rel) |        PATH B  (mean    sd   rel) | B/A mean | B/A rel-sd")
for prn in sorted(set(hist["A"]) | set(hist["B"])):
    ra, rb = hist["A"].get(prn, []), hist["B"].get(prn, [])
    if len(ra) < 5 or len(rb) < 5:
        continue
    da = [v["deep_snr"] for v in ra]; db = [v["deep_snr"] for v in rb]
    ma, mb = st.mean(da), st.mean(db)
    sa, sb = st.pstdev(da)/ma, st.pstdev(db)/mb
    print("%3d | %14.1f %7.1f %5.1f%% | %14.1f %7.1f %5.1f%% | %8.3f | %8.3f" % (
        prn, ma, st.pstdev(da), 100*sa, mb, st.pstdev(db), 100*sb, mb/ma, sb/sa))

print("\n=== LAYER 2b: coh_frac (the phase-alignment quality) ===")
print("PRN |   A mean    sd |   B mean    sd")
for prn in sorted(set(hist["A"]) & set(hist["B"])):
    ra, rb = hist["A"][prn], hist["B"][prn]
    if len(ra) < 5 or len(rb) < 5:
        continue
    ca = [v.get("coh_frac", 0) for v in ra]; cb = [v.get("coh_frac", 0) for v in rb]
    print("%3d | %8.3f %6.3f | %8.3f %6.3f" % (prn, st.mean(ca), st.pstdev(ca),
                                                st.mean(cb), st.pstdev(cb)))

print("\n=== LAYER 1: per-record, per instance ===")
print("PRN | A: |A| rel-sd  n_rec  gaps  raw_coh | B: |A| rel-sd  n_rec  gaps  raw_coh")
for prn in sorted(set(rec1["A"]) & set(rec1["B"])):
    a, b = rec1["A"][prn], rec1["B"][prn]
    if len(a) < 5 or len(b) < 5:
        continue
    print("%3d | %13.1f%% %6.0f %5d %8.3f | %13.1f%% %6.0f %5d %8.3f" % (
        prn, 100*st.mean([x["rel_sd"] for x in a]), st.mean([x["n"] for x in a]),
        sum(x["gaps"] for x in a), st.mean([x["raw_coh"] for x in a]),
        100*st.mean([x["rel_sd"] for x in b]), st.mean([x["n"] for x in b]),
        sum(x["gaps"] for x in b), st.mean([x["raw_coh"] for x in b])))
print("\nraw jsonl -> %s" % OUT)
