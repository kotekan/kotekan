import statistics
def fuse(resid, sibs, bias_min_sats=2):
    """Mirror of the patched broker logic. sibs = [(bias_hz, n_sats), ...] (fresh)."""
    n_sib = 0; sib_bw = 0.0; sib_w = 0.0
    for b, n in sibs:
        if n >= 1:
            sib_bw += b*n; sib_w += n; n_sib += n
    local_ok = len(resid) >= bias_min_sats
    sib_ok   = n_sib     >= bias_min_sats
    if not (local_ok or sib_ok):
        return None, n_sib          # UNSOLVED
    if local_ok:
        rb = statistics.median(resid)
        if n_sib:
            w = float(len(resid)); rb = (rb*w + sib_bw)/(w + sib_w)
    else:
        rb = sib_bw/sib_w           # band consensus ALONE; local discarded
    return rb, n_sib

# THE REAL 2026-07-27 SCENARIO: L5 GPS had 1 sat (PRN11); its siblings held the truth.
L5_SIBS = [(32.12, 13), (32.91, 14)]      # GAL, BDS -- measured values from the .hz files
TRUTH   = 32.5   # GPS itself settled at +24..+34 once it finally solved

cases = [
  ("OLD BUG: 1 local sat, siblings IGNORED (gate inside)", [33.0], [],       None),
  ("FIXED:   1 local sat + band consensus",                [33.0], L5_SIBS,  TRUTH),
  ("FIXED:   ZERO local sats, pure band consensus",        [],     L5_SIBS,  TRUTH),
  ("SAFETY:  1 local sat, NO siblings -> still UNSOLVED",  [33.0], [],       None),
  ("SAFETY:  1 local BAD sat + band -> band dominates",    [-450.0], L5_SIBS, TRUTH),
  ("NORMAL:  many local sats, no siblings",                [31.0,33.0,32.0], [], TRUTH),
]
print("%-52s %-12s %-10s %s" % ("case","bias","n_sib","verdict"))
ok=True
for name, resid, sibs, expect in cases:
    b, n = fuse(resid, sibs)
    if expect is None:
        good = b is None
        got = "UNSOLVED" if b is None else "%+.2f Hz" % b
    else:
        good = b is not None and abs(b-expect) < 3.0
        got = "UNSOLVED" if b is None else "%+.2f Hz" % b
    ok &= good
    print("%-52s %-12s %-10d %s" % (name, got, n, "PASS" if good else "*** FAIL ***"))
print()
print("ALL PASS" if ok else "FAILURES PRESENT")
