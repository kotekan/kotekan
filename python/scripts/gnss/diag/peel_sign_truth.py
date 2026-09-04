#!/usr/bin/env python3
"""Compare the sign the peel ACTUALLY APPLIED against the deterministic pilot overlay.

    peel_sign_truth.py [/tmp/gnss_run/peel_sign_l1_bds.csv]

THE QUESTION. Satellites at thousands of sigma peel 0.0 dB because the applied overlay sign
is wrong on ~50% of records -- the "pure sign" class (phase clean: <cos 2d> ~ 0.98, so this
is not a carrier problem). Every aggregate split in the tracker scores the phase residual
AGAINST the applied sign, so none of them can see what the sign SHOULD have been. This can,
because a PILOT overlay is fully deterministic: no decode, no broker, no ephemeris.

WHY NO ABSOLUTE TIME IS NEEDED. capture_utc0 is a placeholder (1.0) in the live configs, so
the dump's `utc` column is not GPS time. It does not have to be: every PRN on a chain shares
ONE receiver clock and ONE overlay epoch, so the record index n = wstart / (T_p * fs) maps to
overlay chip (n + k) mod 1800 with a SINGLE unknown k for the whole chain. Fit k on a
satellite that is demonstrably working (gcoh high, |cos d| ~ 1), then apply that same k to the
broken ones. If a broken sat's applied signs match the overlay at a DIFFERENT k, the fault is
an indexing offset; if they match at no k, the signs are not the overlay at all.

READING THE RESULT
  match ~100% on the working sats, ~50% on the broken ones, best-k EQUAL   -> the table is
      right and the bug is in what bit_at() returns for particular records: look at the
      per-record mismatch pattern (bf, straddle, position in the overlay cycle) printed below.
  best-k DIFFERS by a constant                                             -> indexing offset
      (off-by-one record, or a stale epoch) on that satellite.
  no k matches above chance on a broken sat                                -> the applied
      signs are not this overlay -- wrong PRN table, or the sign is being manufactured
      somewhere other than the bit_pred lookup.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bds_b1c_secondary_check import b1c_codes, SEC_LEN  # noqa: E402

T_P = 10e-3      # B1C primary period = one record
FS = 20e6        # L1 front end
SAMP_PER_REC = T_P * FS


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            p = line.rstrip("\n").split(",")
            if len(p) < 13:
                continue
            try:
                rows.append((int(p[0]), int(p[1]), int(p[2]), float(p[4]), int(p[5]),
                             int(p[6]), int(p[7]), float(p[8]), float(p[9]), float(p[10]),
                             float(p[11]), float(p[12])))
            except ValueError:
                continue
    dt = np.dtype([("prn", "i4"), ("gain_n", "i8"), ("wstart", "i8"), ("bf", "f8"),
                   ("s_head", "i4"), ("s_tail", "i4"), ("have_bits", "i4"), ("d", "f8"),
                   ("cos_d", "f8"), ("cos_2d", "f8"), ("g_abs", "f8"), ("gcoh", "f8")])
    return np.array(rows, dtype=dt)


def best_phase(rec_idx, applied, sec):
    """Best overlay phase k and its match fraction, over all SEC_LEN alignments.

    Both polarities are tried: a global sign flip on the overlay is unobservable in the
    peel (the gain absorbs it), so it must not be reported as a fault."""
    best = (-1, -1.0, 1)
    for pol in (1, -1):
        for k in range(SEC_LEN):
            truth = pol * sec[(rec_idx + k) % SEC_LEN]
            frac = float(np.mean(truth == applied))
            if frac > best[1]:
                best = (k, frac, pol)
    return best


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gnss_run/peel_sign_l1_bds.csv"
    r = load(path)
    if r.size == 0:
        print("no rows in %s" % path)
        return 1
    r = r[r["have_bits"] == 1]
    if r.size == 0:
        print("%d rows, but NONE with have_bits=1 -- the broker has not pushed bit_pred yet."
              % load(path).size)
        return 1
    print("%s: %d records with predicted signs, %d PRNs\n"
          % (path, r.size, len(np.unique(r["prn"]))))

    # Record index relative to the first sample seen, common to every PRN on the chain.
    w0 = r["wstart"].min()
    rec_idx_all = np.rint((r["wstart"] - w0) / SAMP_PER_REC).astype(np.int64)

    print("  PRN   n     gcoh   <cos d>   best-k  polarity  APPLIED matches overlay")
    fits = {}
    for prn in sorted(np.unique(r["prn"])):
        m = r["prn"] == prn
        if m.sum() < 400:
            continue
        sec = b1c_codes(int(prn))[1].astype(np.int32)
        k, frac, pol = best_phase(rec_idx_all[m], r["s_head"][m], sec)
        gc = float(np.median(r["gcoh"][m]))
        cd = float(np.mean(r["cos_d"][m]))
        fits[int(prn)] = (k, frac, pol, gc, cd, m)
        state = "PEELING" if gc > 0.5 else "STUCK"
        print("  %-4d %-6d %5.2f  %+7.3f   %5d    %+d      %6.1f%%   %s"
              % (prn, m.sum(), gc, cd, k, pol, 100 * frac, state))

    ok = [p for p, v in fits.items() if v[3] > 0.5]
    bad = [p for p, v in fits.items() if v[3] <= 0.5]
    if not ok or not bad:
        print("\n  (need both a working and a stuck PRN in the same dump to compare)")
        return 0

    print("\n  working PRNs' best-k: %s" % {p: fits[p][0] for p in ok})
    print("  stuck   PRNs' best-k: %s" % {p: fits[p][0] for p in bad})

    # Per-record structure of the mismatches on the stuck sats, against the k that the
    # WORKING sats agree on (the chain's true overlay phase).
    kws = [fits[p][0] for p in ok]
    if len(set(kws)) == 1:
        k_true = kws[0]
        print("\n  the working PRNs agree on k = %d; scoring the stuck PRNs at THAT k:" % k_true)
        for p in bad:
            _, _, pol, gc, cd, m = fits[p]
            sec = b1c_codes(int(p))[1].astype(np.int32)
            idx = rec_idx_all[m]
            for polarity in (1, -1):
                truth = polarity * sec[(idx + k_true) % SEC_LEN]
                agree = truth == r["s_head"][m]
                frac = float(np.mean(agree))
                if polarity == 1 or frac > 0.5:
                    pass
            truth = sec[(idx + k_true) % SEC_LEN]
            agree = truth == r["s_head"][m]
            bf = r["bf"][m]
            cosd = r["cos_d"][m]
            print("    PRN%-3d applied==truth on %5.1f%% of records" % (p, 100 * np.mean(agree)))
            print("           <cos d> where applied==truth: %+.3f   where it differs: %+.3f"
                  % (np.mean(cosd[agree]) if agree.any() else float("nan"),
                     np.mean(cosd[~agree]) if (~agree).any() else float("nan")))
            print("           mean bf   where applied==truth: %.3f    where it differs: %.3f"
                  % (np.mean(bf[agree]) if agree.any() else float("nan"),
                     np.mean(bf[~agree]) if (~agree).any() else float("nan")))
            # Does the mismatch RUN (blocks) or alternate (per-record)? Runs point at an
            # epoch/indexing slip; alternation points at the lookup itself.
            flips = np.count_nonzero(np.diff(agree.astype(np.int8)))
            print("           mismatch runs: %d transitions over %d records (%.2f per record;"
                  " ~0.5 = alternating, <<0.5 = blocky)"
                  % (flips, agree.size, flips / max(agree.size - 1, 1)))
    else:
        print("\n  ⚠️ the working PRNs do NOT agree on k -- the per-PRN overlay tables or the"
              " record indexing differ between satellites; that is itself the finding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
