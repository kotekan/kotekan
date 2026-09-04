"""ARRAY COHERENCE vs SKY POSITION: (coherent map) - (element map), in dB.

    gnss_beam_coh.py <coh.npz> <inc.npz> <out.png> [title]

WHY THIS IS EXACT, and not an approximation. The two accumulators are built from the SAME
rows in the same pixels -- `gnss_beam_elem2obs.py` writes both quantities on every sample -- so per
pixel n_coh == n_inc and

    mean(coh_dB) - mean(inc_dB) = mean(coh_dB - inc_dB)

identically. Differencing the two MEAN maps therefore IS the mean per-sample difference; no
Jensen-gap argument is needed. (It would NOT be if the two maps had been thinned differently,
which is precisely why the pipeline refuses to coadd coh and inc into one map.)

WHAT IT MEASURES, AND ⚠️ WHERE THE ZERO IS. `coh` is |SUM_e u_e| / SUM_e q_e -- it divides by
the SUM of q over elements, so the element count is ALREADY normalised out, while `inc` is
median_e |u_e|/q_e. Therefore:

    0 dB        every element in phase   (SUM u = N*u, over N*q, == u/q == inc)
    -13.4 dB    random phases, 22 live elements: |SUM u| ~ sqrt(N)*|u|, i.e. -20log10(sqrt(22))

⚠️ THE CEILING IS 0, NOT +27 dB. (An earlier version of this file annotated it the other way
round -- the N-normalisation in `coh` is easy to miss, and the plot then flatters a random
array by 27 dB.) The interesting contrast is -13.4 (random) vs 0 (phased).

⚠️ AND WHAT "RANDOM" MEANS HERE IS NOT NECESSARILY A FAULT. `u_e` are the RAW per-element
parts: their phases still carry each element's cable/position term AND the geometric delay
across the array toward the satellite, neither of which has been removed. An uncalibrated,
unsteered sum SHOULD sit near the random floor for a source away from the phase centre. To
read this map as "is the array healthy" the elemcal gains must be applied first
([[chord-elemgain]]); as it stands it maps how far the raw parts sit from a phased sum, and
its STRUCTURE (which sky regions do better) is the part worth reading.

@author Keith Vanderlinde
"""
import sys

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

coh, inc, png = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
title = sys.argv[4] if len(sys.argv) > 4 else "array coherence"

nside = int(coh["nside"])
assert int(inc["nside"]) == nside, "different pixelisations"
n_c, n_i = coh["n"], inc["n"]
if not np.array_equal(n_c, n_i):
    # Not fatal, but it means the two maps did NOT come from one sample set, and the identity
    # above no longer holds. Say so rather than quietly differencing incomparable means.
    print("⚠️ hit counts differ (%d vs %d pixels populated) -- restricting to the intersection"
          % (int((n_c > 0).sum()), int((n_i > 0).sum())))

ok = (n_c > 0) & (n_i > 0)
d = np.full(n_c.size, np.nan)
d[ok] = coh["s1"][ok] / n_c[ok] - inc["s1"][ok] / n_i[ok]

n_live = 22.0                        # elements actually carrying power (08-26 element_power)
rand_db = -20.0 * np.log10(np.sqrt(n_live))   # random phases sit BELOW the phased sum
full_db = 0.0                                  # fully phased == one element, post-normalisation

# Zenith-centred orthographic, NORTH UP / EAST RIGHT -- the sky as seen looking up, matching
# gnss_beam_map.py's convention so the two renders can be laid side by side.
npx = 480
y, x = np.mgrid[0:npx, 0:npx]
xs = (x - npx / 2.0) / (npx / 2.0)
ys = (y - npx / 2.0) / (npx / 2.0)
r = np.hypot(xs, ys)
inside = r <= 1.0
el = np.degrees(np.arccos(np.clip(r, 0, 1)))
az = (np.degrees(np.arctan2(xs, ys))) % 360.0
img = np.full((npx, npx), np.nan)
pix = hp.ang2pix(nside, np.radians(90.0 - el[inside]), np.radians(az[inside]))
img[inside] = d[pix]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
im = ax[0].imshow(img, origin="lower", cmap="magma",
                  vmin=float(np.nanpercentile(img, 2)), vmax=0.0)
ax[0].set_title("coherent sum vs one element (dB); 0 = phased, %.1f = random"
                % rand_db)
cb = fig.colorbar(im, ax=ax[0])
cb.ax.axhline(rand_db, color="cyan", lw=2)
cb.ax.axhline(full_db, color="lime", lw=2)
for a in ax[:1]:
    a.set_xticks([]); a.set_yticks([])
    a.text(npx / 2, npx * 0.98, "N", ha="center", va="top", fontsize=11)
    a.text(npx * 0.98, npx / 2, "E", ha="right", va="center", fontsize=11)
    a.text(npx * 0.02, npx / 2, "W", ha="left", va="center", fontsize=11)
    a.text(npx / 2, npx * 0.02, "S", ha="center", va="bottom", fontsize=11)

v = d[np.isfinite(d)]
ax[1].hist(v, bins=60, color="0.3")
ax[1].axvline(rand_db, color="c", lw=2,
              label="random phases, %.0f elem (%.1f dB)" % (n_live, rand_db))
ax[1].axvline(full_db, color="limegreen", lw=2,
              label="fully phased (0 dB)")
ax[1].axvline(np.median(v), color="orange", ls="--", lw=2,
              label="median %.1f dB" % np.median(v))
ax[1].set_xlabel("coherent sum vs one element (dB)")
ax[1].set_ylabel("pixels")
ax[1].legend(fontsize=8)
fig.suptitle("%s   (%d pixels)" % (title, int(np.isfinite(d).sum())))
fig.tight_layout()
fig.savefig(png, dpi=110)
print("%s: median %+.1f dB, 10-90%% %+.1f..%+.1f  (random %.1f, phased %.1f)"
      % (png, np.median(v), np.percentile(v, 10), np.percentile(v, 90), rand_db, full_db))
