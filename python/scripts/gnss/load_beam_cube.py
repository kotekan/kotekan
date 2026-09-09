"""Load a CHORD GNSS beam cube into dense numpy arrays. Public-friendly, no repo dependency."""
import json
import numpy as np


def load_beam_cube(path, chain=None):
    """-> dict: meta, and per chain {'n','s1','s2'} dense [n_sub, n_elem, npix] + freq_ids.

    Stored sparsely (only visited pixels); this expands to full healpix RING maps of
    npix = 12*nside^2. Pixelisation is the LOCAL HORIZON frame: theta = 90 - elevation,
    phi = azimuth (N=0, clockwise). Values are ACCUMULATORS -- take s1/n for the mean.
    """
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    npix = 12 * meta["nside"] ** 2
    out = {}
    for i, c in enumerate(meta["chains"]):
        if chain and c["chain"] != chain:
            continue
        pix = z["pix_%d" % i]
        d = {"freq_ids": [f[0] for f in c["freq_ids"]], "n_sub": c["n_sub"], "n_elem": c["n_elem"]}
        for q, dt in (("n", np.int64), ("s1", np.float64), ("s2", np.float64)):
            a = np.zeros((c["n_sub"], c["n_elem"], npix), dt)
            a[:, :, pix] = z["%s_%d" % (q, i)]
            d[q] = a
        out[c["chain"]] = d
    return {"meta": meta, "chains": out}


if __name__ == "__main__":
    import sys
    cube = load_beam_cube(sys.argv[1], chain="gps_l5")
    m, c = cube["meta"], cube["chains"]["gps_l5"]
    print("day %s  nside %d  units %s  pointing %s" % (m["day"], m["nside"], m["units"], m["pointing"]))
    print("gps_l5: n%s  freq_ids %d..%d" % (c["n"].shape, c["freq_ids"][0], c["freq_ids"][-1]))
    with np.errstate(all="ignore"):
        mean = np.where(c["n"].sum((0, 1)) > 0,
                        c["s1"].sum((0, 1)) / np.maximum(c["n"].sum((0, 1)), 1), np.nan)
    good = np.isfinite(mean) & (mean > 0)
    print("sky map: %d/%d pixels hit, peak %.1f dB, median %.1f dB"
          % (good.sum(), mean.size, 10 * np.log10(mean[good].max()),
             10 * np.log10(np.median(mean[good]))))
