"""Where does the MUOS band's power (or a night's leftover after the subtraction) come from? Per recv1 N2
file, the time-averaged covariance of each MUOS channel (1844-1944), input-whitened; steering vectors to each
satellite (and Tau A) made by turning a reference MUOS-5 eigenvector (another file, same channel) by the
geometry, as dtv_steer.py turns the stage's weights: a_j = exp(i (arg v5_ref + SIGN 2 pi f/c r . (s_j(t) -
s5(t_ref)))). Per file, medians over the channels: the top two eigenvalues, each direction's beam power B
(noise units; a random direction reads ~1.1), the top eigenvector's overlap with each direction, and the top
eigenvalue after projecting the directions out. Take the reference from a night without MUOS subtraction.

    muos_dirs.py extract OUT.npz H5 [H5 ...]            # per-file covariances of the MUOS channels
    muos_dirs.py analyze REF.npz[:k] X.npz [X.npz ...]  # file k (default 0) of REF gives v5_ref
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dtv_steer as D  # noqa: E402

FIDS = np.arange(1844, 1945)
K0 = 1536


def one(path):
    import h5py

    with h5py.File(path, "r") as f:
        prod = f["index_map/prod"][:]
        ia, ib = prod["input_a"].astype(int), prod["input_b"].astype(int)
        i0, i1 = FIDS[0] - K0, FIDS[-1] - K0 + 1
        vis = f["vis"][i0:i1]
        flg = f["flags"][i0:i1] > 0
        t = f["bin_t_inst_ns"][:]
    V = np.zeros((len(FIDS), 128, 128), np.complex64)
    keep = np.zeros((len(FIDS), 128), bool)
    tm = np.nan
    for c in range(len(FIDS)):
        ok = flg[c].any(axis=0)
        if not ok.any():
            continue
        keep[c] = flg[c][:, ok].all(axis=1)
        v = vis[c][:, ok].mean(axis=1)
        V[c][ia, ib] = v
        V[c][ib, ia] = np.conj(v)
        tm = t[ok].mean() / 1e9
    return path, V, keep, tm


def extract(out, files):
    with Pool(min(8, len(files))) as p:
        res = sorted(p.map(one, files), key=lambda x: x[3])
    np.savez(
        out,
        files=np.array([os.path.basename(r[0]) for r in res]),
        V=np.stack([r[1] for r in res]),
        keep=np.stack([r[2] for r in res]),
        t=np.array([r[3] for r in res]),
        fids=FIDS,
    )
    print("saved", out, len(res), "files")


def whiten(V, keep):
    d = np.real(np.diag(V)).astype(float)
    k = keep & (d > 0)
    w = 1.0 / np.sqrt(d[k])
    return (V[np.ix_(k, k)] * w[:, None] * w[None, :]).astype(np.complex128), k


def taua(unix):
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    loc = EarthLocation(
        lat=49.3199845 * u.deg, lon=-119.62262691 * u.deg, height=528.4 * u.m
    )
    aa = SkyCoord(ra=83.63308 * u.deg, dec=22.0145 * u.deg).transform_to(
        AltAz(obstime=Time(unix, format="unix"), location=loc)
    )
    a, z = aa.alt.rad, aa.az.rad
    return np.array([np.cos(a) * np.sin(z), np.cos(a) * np.cos(z), np.sin(a)])


def analyze(ref, xs, tle, pos):
    rp, _, rk = ref.partition(":")
    R = np.load(rp)
    k = int(rk or 0)
    sky = D.Sky(tle)
    r = D.positions(pos)
    s5r = sky.enu("MUOS-5", R["t"][k])
    vref, kref = {}, {}
    for c, fid in enumerate(FIDS):
        Rw, kk = whiten(R["V"][k, c], R["keep"][k, c])
        if kk.sum() < 64:
            continue
        ev, U = np.linalg.eigh(Rw)
        u = np.zeros(128, complex)
        u[kk] = U[:, -1]
        vref[fid], kref[fid] = u, kk
    names = ["MUOS-5", "MUOS-1", "MUOS-2", "TauA"]
    print(
        "reference: %s, MUOS-5 top eigenvectors of %d channels"
        % (R["files"][k], len(vref))
    )
    hdr = "%-34s %5s %6s %6s | %s | %s | %s" % (
        "file",
        "nch",
        "l1",
        "l2",
        " ".join("B:%-6s" % n[-6:] for n in names),
        " ".join("c1:%-5s" % n[-5:] for n in names),
        "l1 after projecting out: 5 / 1 / 2 / 5+1+2 / 5+1+2+TA",
    )
    for x in xs:
        X = np.load(x)
        print("\n" + x)
        print(hdr)
        for n in range(len(X["t"])):
            tt = X["t"][n]
            S = {nm: sky.enu(nm, tt) for nm in names[:3]}
            S["TauA"] = taua(tt)
            rows = []
            for c, fid in enumerate(FIDS):
                if fid not in vref:
                    continue
                Rw, kk = whiten(X["V"][n, c], X["keep"][n, c] & kref[fid])
                m = kk.sum()
                if m < 64:
                    continue
                ev, U = np.linalg.eigh(Rw)
                noise = ev[8 : m - 20].mean()
                A = {
                    nm: np.exp(
                        1j * (np.angle(vref[fid][kk]) + D.turn(r[kk], fid, s5r, S[nm]))
                    )
                    / np.sqrt(m)
                    for nm in names
                }
                B = [np.real(A[nm].conj() @ Rw @ A[nm]) / noise for nm in names]
                c1 = [abs(A[nm].conj() @ U[:, -1]) ** 2 for nm in names]
                proj = []
                for sel in (["MUOS-5"], ["MUOS-1"], ["MUOS-2"], names[:3], names):
                    Q, _ = np.linalg.qr(np.stack([A[nm] for nm in sel], axis=1))
                    P = np.eye(m) - Q @ Q.conj().T
                    proj.append(np.linalg.eigvalsh(P @ Rw @ P)[-1] / noise)
                rows.append([ev[-1] / noise, ev[-2] / noise] + B + c1 + proj)
            a = np.median(np.array(rows), axis=0)
            print(
                "%-34s %5d %6.2f %6.2f | %s | %s | %s"
                % (
                    X["files"][n][:34],
                    len(rows),
                    a[0],
                    a[1],
                    " ".join("%8.2f" % v for v in a[2:6]),
                    " ".join("%8.3f" % v for v in a[6:10]),
                    " / ".join("%.2f" % v for v in a[10:]),
                )
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=("extract", "analyze"))
    ap.add_argument("out_or_ref")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tle", default=os.path.join(D.WORK, "muos.tle"))
    ap.add_argument(
        "--positions",
        default=os.path.join(
            D.REPO, "config", "elem_positions_taua_20261024_by_dish.json"
        ),
    )
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.out_or_ref, a.files)
    else:
        analyze(a.out_or_ref, a.files, a.tle, a.positions)


if __name__ == "__main__":
    main()
