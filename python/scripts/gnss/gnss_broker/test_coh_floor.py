"""coh_cn0's whiteness audit reads ~1 on white noise: the probe fold floor is not clipped low.

    python3 -m gnss_broker.test_coh_floor

floor_white (and floor_white_sky) is the probes' directly measured fold floor over the anchor's
sigma^2/n. On white noise each probe's folded power is one exponential draw, so the clipped mean
that measures it must clear the exponential's tail; at 8x the median it drops ~2.6% of the mass
and reads ~2% low. The anchor (per-instance means over n records) keeps its 8x clip, which drops
a contaminated probe instance; that is checked through coh_cn0's served sigma2.
"""

import random
import sys

from gnss_broker import combdll

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class _Frame:
    """The slice of telem.Frame coh_cn0 reads: n_rec records of one row per PRN."""

    hops_per_record = 2048

    def __init__(self, rows):
        self._rows = rows  # {prn: [(raw, sky) complex per record]}
        self.n_rec = len(next(iter(rows.values())))

    def has_record(self, r):
        return True

    def hop(self, r):
        return r * self.hops_per_record

    def prns(self):
        return list(self._rows)

    def row(self, r, prn):
        a, sky = self._rows[prn][r]
        row = [0.0] * 26
        row[3], row[4], row[5] = a.real, a.imag, 1.0  # REC_P_RE/IM, REC_P_ENERGY
        row[24], row[25] = sky.real, sky.imag  # REC_SKY_RE/IM
        return row


class _Client:
    def __init__(self, frames):
        self._frames = frames

    def windows(self, chain, lag=1):
        return [100]

    def frame_set(self, chain, w):
        return self._frames


def white_probes(n_inst, n_rec, probes, seed):
    """n_inst instances, each carrying n_rec records of unit-power complex white noise per
    probe PRN, independently in the raw and the sky prompt."""
    rng = random.Random(seed)
    s = 0.5 ** 0.5

    def draw():
        return complex(rng.gauss(0.0, s), rng.gauss(0.0, s))

    return {
        "i%d" % i: _Frame({p: [(draw(), draw()) for _ in range(n_rec)] for p in probes})
        for i in range(n_inst)
    }


def _coh(frames, probes):
    return combdll.coh_cn0(
        _Client(frames),
        "gps_l5",
        prns=(),
        probe_prns=probes,
        min_instances=1,
        resid_s=0.0,
    )


def test_floor_white_on_white_noise():
    print("coh_cn0: floor_white and floor_white_sky on white noise, 40000 probe folds")
    probes = (201, 202)
    out = _coh(white_probes(20000, 4, probes, seed=7), probes)
    fw, fws = out[201]["floor_white"], out[201]["floor_white_sky"]
    print("    floor_white %.4f, floor_white_sky %.4f" % (fw, fws))
    # Standard error of 40000 exponential draws is 0.5%; an 8x clip reads ~0.978.
    check(abs(fw - 1.0) < 0.012, "floor_white within 1.2%% of 1 (%.4f)" % fw)
    check(abs(fws - 1.0) < 0.012, "floor_white_sky within 1.2%% of 1 (%.4f)" % fws)


def test_anchor_drops_a_contaminated_instance():
    print("coh_cn0: the anchor's 8x clip drops a contaminated probe instance")
    probes = (201, 202)
    frames = white_probes(50, 4, probes, seed=11)
    clean = [
        sum(abs(a) ** 2 for a, _ in f._rows[p]) / f.n_rec
        for f in frames.values()
        for p in probes
    ]
    # 12x the noise power: above 8x the anchor population's median (~0.9), under 20x
    frames["bad"] = _Frame({201: [(complex(12.0 ** 0.5, 0.0),) * 2] * 4})
    out = _coh(frames, probes)
    want = sum(clean) / len(clean)
    got = out[202]["sigma2"] * 4
    check(
        abs(got / want - 1.0) < 1e-9,
        "anchor %.4f is the clean instances' mean %.4f" % (got, want),
    )


def test_clipped_mean_of_an_exponential():
    print("_floor: clipped mean of an exponential population")
    rng = random.Random(3)
    vals = [rng.expovariate(1.0) for _ in range(20000)]
    m = sum(vals) / len(vals)
    probe = combdll._floor(vals, combdll.PROBE_CLIP_X)
    check(
        abs(probe / m - 1.0) < 0.003, "at PROBE_CLIP_X: %.4f of the mean" % (probe / m)
    )
    eight = combdll._floor(vals, 8.0)
    check(
        eight / m < 0.985, "at 8x the median: %.4f of the mean (the cut)" % (eight / m)
    )
    check(combdll._floor([1.0], 8.0) is None, "one value is no floor")


if __name__ == "__main__":
    test_floor_white_on_white_noise()
    test_anchor_drops_a_contaminated_instance()
    test_clipped_mean_of_an_exponential()
    if _fails:
        print("FAIL: %d check(s)" % len(_fails))
        sys.exit(1)
    print("PASS")
