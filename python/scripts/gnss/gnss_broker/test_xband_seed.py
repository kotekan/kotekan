"""--seed-from-chain and --detectors-prns: a satellite seeded from another chain's track carries
its physical code phase across mod this chain's long code, and a per-PRN detector list leaves
every other PRN model-owned.

    python3 -m unittest gnss_broker.test_xband_seed
"""

import unittest
from fractions import Fraction

from gnss_broker import deadreckon as dr
from gnss_broker import signals
from gnss_broker import telem
from gnss_broker import trkresid as tr

HPS = 1920000  # bladeRF: 64-sample hops at 122.88 MSPS
HOP = 1920000 * 86400 * 3  # three days of uptime
B2A = signals.get("bds_b2a")
L_B2A = B2A.code_length * B2A.long_code_segments  # 1023000
F_B3I = 1268.52e6


class _Obj(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


def arg_for(phase, dop, hop):
    """The record argument that puts the replica at physical `phase` (B2a chips) at `hop`."""
    t_abs = Fraction(hop, HPS)
    ramp = (
        t_abs
        * int(B2A.chip_rate_hz)
        * (1 + Fraction(dop).limit_denominator(10 ** 9) / int(B2A.carrier_hz))
    )
    return float((Fraction(phase) - ramp) % L_B2A)


class _Frame(object):
    def __init__(self, recs):
        self.recs = recs  # [(hop, {prn: (cp, dop)})]
        self.n_rec = len(recs)

    def has_record(self, r):
        return True

    def hop(self, r):
        return self.recs[r][0]

    def prns(self):
        return sorted({p for _h, rows in self.recs for p in rows})

    def row(self, r, prn):
        got = self.recs[r][1].get(prn)
        if got is None:
            return None
        row = [0.0] * (max(telem.REC_CP, telem.REC_DOPPLER, telem.REC_P_ENERGY) + 1)
        row[telem.REC_CP], row[telem.REC_DOPPLER], row[telem.REC_P_ENERGY] = (
            got[0],
            got[1],
            1.0,
        )
        return row


class _Telem(object):
    def __init__(self, frame):
        self.frame = frame

    def windows(self, chain, lag=1):
        return [1] if chain == "bds_b2a" else []

    def frame_set(self, chain, w):
        return {"gx10": self.frame}


def make_ctx(phase0, dop, present=True):
    """A bds_b3i context whose bds_b2a neighbour holds PRN 58 at physical phase0 (B2a chips) at
    HOP, Doppler dop at B2a, with records every 1 ms for 10 ms."""
    recs = []
    for k in range(10):
        hop = HOP + k * 1920
        dt = k * 1e-3
        ph = phase0 + dt * B2A.chip_rate_hz * (1.0 + dop / B2A.carrier_hz)
        recs.append((hop, {58: (arg_for(ph, dop, hop), dop)}))
    args = _Obj(
        seed_from_chain="bds_b2a",
        hops_per_sec=HPS,
        code_doppler_sign=1.0,
        carrier_hz=F_B3I,
    )
    rx = _Obj(trk=lambda chain, t_now=None: ({58: 0.0} if present else {}))
    return _Obj(
        args=args,
        rx=rx,
        t0=0.0,
        telem_client=_Telem(_Frame(recs)),
        drp=_Obj(mod=10230 * 20),
    )


class XbandSeed(unittest.TestCase):
    def test_source_phase_is_the_newest_record_phase(self):
        phase0, dop = 512345.25, -1234.5
        ctx = make_ctx(phase0, dop)
        xs = dr.xband_source(ctx, 58)
        self.assertIsNotNone(xs)
        t9 = (HOP + 9 * 1920) / float(HPS)
        want = (phase0 + 9e-3 * B2A.chip_rate_hz * (1.0 + dop / B2A.carrier_hz)) % L_B2A
        self.assertAlmostEqual(xs["t"], t9, places=9)
        self.assertLess(abs(tr.wrap(xs["phys"] - want, L_B2A)), 1e-3)
        self.assertAlmostEqual(xs["dop"], dop, places=3)

    def test_no_source_without_presence(self):
        self.assertIsNone(dr.xband_source(make_ctx(100.0, 0.0, present=False), 58))
        self.assertIsNone(dr.xband_source(make_ctx(100.0, 0.0), 57))

    def test_phase_carries_mod_this_chains_long_code(self):
        phase0, dop = 812345.5, 2500.0
        ctx = make_ctx(phase0, dop)
        xs = dr.xband_source(ctx, 58)
        t1 = xs["t"] + 0.25
        got = dr.xband_phys(ctx, xs, t1, 0.0)
        want = (xs["phys"] + 0.25 * B2A.chip_rate_hz * (1.0 + dop / B2A.carrier_hz)) % (
            10230 * 20
        )
        self.assertLess(abs(tr.wrap(got - want, 10230 * 20)), 1e-6)
        # A Doppler rate adds half rate x t^2 of code Doppler, in this chain's carrier.
        drate = -0.8  # Hz/s at B3I
        got2 = dr.xband_phys(ctx, xs, t1, drate)
        want2 = 0.5 * B2A.chip_rate_hz * drate / F_B3I * 0.25 ** 2
        self.assertLess(abs(tr.wrap(got2 - got, 10230 * 20) - want2), 1e-9)


class DetectorsPrns(unittest.TestCase):
    def ctx(self, only):
        return _Obj(
            detectors=["http://x/b2a_search"],
            args=_Obj(detectors_monitor=False, detectors_prns=only, hint_skip_dr=False),
            mp_flipped=set(),
        )

    def test_listed_prn_is_search_owned_others_model_owned(self):
        c = self.ctx([58])
        self.assertTrue(dr.seeds_are_model_owned(c, 19))
        self.assertFalse(dr.seeds_are_model_owned(c, 58))

    def test_empty_list_keeps_the_chain_level_rule(self):
        self.assertFalse(dr.seeds_are_model_owned(self.ctx([]), 19))


class ParseSats(unittest.TestCase):
    def test_parse(self):
        import nav_decode

        self.assertEqual(
            nav_decode.parse_sats("C58, r25"), frozenset({("C", 58), ("R", 25)})
        )
        self.assertEqual(nav_decode.parse_sats(""), frozenset())


if __name__ == "__main__":
    unittest.main()
