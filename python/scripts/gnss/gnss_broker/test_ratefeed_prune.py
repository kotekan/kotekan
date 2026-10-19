"""The fine rate feed's seed-set prune: a satellite that left the seed set loses its fine and
KCOH locks, so on its return the coarse feed pulls its rrate row back in at full weight.

    python3 -m unittest gnss_broker.test_ratefeed_prune
"""
import unittest
from types import SimpleNamespace

from gnss_broker.loopstate import RateFeedState
from gnss_broker.ratefeed import stage_rate_feed_coarse, stage_rate_feed_fine

T0 = 5000.0  # s, the cycle clock
SIG_COARSE = 0.2  # Hz, the coarse feed's sigma before any deweight
DEWEIGHT = 8.0


class _Joint:
    """Records each update_rrate call; the rest is what the feeds' log lines read."""

    n_rrate = 0
    rrate_rejected = 0

    def __init__(self):
        self.updates = []

    def update_rrate(self, k, y, t, f_car, sigma_hz):
        self.updates.append((k, sigma_hz))
        return 0.0

    def gauge_rrate(self):
        pass

    def rrate(self, k):
        return 0.0

    def rrate_sigma(self, k):
        return 1.0

    def f_carrier(self):
        return 0.0

    def f_carrier_sigma(self):
        return 0.0


def _ctx(seeds):
    jr = _Joint()
    return SimpleNamespace(
        args=SimpleNamespace(
            rrate_state=True,
            rrate_feed_applied=0,
            rrate_coarse_deweight=DEWEIGHT,
            rrate_fine_hold_s=240.0,
            rrate_phase_span_s=0.0,
            hops_per_sec=195312.5,
            chip_rate_hz=10.23e6,
            carrier_hz=1176.45e6,
            dr_constellation="G",
        ),
        rf=RateFeedState(),
        drp=SimpleNamespace(t_now_abs=1.0e9),
        rx=SimpleNamespace(joint_receiver=lambda *a, **kw: jr),
        jr=jr,
        band_id="gps_l5",
        code_len=10230,
        status={},
        seeds=seeds,
        t0=T0,
    )


class TestSeedSetPrune(unittest.TestCase):
    def test_kcoh_lock_is_dropped_with_the_satellite(self):
        ctx = _ctx(seeds={8: {}})
        ctx.rf.kcoh_t.update({7: T0 - 10.0, 8: T0 - 10.0})
        stage_rate_feed_fine(ctx)
        self.assertNotIn(7, ctx.rf.kcoh_t)
        self.assertIn(8, ctx.rf.kcoh_t)

    def test_fine_lock_is_dropped_with_the_satellite(self):
        ctx = _ctx(seeds={8: {}})
        ctx.rf.fine_t.update({7: T0 - 10.0, 8: T0 - 10.0})
        stage_rate_feed_fine(ctx)
        self.assertNotIn(7, ctx.rf.fine_t)
        self.assertIn(8, ctx.rf.fine_t)

    def test_a_returning_satellite_is_fed_at_full_weight(self):
        ctx = _ctx(seeds={8: {}})
        ctx.rf.kcoh_t.update({7: T0 - 10.0, 8: T0 - 10.0})
        stage_rate_feed_fine(ctx)  # 7 has left the seed set
        ctx.seeds[7] = {}  # ... and is back on the next poll
        ctx.t0 += 2.0
        ctx.rf.resid2 = {7: 0.1, 8: 0.1}
        stage_rate_feed_coarse(ctx)
        sig = {k[1]: s for k, s in ctx.jr.updates}
        self.assertEqual(sig.get(7), SIG_COARSE)
        self.assertEqual(sig.get(8), SIG_COARSE * DEWEIGHT)  # still locked: deweighted


if __name__ == "__main__":
    unittest.main()
