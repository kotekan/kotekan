"""rate_residuals' continuity gate across a hop counter that goes back: an F-engine re-base
restarts pow_hop, and the learned emit spacing (unit_hop) must survive it, or every later step
reads as a window gap and the rrate coarse feed starves.

    python3 -m unittest gnss_broker.test_rate_residuals
"""
import unittest

from gnss_broker.fits import rate_residuals

EMIT = 2048  # hops between combiner emits


def _status(hop, prn=7, rate=-2.5):
    return {
        prn: {
            "deep_rate_hz": rate,
            "deep_rate_q": 20.0,
            "amp_snr": 50.0,
            "pow_hop": hop,
        }
    }


class TestHopCounterReset(unittest.TestCase):
    def _run(self, hops):
        prev_hop, prev_val, unit_hop, outs = {}, {}, [0], []
        for h in hops:
            out, _ = rate_residuals(
                _status(h),
                10.0,
                0.0,
                prev_hop=prev_hop,
                prev_val=prev_val,
                unit_hop=unit_hop,
            )
            outs.append(out)
        return outs, unit_hop

    def test_unit_survives_a_reset(self):
        _, unit_hop = self._run([1000, 1000 + EMIT, 1000 + 2 * EMIT, 100, 100 + EMIT])
        self.assertEqual(unit_hop[0], EMIT)

    def test_samples_after_a_reset_are_accepted(self):
        outs, _ = self._run(
            [1000, 1000 + EMIT, 1000 + 2 * EMIT, 100, 100 + EMIT, 100 + 2 * EMIT]
        )
        self.assertEqual([7 in o for o in outs], [False, True, True, False, True, True])

    def test_a_real_gap_is_still_gapped(self):
        outs, _ = self._run([1000, 1000 + EMIT, 1000 + 5 * EMIT, 1000 + 6 * EMIT])
        self.assertEqual([7 in o for o in outs], [False, True, False, True])


if __name__ == "__main__":
    unittest.main()
