"""Task #49: the DLL presence gate must not be on-peak-biased.

THE LATCH. `present` gated on summed PROMPT power, and prompt power is suppressed precisely
when the tap is off-peak -- so off-peak -> low prompt -> fails the gate -> never trimmed ->
stays off-peak. The loop cannot pull in from the shoulder, which is the whole region a DLL
exists for. Measured 2026-08-12: 26 of 36 well-detected satellites (72%) excluded, and the
excluded ones carried the LARGER discriminators.

⚠️ THE SAME MISTAKE WAS ALREADY MADE ONCE. The gate had been on q, which has this property,
and was moved to prompt power -- which has it too. So the fixture below is built around the
WITNESS CASE (E33 on gal_e5a) and asserts the two gates DISAGREE on it. A test that only
checked "the deep gate admits strong satellites" would have passed for the prompt gate as
well, and would have certified the previous fix exactly as it certified this one.
"""
import unittest

import gnss_broker.fleet as F


# E33 on gal_e5a, 2026-08-12 18:15 UTC, el 69.3, all 12 instances agreeing to a few percent:
# E:P:L = 0.11 : 1.0 : 2.0, disc -0.898, deep_snr 35.2 against a deep_floor of 2.67 (13x).
# The tap triple sits on a near-linear RISING flank -- the peak is well beyond +0.5 chips --
# so the prompt is suppressed and this satellite is exactly the one the DLL must correct.
E33 = {"e": 1.019e-08, "p": 9.462e-08, "l": 1.903e-07, "deep_snr": 35.2, "deep_floor": 2.67}
# A satellite sitting ON its peak: prompt dominates, disc ~ 0, strongly detected.
ONPEAK = {"e": 6.0e-08, "p": 2.4e-07, "l": 6.1e-08, "deep_snr": 126.0, "deep_floor": 2.67}
# Pure noise: three comparable taps, deep at its rectification floor. Must NEVER be trimmed --
# trimming on noise is strictly worse than the latch.
NOISE = {"e": 1.0e-09, "p": 1.02e-09, "l": 0.99e-09, "deep_snr": 2.6, "deep_floor": 2.67}


def make_fleet(spec, n_inst=12):
    """A fleet of identical instances, in the shape fleet_dll consumes."""
    rows = []
    for _ in range(n_inst):
        rows.append([{"prn": prn, "pow_hop": 1000, "pow_fft_len": 16384, "n_chan": 7.0,
                      "e_pow": s["e"], "p_pow": s["p"], "l_pow": s["l"],
                      "deep_snr": s["deep_snr"], "deep_floor": s["deep_floor"],
                      "coherence_s": 1.0486, "amp_snr": 1.0}
                     for prn, s in spec.items()])
    return rows


def run(spec, deep_gate_prns=None, n_bright=6, n_noise=3):
    """Drive the SHIPPED fleet_dll against a REALISTIC population.

    ⚠️ THE POPULATION IS PART OF THE FIXTURE. The prompt gate is RELATIVE -- p_floor is the
    MEDIAN prompt power over the PRNs plus k sigma -- so it only bites when there are on-peak
    satellites raising the bar. A fleet padded mostly with noise pulls the median down and E33
    sails through, which is exactly what the first version of this test did: it reported the
    latch as absent. That is the same relative-vs-absolute trap as the presence witness in #47.
    So: 6 on-peak satellites at spread powers (the bar), the satellites under test, and 3 noise
    PRNs -- 10+ rows, which _floor() needs before it will characterise a population at all."""
    full = dict(spec)
    for i in range(n_bright):                       # the bar: healthy, on-peak, varied
        b = dict(ONPEAK)
        b["p"] = ONPEAK["p"] * (0.85 + 0.05 * i)
        full[800 + i] = b
    for i in range(n_noise):
        full[900 + i] = NOISE
    fleet = make_fleet(full)
    it = iter(fleet)
    orig = F._get
    F._get = lambda url: next(it)
    try:
        return F.fleet_dll(["u%d" % i for i in range(12)], 0, 2, 3.0, 2.2,
                           deep_gate_prns=deep_gate_prns)
    finally:
        F._get = orig


class TestDeepGate(unittest.TestCase):

    def test_prompt_gate_excludes_the_off_peak_satellite(self):
        """The latch, reproduced: E33 is 13x its detection floor and is NOT present."""
        out = run({33: E33, 7: ONPEAK})
        self.assertFalse(out[33]["present"], "fixture no longer reproduces the latch")
        self.assertEqual(out[33]["present_gate"], "prompt")

    def test_deep_gate_admits_it(self):
        out = run({33: E33, 7: ONPEAK}, deep_gate_prns={33})
        self.assertTrue(out[33]["present"])
        self.assertEqual(out[33]["present_gate"], "deep")

    def test_the_two_gates_DISAGREE_on_the_witness_case(self):
        """⚠️ THE LOAD-BEARING ASSERTION. If both gates gave the same verdict here, every
        other test in this file would pass for the on-peak-biased gate too -- which is how
        the q -> prompt 'fix' was certified while changing nothing."""
        a = run({33: E33, 7: ONPEAK})[33]["present"]
        b = run({33: E33, 7: ONPEAK}, deep_gate_prns={33})[33]["present"]
        self.assertNotEqual(a, b)

    def test_deep_gate_still_refuses_noise(self):
        """The gate must buy pull-in, not trimming on nothing."""
        out = run({33: E33, 901: NOISE}, deep_gate_prns=True)
        self.assertFalse(out[901]["present"])

    def test_opt_in_leaves_everyone_else_on_the_prompt_gate(self):
        """The per-PRN rollout is the point: enabling 33 must not silently move the fleet."""
        base = run({33: E33, 7: ONPEAK})
        opt = run({33: E33, 7: ONPEAK}, deep_gate_prns={33})
        self.assertEqual(base[7]["present"], opt[7]["present"])
        self.assertEqual(opt[7]["present_gate"], "prompt")

    def test_missing_floor_falls_through_to_prompt(self):
        """No floor means no bar. Defaulting to present would trim on noise, so the deep gate
        must decline and leave the prompt verdict standing."""
        spec = dict(E33); spec["deep_floor"] = 0.0
        out = run({33: spec}, deep_gate_prns={33})
        self.assertEqual(out[33]["present_gate"], "prompt")

    def test_deep_gate_is_not_a_mere_inversion(self):
        """A healthy on-peak satellite must stay admitted under the deep gate."""
        self.assertTrue(run({33: E33, 7: ONPEAK}, deep_gate_prns={7})[7]["present"])

    def test_prompt_gate_verdict_depends_on_THE_POPULATION_not_the_satellite(self):
        """The pathology in one assertion, and it is worse than 'off-peak sats are excluded'.

        PRN 7 is UNCHANGED between these two runs -- same powers, same deep_snr, on its peak.
        Only its NEIGHBOURS differ. Against a weak population it is present; against a strong
        one the median rises above it and the same satellite is excluded. A presence gate whose
        answer depends on who else is in the sky cannot be the precondition for correcting an
        individual satellite's code phase. The deep gate is absolute (deep_snr vs that
        instance's own rectification floor), so it returns the same verdict in both."""
        weak = run({7: ONPEAK}, n_bright=0, n_noise=9)
        strong = run({7: ONPEAK}, n_bright=6, n_noise=3)
        self.assertTrue(weak[7]["present"])
        self.assertFalse(strong[7]["present"])
        for pop in (dict(n_bright=0, n_noise=9), dict(n_bright=6, n_noise=3)):
            self.assertTrue(run({7: ONPEAK}, deep_gate_prns={7}, **pop)[7]["present"],
                            "the deep gate must not depend on the population either")


if __name__ == "__main__":
    unittest.main(verbosity=2)
