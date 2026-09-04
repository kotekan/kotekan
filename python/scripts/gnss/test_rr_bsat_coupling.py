"""#33 gap 3: the carrier-aided code loop (rr_bsat_chips_per_m).

state_filter.py already implements d(b_sat)/dt = rr_bsat_chips_per_m * rrate in the
F matrix, defaulted to 0.0 with an explicit demand that the SIGN be measured on sky
before arming. These tests pin the structure so the arming, when it comes, changes
exactly one number:

  1. 0.0 is the identity -- bit-identical trajectories against a default filter.
  2. The coupling moves b_sat at the documented scale (f_chip/c), and only when the
     SAME key holds both rows -- the broker keys both sides (tag, prn), verified at
     the feed sites (cycle() at the joint feed, update_rrate() at JRR-KCOH).
  3. THE MASS-BIRTH RE-EARN (chord-joint-filter-mass-birth): the divergence mode this
     coupling gives a new route into is "offset read as RATE". With coupling ON, a
     simultaneous multi-sat birth with static per-sat biases must still converge --
     the rrate rows must not absorb the biases and then drag b_sat via the coupling.

Run: python3 -m unittest test_rr_bsat_coupling
"""
import unittest

from gnss_broker.state_filter import JointReceiverState

L = 204600.0
F_L5 = 1176.45e6
CHIP_HZ = 10.23e6
C = 299792458.0
K_PHYS = CHIP_HZ / C          # 0.03412 chips per (m/s)


def mk(**kw):
    kw.setdefault("code_len", L)
    kw.setdefault("ref_band", "L5")
    kw.setdefault("clk0", 150.0)   # warm start; cold bootstrap is test_joint_bootstrap's
    return JointReceiverState(**kw)


class TestIdentity(unittest.TestCase):
    def test_zero_coupling_is_bit_identical(self):
        a, b = mk(), mk(rr_bsat_chips_per_m=0.0)
        for k in range(60):
            t = float(k)
            for prn, bias in ((3, 1.5), (7, -2.0), (11, 0.4)):
                key = ("G", prn)
                a.update(key, 150.0 + bias, 0.3, t)
                b.update(key, 150.0 + bias, 0.3, t)
                a.update_rrate(key, -0.5 * prn, t, F_L5, sigma_hz=0.2)
                b.update_rrate(key, -0.5 * prn, t, F_L5, sigma_hz=0.2)
            a.predict(t + 1.0)
            b.predict(t + 1.0)
        self.assertEqual(list(a.x), list(b.x))


class TestCouplingMoves(unittest.TestCase):
    def test_predict_drags_bias_at_the_documented_scale(self):
        """rrate = +1 m/s held for 60 s of pure predicts must move b_sat by k*60
        chips -- exactly, it is one F-matrix term -- and k=0 must not move it."""
        for k_c, want in ((+K_PHYS, +K_PHYS * 60.0), (-K_PHYS, -K_PHYS * 60.0),
                          (0.0, 0.0)):
            js = mk(rr_bsat_chips_per_m=k_c)
            key = ("G", 5)
            t = 0.0
            for i in range(20):
                js.update(key, 150.0, 0.3, t)
                t += 1.0
            js.predict(t)              # sync the filter clock: update() predicted to t-1
            js._add_rrate(key, 1.0, t)
            i_b = js._idx[key]
            b0 = float(js.x[i_b])
            for i in range(60):
                t += 1.0
                js.predict(t)
            moved = float(js.x[i_b]) - b0
            self.assertAlmostEqual(moved, want, places=9,
                                   msg="k=%+.5f moved %+.6f want %+.6f"
                                       % (k_c, moved, want))

    def test_no_coupling_without_a_matching_bias_row(self):
        """A key with an rrate row but no b_sat row must not couple anywhere --
        specifically it must not birth a bias row as a side effect of predict."""
        js = mk(rr_bsat_chips_per_m=K_PHYS)
        js._add_rrate(("G", 9), 2.0, 0.0)
        n0 = js.x.size
        for i in range(30):
            js.predict(float(i + 1))
        self.assertEqual(js.x.size, n0)
        self.assertNotIn(("G", 9), js._idx)


class TestMassBirthReEarned(unittest.TestCase):
    def _run(self, k_c):
        """12 sats born the same second, static true biases, honest code + carrier
        measurements with the carrier reading TRUE rrate = 0 (the model error is a
        static offset, NOT a rate -- the exact confusion that diverged before).
        Returns (worst demeaned bias error, worst |rrate|)."""
        js = mk(rr_bsat_chips_per_m=k_c)
        truth = {p: ((-1) ** p) * (0.5 + 0.25 * (p % 5)) for p in range(1, 13)}
        mean_b = sum(truth.values()) / len(truth)
        t = 0.0
        for cyc in range(300):
            for p, b in truth.items():
                key = ("G", p)
                js.update(key, 150.0 + b, 0.3, t)
                js.update_rrate(key, 0.0, t, F_L5, sigma_hz=0.2)
            js.gauge()
            js.gauge_rrate()
            js.predict(t + 1.0)
            t += 1.0
        errs = [abs(js.wrap(float(js.x[js._idx[("G", p)]]) - (b - mean_b)))
                for p, b in truth.items()]
        rrs = [abs(js.rrate(("G", p))) for p in truth]
        return max(errs), max(rrs)

    def test_birth_converges_with_coupling_on_either_sign(self):
        base_err, base_rr = self._run(0.0)
        for k_c in (+K_PHYS, -K_PHYS):
            err, rr = self._run(k_c)
            # a rate-absorbs-offset runaway shows up as chips of bias error and
            # m/s of phantom rrate -- orders beyond these bars
            self.assertLess(err, max(3.0 * base_err, 0.15),
                            msg="k=%+.5f bias err %.4f (base %.4f)"
                                % (k_c, err, base_err))
            self.assertLess(rr, max(3.0 * base_rr, 0.5),
                            msg="k=%+.5f phantom rrate %.4f (base %.4f)"
                                % (k_c, rr, base_rr))


if __name__ == "__main__":
    unittest.main()
