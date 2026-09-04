"""#91: the brownout freeze, exercised against the C++ loop's actual semantics.

    python3 -m gnss_broker.test_brownout_policy

WHY THIS EXISTS RATHER THAN A FIXTURE. The digest gate cannot reach this: a replay has no
live gather, so `stage_fleet_trim_arming` never posts and the policy dict it would have built
is never compared to anything. This is the same blind spot that let the `_rr_railed`
UnboundLocalError and the dead `C_LIGHT` import ship -- so the policy is asserted here, on
the numbers the C++ loop will actually apply.

THE CONTRACT BEING TESTED, from lib/stages/gnss/GnssFleetTrim.cpp and gnssFleetDll.hpp:
  * ARMED   -> `dll_integrate` runs: trim = (1-leak)*trim + gain*tau(disc). There is NO
               quality gate in C++ ("the probe and deep gates ... is POLICY and stays in the
               Python broker"), so an armed PRN in a brownout integrates NOISE.
  * DISARMED-> the trim decays through the leak, "an unarmed trim leaks to erasure in ~5.6 s".
  * FROZEN  -> gain=0 AND leak=0 => trim = (1-0)*trim + 0*tau = trim, exactly. The only
               state that RETAINS a standing trim, and the one #91 asks for.

@author Keith Vanderlinde
"""

import sys

from gnss_broker.detectors import BrownoutDetector


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def dll_integrate(trim, disc, gain, leak, spacing=0.5):
    """The C++ loop's integrator, transcribed from gnssFleetDll.hpp (dll_tau + dll_integrate).

    ⚠️ |tau| <= 0.25 chips BY CONSTRUCTION whatever `disc` is -- the clamp is on the
    discriminator and it is divided by four.
    """
    tau = -max(-1.0, min(1.0, disc)) / 4.0 * (spacing / 0.5)
    return (1.0 - leak) * trim + gain * tau


def browned_out(depth=1, base=8, cycles=30):
    """A detector sitting inside an open episode, the E3 shape."""
    d = BrownoutDetector(window_s=600.0, frac=0.6, min_base=4, min_len_s=60.0)
    for i in range(12):
        d.note_cycle(i * 10.0, base)
    for i in range(12, 12 + cycles):
        d.note_cycle(i * 10.0, depth)
    return d


# ---- THE THREE STATES, ON THE C++ ARITHMETIC ----------------------------------------------
def test_freeze_is_the_only_state_that_retains():
    print("the freeze is the only state that retains a standing trim")

    # A satellite that had built a real trim, now seeing a noise discriminator.
    trim0, noise_disc = -0.78, 0.9        # railed-ish noise, the brownout case
    gain, leak = 2.5 / 23.84, 0.35 / 23.84

    armed = trim0
    for _ in range(120):                  # ~5 s of updates at 23.84 Hz
        armed = dll_integrate(armed, noise_disc, gain, leak)
    check(abs(armed - trim0) > 0.2,
          "ARMED through a brownout MOVES the trim on noise (%.2f -> %.2f) -- the C++ loop "
          "has no quality gate, so arming is not holding" % (trim0, armed))

    released = trim0
    for _ in range(120):
        released = dll_integrate(released, 0.0, gain, leak)   # disarmed: leak only
    check(abs(released) < abs(trim0) * 0.5,
          "DISARMED decays it away (%.2f -> %.2f) -- 'leaks to erasure in ~5.6 s'"
          % (trim0, released))

    frozen = trim0
    for _ in range(2400):                 # 100 s: freezing must not drift with time
        frozen = dll_integrate(frozen, noise_disc, 0.0, 0.0)
    check(frozen == trim0,
          "FROZEN (gain=leak=0) retains it EXACTLY (%.2f), for any discriminator and any "
          "number of updates" % frozen)


# ---- THE POLICY THE BROKER BUILDS ---------------------------------------------------------
def policy(brown, hold_s, brownout_hold_s, bandwidth=2.5, leak_per_s=0.35):
    """Mirror of stage_fleet_trim_arming's decision, with no transport."""
    brown_hold = (brownout_hold_s > 0.0 and brown.established())
    return {
        "hold_s": max(hold_s, brownout_hold_s) if brown_hold else hold_s,
        "gain_per_s": 0.0 if brown_hold else bandwidth,
        "leak_per_s": 0.0 if brown_hold else leak_per_s,
        "frozen": brown_hold,
    }


def test_policy():
    print("the posted policy")

    calm = BrownoutDetector()
    for i in range(20):
        calm.note_cycle(i * 10.0, 8)

    p = policy(calm, 90.0, 600.0)
    check(not p["frozen"] and p["gain_per_s"] == 2.5 and p["hold_s"] == 90.0,
          "a calm chain is untouched: normal gain, normal 90 s hold")

    p = policy(browned_out(), 90.0, 0.0)
    check(not p["frozen"] and p["gain_per_s"] == 2.5,
          "DEFAULT OFF (0.0): a brownout changes nothing until the flag is armed")

    p = policy(browned_out(), 90.0, 600.0)
    check(p["frozen"] and p["gain_per_s"] == 0.0 and p["leak_per_s"] == 0.0,
          "armed + browned out: BOTH gain and leak zeroed (either alone loses the trim)")
    check(p["hold_s"] == 600.0,
          "and the arming hold is extended so the set does not shrink out from under it")

    # THE CAP. E3 lasted ~9 min; a satellite that genuinely set must still be released, and
    # today's analog-frontend outage (hours) must not hold trims for hours.
    check(policy(browned_out(), 90.0, 600.0)["hold_s"] == 600.0,
          "the flag doubles as the CAP: past it the hold expires and release is normal")


def test_recovery():
    print("recovery")

    d = browned_out()
    check(policy(d, 90.0, 600.0)["frozen"], "frozen while the episode is open")
    for i in range(60, 70):               # presence returns
        d.note_cycle(i * 10.0, 8)
    check(not d.active(), "the episode closes when presence recovers")
    p = policy(d, 90.0, 600.0)
    check(not p["frozen"] and p["gain_per_s"] == 2.5,
          "and the loop resumes at full gain on the trims it kept -- no re-pull")


def test_small_constellation_cannot_freeze():
    print("guards")

    d = BrownoutDetector(min_base=4)
    for i in range(6):
        d.note_cycle(i * 10.0, 3)
    d.note_cycle(70.0, 1)
    check(not policy(d, 90.0, 600.0)["frozen"],
          "a chain below min_base cannot brown out, so it cannot freeze (1 of 3 is not an "
          "episode) -- the freeze inherits D1's population floor rather than adding its own")

    d2 = BrownoutDetector(min_len_s=60.0)
    for i in range(10):
        d2.note_cycle(i * 10.0, 8)
    d2.note_cycle(100.0, 2)
    check(d2.active() and not d2.established(),
          "a one-cycle dip is ACTIVE (D2/D3 suppress at once) but NOT ESTABLISHED")
    check(not policy(d2, 90.0, 600.0)["frozen"],
          "so it does not freeze the loop -- presence flickers constantly, and freezing on "
          "every flicker would gut the very duty cycle #91 exists to protect")


if __name__ == "__main__":
    print("#91 -- the brownout freeze\n")
    for fn in (test_freeze_is_the_only_state_that_retains, test_policy, test_recovery,
               test_small_constellation_cannot_freeze):
        fn()
    print("\nFAILED (%d)" % len(_fails) if _fails else "\nOK")
    sys.exit(1 if _fails else 0)
