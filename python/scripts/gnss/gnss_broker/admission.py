"""#90: the ABSENT-PRN re-seed admission gate.

WHAT THIS DECIDES. The spectrum-tau re-seed (#50) normally requires `present` -- the #49 deep
gate. On the searchless chains that gate is a ONE-WAY DOOR: off-peak kills presence, and with
no search to re-admit (#79 is gps_l5-only) the PRN can never earn its correction back. This
gate is the escape hatch: a SEEDED, absent PRN may fire the re-seed anyway, on evidence that
replaces what presence was guarding against.

WHY IT IS A CLASS AND NOT A BLOCK IN THE LOOP. Every one of the guards below was bought with
sky time on 2026-08-25 -- four flights in one evening, each one a ~10-line decision that could
only be exercised by restarting the broker against the live fleet, because the logic lived at
nesting depth 8 inside the DLL loop with its state hoisted 6000 lines away. The rules are a
pure function of (per-PRN history, this fit, the chain's population); making that explicit is
what lets `test_admission.py` replay all eight of that evening's fires offline in milliseconds.

THE RULES, in the order they are applied, each with the flight that bought it:

  1. ARMED         -- `--reseed-admit-absent`. Default off. (F3 DISARMED it 2026-08-25 23:3x.)
  2. SEEDED        -- the model says the sat is up; drop-on-set has already pruned the rest.
  3. WAS PRESENT   -- flight 3: a TRUE latch is present->absent. At process start EVERY sat is
                      absent, so the whole constellation accrued strikes and six fires landed
                      inside the startup solve -- including PRN 34, never present, whose
                      SIDELOBE-STABLE tau grew +0.62 -> +0.82 THROUGH two fires (a sidelobe
                      repeats its value, so it passes the consistency guard below).
  4. PAST STARTUP  -- flight 3a: presence FLAPS during the startup solve (PRN 9: present at
                      q 2.32 at t+28 s, absent by t+2 min, fired legally). A natural latch
                      takes tens of minutes to develop; startup convergence self-heals and
                      must not be steered. 600 s hold-off on the PROCESS clock.
  5. NOT BROWNED OUT -- flight 3b: four fires rode a band-wide e5b presence dip (7 -> 3 sats
                      over 6 min while e5a held steady). That is a #91-class fold event: the
                      seeds are not the fault and per-sat steps just add churn. Suppress while
                      the chain's present count is under `brownout_frac` of its own 600 s peak.
  6. TWO STRIKES   -- flight 2 (F2 tripped 2026-08-25 00:16: fires alternated sign chasing a
                      swinging fit, ratios up to 3.2 attached to contradictory taus seconds
                      apart -- ratio-at-poll-cadence is NOT significance in the latched
                      regime). A fire needs two qualifying fits that are DECORRELATED
                      (>= `min_gap_s` apart, i.e. different fold windows) and CONSISTENT
                      (|dtau| <= `tau_tol_chips`: a real offset repeats its value; noise and
                      span-edge sidelobes do not), outside a post-fire cooldown.

⚠️ STRIKE MEMORY SURVIVES NON-QUALIFYING CYCLES. The flight-2 harness cleared it on every
ratio dip, which is why the two-strike rule looked like it never fired. It is cleared only by
an INCONSISTENT qualifying fit (which replaces it), expiry, presence, or a fire.

⚠️ THE POPULATION HISTORY ACCRUES ONLY ON CYCLES THAT REACH RULE 5. That is deliberate and
preserved from the original: `note_population` is called from inside the gate, after rules 1-4
pass, so the 600 s peak is measured over the same cycles the guard is asked about. Sampling it
every cycle instead would change which baseline a fire sees.

⚠️ A FIRED SEED STEP IS WIPED BY THE NEXT dr_birth in ~20 s unless signal appears -- false
fires self-erase, real ones must catch within one birth cycle. This is why the caller also
opens an arming window (`_ft_hold`) on a fire: on the dead-reckon chains the slew returns the
seed to the model at the cap rate, so the DURABLE per-sat actuator is the C++ trim, and the
trim only pulls while armed.

@author Keith Vanderlinde
"""


class Decision:
    """What the gate decided, plus the lines the caller should log.

    `logs` is a list of (rate_limit_key, message, every_s) -- the gate does not own the
    logger, because the same decision is replayed in tests where there is none.
    """

    __slots__ = ("fire", "reason", "logs")

    def __init__(self, fire=False, reason="", logs=None):
        self.fire = fire
        self.reason = reason
        self.logs = logs if logs is not None else []

    def __repr__(self):
        return "Decision(fire=%r, reason=%r)" % (self.fire, self.reason)


class AdmissionGate:
    """The #90 absent-admission decision, with its four flights' guards.

    All wall-clock and cycle times are passed IN; the gate never reads the clock itself, so a
    test can replay an entire evening in one pass.
    """

    def __init__(self, armed=False, startup_hold_s=600.0, cooldown_s=180.0,
                 min_gap_s=60.0, max_gap_s=600.0, tau_tol_chips=0.5,
                 brownout_window_s=600.0, brownout_frac=0.6, brownout_min_base=4):
        self.armed = bool(armed)
        self.startup_hold_s = float(startup_hold_s)
        self.cooldown_s = float(cooldown_s)
        self.min_gap_s = float(min_gap_s)
        self.max_gap_s = float(max_gap_s)
        self.tau_tol_chips = float(tau_tol_chips)
        self.brownout_window_s = float(brownout_window_s)
        self.brownout_frac = float(brownout_frac)
        self.brownout_min_base = int(brownout_min_base)
        # PRN -> (tau, wall time) of the pending strike.
        self.pending = {}
        # PRN -> wall time of the last fire (post-fire cooldown stamp).
        self.cooldown = {}
        # PRNs that have been PRESENT at least once this process (rule 3's memory).
        self.was_present = set()
        # (t_cycle, n_present) history for rule 5, trimmed to brownout_window_s.
        self.population = []

    # ---- state updates -------------------------------------------------------------------

    def note_present(self, prn):
        """The PRN is present this cycle: it can never be mid-latch, so drop any pending
        strike, and remember it for rule 3."""
        self.pending.pop(prn, None)
        self.was_present.add(prn)

    def note_population(self, t_cycle, n_present):
        """Record this cycle's chain-wide present count, once per cycle, 600 s window."""
        if not self.population or self.population[-1][0] != t_cycle:
            self.population.append((t_cycle, n_present))
            while self.population and t_cycle - self.population[0][0] > self.brownout_window_s:
                self.population.pop(0)

    def browned_out(self, n_present):
        """True when the chain's present count has collapsed against its own recent peak."""
        base = max(n for _, n in self.population)
        return base >= self.brownout_min_base and n_present < self.brownout_frac * base, base

    # ---- the decision --------------------------------------------------------------------

    def decide(self, prn, tau, seeded, t_wall, t_cycle, n_present, uptime_s):
        """Should this absent PRN's qualifying spectrum fit fire a re-seed?

        Called ONLY for a fit that already passed the #50 qualification (armed PRN set, q under
        the ceiling, peak/floor ratio over the bar, a finite tau) AND whose presence flag is
        false. `seeded` is whether the PRN is in the seed table, `uptime_s` is process age.
        """
        # Rules 1-4: is this PRN's absence even the kind this gate is for?
        if not (self.armed and seeded and prn in self.was_present
                and uptime_s >= self.startup_hold_s):
            return Decision()

        # Rule 5: is the CHAIN the patient rather than this satellite?
        self.note_population(t_cycle, n_present)
        brown, base = self.browned_out(n_present)
        if brown:
            return Decision(reason="brownout", logs=[(
                "rs-admit-bw",
                "RESEED-ADMIT suppressed: band-wide presence dip (%d present vs %d baseline) "
                "-- #91-class, holding, not re-seeding" % (n_present, base), 60.0)])

        # Rule 6: two decorrelated, consistent strikes, outside the cooldown.
        if t_wall - self.cooldown.get(prn, 0.0) < self.cooldown_s:
            return Decision(reason="cooldown")

        pv = self.pending.get(prn)
        if pv and abs(tau - pv[0]) <= self.tau_tol_chips and t_wall - pv[1] < self.max_gap_s:
            if t_wall - pv[1] >= self.min_gap_s:
                self.pending.pop(prn, None)
                self.cooldown[prn] = t_wall
                return Decision(fire=True, reason="strike2")
            # Consistent but too fresh: HOLD the pending strike unchanged -- the
            # decorrelation clock keeps running.
            return Decision(reason="too-fresh")

        if pv is None or abs(tau - (pv[0] if pv else 0.0)) > self.tau_tol_chips \
                or t_wall - pv[1] >= self.max_gap_s:
            self.pending[prn] = (tau, t_wall)
            return Decision(reason="strike1")

        return Decision(reason="hold")
