"""The code loop's per-satellite decisions: the DLL control loop.

The DLL control loop is the only part of the fleet delay-lock stage that actuates; everything
else hung off that poll is instrumentation (see `instruments.py`).

On both paths, fleet and single combiner, a seed with no code_phase_rate (absent or zero) means
hold, not integrate. The uncalibrated receiver clock runs ~3.45 chips/s, ~11 chips per loop
round trip, and the seed's code_phase_rate feeds that forward. Without it the trim would face the
whole 11 chips, far outside the +-0.5 chip pull-in, and the failure would read as the DLL
diverging rather than as the feed-forward being absent.

The `DLL:` line built here lists only the satellites that passed this cycle's gates, so a
statistic computed over it measures survivors: a satellite whose q craters leaves the sample.
Judge on a series that keeps every seeded satellite (ctx.qpop, the QPOP line).

@author Keith Vanderlinde
"""

from gnss_broker.transport import _log_rl
from gnss_broker import combdll


def stage_dll_control(ctx):
    """3c: the DLL control loop -- per satellite, decide and integrate the code trim.

    The only part of the DLL stage that actuates. Appends one ctx.dllp.report entry (the `DLL:`
    line) per satellite that passed this cycle's gates; see the module docstring for the rate
    gate and for why that line measures survivors."""
    for prn in list(ctx.seeds):
        rec = ctx.status.get(prn, {})
        fl = ctx.dllp.fleet.get(prn)
        # Rate gate (docs/CHORD_GNSS_SHARED_DLL.md section 7): the trim absorbs only the
        # residual left by the seed's code_phase_rate feed-forward, so with an absent or zero
        # rate it holds on either path (see the module docstring).
        if not float(ctx.seeds[prn].get("code_phase_rate", 0.0) or 0.0):
            _log_rl(
                "dll-norate-%d" % prn,
                "%sDLL PRN %d: no live code_phase_rate, holding trim"
                % ("fleet " if fl is not None else "", prn),
            )
            continue
        if fl is not None:
            # Fleet path: gate on `present` (fleet.apply_presence: q and prompt power against
            # floors measured each cycle from the noise probes, plus the deep-gate and displaced
            # routes), never on a constant bar. Summing instances tightens the noise
            # distribution instead of raising q, so a fixed bar suits one fleet size only.
            # sig_of(rec) is one instance's view of a fraction of the band and would veto
            # exactly the satellites the fleet sum exists for.
            if not fl["present"]:
                continue
            # One integration per new window: an exact integer test on the window's hop, where
            # the single-combiner path below can only watch for a changed float.
            if fl["hop"] == ctx.dls.last_hop.get(prn):
                continue
            ctx.dls.last_hop[prn] = fl["hop"]
            disc = fl["disc"]
        else:
            disc = float(rec.get("dll_disc", 0.0))
            if ctx.sig_of(rec) < ctx.args.lock_snr or disc == 0.0:
                continue
            # One integration per new measurement: the combiner emits a fresh disc once per
            # integration window, and this loop may poll several times within one window;
            # re-integrating a stale value over-applies the gain. A changed value marks a
            # fresh emit.
            if disc == ctx.dls.last.get(prn):
                continue
            ctx.dls.last[prn] = disc
        # Leaky integrator: the trim mean-reverts, so discriminator noise cannot random-walk it
        # to the clamp; noise averages over ~1/leak updates. The leak also caps the correction:
        # steady state is trim = (gain/leak)*tau with |tau| <= 0.25*(spacing/0.5) chips
        # (combdll.dll_tau), so at 0.5-chip spacing, gain 0.25 and leak 0.05 nothing beyond
        # 1.25 chips is reachable and the +-3.0 chip clamp never binds.
        #
        # The fleet path, where `present` already confirms signal, takes dll_leak_present; the
        # single-combiner path, which has no such gate, takes dll_leak. Both are frozen at 0.05
        # in cli._FROZEN, which says why a smaller leak on the fleet path is worse.
        leak = ctx.args.dll_leak_present if fl is not None else ctx.args.dll_leak
        # One expression, one place: the update is combdll.dll_integrate, the twin of
        # gnss::dll_integrate, and scripts/gnss/fleetdll_gate.py compares the C++ and Python
        # arms on identical bytes.
        #
        # Not while the C++ fleet loop owns this PRN. Two integrators on one code error
        # double-apply during pull-in and fight at the null, and they stack invisibly: this
        # trim is added to the seed's code phase at POST time (seeding.py), while the C++ trim
        # is applied at the tracker. Authority is per PRN and follows dls.armed_last, the set
        # last posted to the gather (trimarm.py), not what this cycle is about to decide.
        # Python integrates the PRNs outside that set and stands down for each one the fast
        # loop takes, so while the POSTs succeed no PRN is trimmed by both and none falls
        # between them. With --fleet-trim-url set, trimarm arms every PRN present within the
        # last fleet_trim_hold_s, so Python trims a PRN only on the cycle it turns present
        # after a longer absence, or on the single-combiner path while it is not held armed.
        # Standing down it still measures: the dedup state advances and the DLL line still
        # reports the PRN.
        #
        # The standing trim is not popped at handover: it stays in the seed as a valid offset,
        # and the C++ loop corrects the residual. On release the C++ trim leaks away gradually
        # (the gather keeps posting the decaying value) while this integrator resumes from its
        # standing value, so what the C++ loop had corrected must be re-acquired here.
        if prn not in ctx.dls.armed_last:
            ctx.dls.trim[prn] = combdll.dll_integrate(
                ctx.dls.trim.get(prn, 0.0),
                disc,
                ctx.args.dll_gain,
                leak,
                3.0,
                ctx.args.dll_spacing,
            )
        ctx.dllp.report.append(
            "PRN %d disc %+.3f trim %+.2f%s"
            # .get: a PRN the C++ loop owns may never have had a Python trim; it reads 0.
            % (
                prn,
                disc,
                ctx.dls.trim.get(prn, 0.0),
                ""
                if fl is None
                else " [fleet %d/%d q %.2f p %.1fx%s]"
                % (
                    fl["n_src"],
                    len(ctx.dll_combiners),
                    fl["q"],
                    fl["p_pow"] / fl["p_med"] if fl.get("p_med") else 0.0,
                    # Which gate admitted this PRN, printed only for the deep gate so the
                    # deep-gated set is identifiable in the log without diffing against the
                    # prompt-gated majority.
                    ""
                    if fl.get("present_gate") != "deep"
                    else " DEEP %.1f/%.1f"
                    % (fl.get("deep_gate_snr", 0.0), fl.get("deep_gate_floor", 0.0)),
                ),
            )
        )
