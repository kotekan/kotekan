"""The per-chain cycle context: what a stage needs that is not its own local state.

WHY THIS EXISTS. The broker's stages were extracted as NESTED routines, which cost nothing
because a closure supplies every free name automatically. That is also the problem: it makes
the interface invisible. Measured 2026-08-26, the 29 stages between them read **210 distinct
free names** out of `main()`, and no stage declares which. Moving a stage into its own module
is impossible until that set is named, and reading one honestly is hard for the same reason.

THE SPLIT BELOW IS LOAD-BEARING, not organisational:

  STABLE      bound once during startup and NEVER rebound inside the cycle loop (measured: 39
              of the 45 names the stages actually share). Safe to hold by reference forever.
              Note that several are mutable containers -- `seeds`, `cp_held`, `dr_state` -- and
              they are mutated IN PLACE, never replaced, which is exactly why holding a
              reference is sound. If any of them ever starts being reassigned per cycle, this
              context silently goes stale, so that is the rule to preserve.

  PER-CYCLE   rebound on every pass: `t0`, `best`, `status`, `pred`, `up`, `probe_set`. These
              are refreshed by `begin_cycle()` at the top of the loop. A stage that reads one
              of these off the context is reading THIS cycle's value; a stage that captured it
              at construction time would read the first cycle's forever.

⚠️ `t0` IS THE FROZEN CYCLE CLOCK, NOT THE WALL CLOCK. Every gate, age and EMA in a pass
evaluates at one instant rather than smearing over the pass's own processing time. Where a
stage genuinely needs real elapsed time -- the #90 admission gate's decorrelation window is the
standing example, because it is a statement about fold windows and not about cycles -- it must
call `time.time()` explicitly and say why.

⚠️ THIS IS A REFERENCE HOLDER, NOT A FACADE. It deliberately does no work: no properties, no
derived values, no logic. Anything computed belongs in the stage that computes it or on one of
the owner objects (`dllp`, `drp`, `handover`, ...). A context that starts computing becomes a
second place where behaviour lives, and then nobody can tell which one a bug is in.

@author Keith Vanderlinde
"""


_UNSET = object()


class ChainContext(object):
    """Reference holder for one chain's cycle. See the module docstring for the two halves."""

    __slots__ = (
        # ---- identity and configuration -----------------------------------------------
        "args", "band_id", "chain_id", "code_len", "telem_chain", "base",
        "alm_sys", "alm_min_prn", "lc_seg", "lc_epoch",
        # ---- services and endpoints ---------------------------------------------------
        "rx", "publisher", "telem_client", "detectors", "dll_combiners",
        "spectrum_endpoints", "spec_writer", "state_dir", "xb_read_dir", "sig_of",
        "combiner", "gating", "capable", "receiver_state", "alm_now",
        "trackers", "joint_consume", "broker_t0", "dr_eph_mod", "dr_min_prn",
        "hist_len", "max_gap_hops", "q_alias_hz",
        "carrier_explain_hz", "carrier_verify_emits",
        "fuse_cached", "cp_to_seed_currency", "sig_of_last",
        # ---- owner objects (each stage's own state lives on its owner) ----------------
        "dllp", "drp", "handover", "adm_gate", "g3_ramp", "cb", "car", "wd", "nho",
        "dls", "hold", "cpt", "rf",
        # ---- long-lived tables, mutated in place --------------------------------------
        "seeds", "dr_state", "bsat", "cp_held", "dr_untrusted",
        "est_last", "kcoh_rates", "rf_last", "elem_arch_t", "elem_poll_t",
        "mp_cooldown", "mp_flipped", "mp_last_det",
        "almanac_sats", "brdc_alm", "det_fresh", "state_w", "clk_persist_t",
        "innov_hist", "minnov_hist", "p2c", "dop_rate_fitted", "dop_rate_rejected",
        "dll_hop_window", "deep_gate", "dg_auto_last", "est_next",
        # ---- per-cycle: written directly by the stage that computes them ---------------
        # Attributes rather than loop locals for one reason: a module-level stage can assign
        # `ctx.have_sig`, where it could never declare `nonlocal have_sig`. That is the whole
        # escape from the nonlocal wall.
        "t0", "best", "status", "pred", "up", "probe_set",
        "utc0_sample0", "xb_pred", "coast_polls", "have_sig",
        "la_samples", "fitted", "cl_report", "dr_pd", "dr_pd0", "dr_pd2",
    )

    # Slots whose "not yet known" value is not None. `utc0_sample0` is 0.0 because it is
    # compared numerically before the first fetch succeeds, and None would raise there rather
    # than read as "no anchor yet".
    DEFAULTS = {"utc0_sample0": 0.0}

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k, self.DEFAULTS.get(k)))

    def begin_cycle(self, t0=_UNSET, best=_UNSET, status=_UNSET, probe_set=_UNSET):
        """Refresh the per-cycle half, at each point one of these is rebound.

        Not a single call at the top of the pass: these do not all become available at the
        same moment (`pred` and `up` are produced by the almanac stage partway through), so
        each is refreshed where it is actually assigned.

        ⚠️ THE SENTINEL IS NOT PEDANTRY. A value here can legitimately be None on a given
        cycle, and a "skip if None" default would silently leave the PREVIOUS cycle's value
        in place -- stale, plausible, and wrong. None must be settable.

        `pred` and `up` are NOT here: the almanac stage writes them straight onto the context
        as attributes, which is what lets that stage live in a module at all.
        """
        if t0 is not _UNSET:
            self.t0 = t0
        if best is not _UNSET:
            self.best = best
        if status is not _UNSET:
            self.status = status
        if probe_set is not _UNSET:
            self.probe_set = probe_set
