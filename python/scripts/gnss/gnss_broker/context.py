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


class ChainContext(object):
    """Reference holder for one chain's cycle. See the module docstring for the two halves."""

    __slots__ = (
        # ---- identity and configuration -----------------------------------------------
        "args", "band_id", "chain_id", "code_len", "telem_chain", "base",
        "alm_sys", "alm_min_prn", "lc_seg", "lc_epoch",
        # ---- services and endpoints ---------------------------------------------------
        "rx", "publisher", "telem_client", "detectors", "dll_combiners",
        "spectrum_endpoints", "spec_writer", "state_dir", "xb_read_dir", "sig_of",
        # ---- owner objects (each stage's own state lives on its owner) ----------------
        "dllp", "drp", "handover", "adm_gate", "g3_ramp",
        # ---- long-lived tables, mutated in place --------------------------------------
        "seeds", "dr_state", "bsat", "cp_held", "dr_untrusted", "t_now",
        "est_last", "kcoh_rates", "rf_last", "elem_arch_t", "elem_poll_t",
        "mp_cooldown", "mp_flipped", "mp_last_det",
        # ---- per-cycle, refreshed by begin_cycle() ------------------------------------
        "t0", "best", "status", "pred", "up", "probe_set",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def begin_cycle(self, t0=None, best=None, status=None, pred=None, up=None,
                    probe_set=None):
        """Refresh the per-cycle half. Called once at the top of each pass, as each value
        becomes available -- they are not all known at the same point in the cycle, so this
        takes whatever is being set and leaves the rest alone."""
        if t0 is not None:
            self.t0 = t0
        if best is not None:
            self.best = best
        if status is not None:
            self.status = status
        if pred is not None:
            self.pred = pred
        if up is not None:
            self.up = up
        if probe_set is not None:
            self.probe_set = probe_set
