"""The per-chain cycle context: what a stage needs that is not its own local state.

Stages read their inputs off this object rather than out of main()'s closure, so the interface
between the broker loop and its stages is named in one place: `__slots__`. It has two halves.

  Stable      bound once during startup and never rebound inside the cycle loop, so safe to
              hold by reference. Several are mutable containers (`seeds`, `cp_held`,
              `dr_state`) that are mutated in place, never replaced; if one is ever reassigned
              per cycle, this context silently goes stale.

  Per-cycle   rebound on every pass: `t0` and `best` through `begin_cycle()`, and `status`,
              `pred`, `up` and `probe_set` by direct assignment where they are computed. A
              stage that reads one off the context reads this cycle's value; a stage that
              captured it at construction would read the first cycle's forever.

`t0` is the frozen cycle clock: wall time sampled once per pass (transport._Transcript.tick), so
every gate, age and EMA in a pass evaluates at one instant rather than across the pass's own
processing time. A stage that needs real elapsed time calls `time.time()` explicitly and says why.

This is a reference holder, not a facade: no properties, no derived values, no logic. Anything
computed belongs in the stage that computes it or on one of the owner objects (`dllp`, `drp`,
...), so that behaviour lives in one place.

@author Keith Vanderlinde
"""


_UNSET = object()


class ChainContext(object):
    """Reference holder for one chain's cycle. See the module docstring for the two halves."""

    __slots__ = (
        # ---- identity and configuration -----------------------------------------------
        "args",
        "band_id",
        "chain_id",
        "code_len",
        "telem_chain",
        "base",
        "alm_sys",
        "alm_min_prn",
        "lc_seg",
        "lc_epoch",
        # ---- services and endpoints ---------------------------------------------------
        "rx",
        "publisher",
        "telem_client",
        "detectors",
        "dll_combiners",
        "geom_post_t",
        "spectrum_endpoints",
        "sig_of",
        "n2_combiners",
        "last_dets",
        "decfb",
        "decfb_log_t",
        "dr_bad",
        "fe_axis",
        "fe_off",
        "capable",
        "capable_t",
        "alm_now",
        "trackers",
        "joint_consume",
        "broker_t0",
        "dr_eph_mod",
        "dr_min_prn",
        "hist_len",
        "max_gap_hops",
        "cp_to_seed_currency",
        "sig_of_last",
        # Helper callables the stages share: closures over this chain's configuration in main().
        "dh_obs",
        "cp_predicted",
        "joint_state",
        "track_ok",
        "p2c_tick",
        "p2c_hold",
        "decoded_entries",
        # The newest F-engine hop this chain has seen (the combiner's pow_hop), for scheduling
        # a fleet-simultaneous PRN swap (prnmap._at_hop), and the cycle-clock time it was read,
        # so the scheduler can age a sample left from an earlier cycle. Both None whenever the
        # axis is unknown, never 0.0, which would read as a valid hop near the epoch and
        # schedule every swap into the past.
        "fe_hop_now",
        "fe_hop_t",
        # ---- owner objects (each stage's own state lives on its owner) ----------------
        "dllp",
        "drp",
        "cb",
        "nho",
        "dls",
        "hold",
        "cpt",
        "rf",
        "nav",
        "qpop",
        "brown",
        "saw",
        "prnmap",
        # ---- long-lived tables, mutated in place --------------------------------------
        "seeds",
        "dr_state",
        "bsat",
        "cp_held",
        "dr_untrusted",
        "est_last",
        "kcoh_rates",
        "rf_last",
        "elem_arch_t",
        "elem_poll_t",
        "mp_cooldown",
        "mp_flipped",
        "mp_last_det",
        "almanac_sats",
        "brdc_alm",
        "det_fresh",
        "innov_hist",
        "minnov_hist",
        "p2c",
        "dop_rate_fitted",
        "dop_rate_rejected",
        "cp_rate_rejected",
        "dll_hop_window",
        "deep_gate",
        "dg_auto_last",
        "est_next",
        "birth_steps",
        # {combiner url: (newest pow_hop, first seen at it)}, for fleet.live_instances
        "fleet_hop_hist",
        # ---- per-cycle: written directly by the stage that computes them ---------------
        # Attributes rather than loop locals because a module-level stage can assign
        # `ctx.have_sig`, where it could not declare `nonlocal have_sig`.
        "t0",
        "best",
        "status",
        "pred",
        # {prn: pred-shaped row} for the satellites --signal-capability excluded from `pred`
        # (with ephemeris, but unable to broadcast this signal); filled only under --probe-dark,
        # for the probe selector and the PRN map.
        "pred_dark",
        "up",
        "probe_set",
        "utc0_sample0",
        "coast_polls",
        "have_sig",
        "payload",
        "jrc",
        "rr_cmd_new",
        "la_samples",
        "fitted",
        "cl_report",
        "dr_pd",
        "dr_pd0",
        "dr_pd2",
        "transit_pd",
    )

    # Slots whose "not yet known" value is not None, as factories. They must be factories, not
    # values: broker_multi runs several chains in one process, and a mutable literal here would
    # be one dict on the class, shared by every chain's context.
    #
    # `utc0_sample0` is 0.0 rather than None because it is used in arithmetic before the first
    # fetch succeeds; None would raise there, while 0.0 reads as "no anchor yet" in the truth
    # tests that guard it.
    DEFAULTS = {
        "utc0_sample0": float,
        "status": dict,
        "birth_steps": dict,
        "fleet_hop_hist": dict,
    }

    def __init__(self, **kw):
        for k in self.__slots__:
            if k in kw:
                setattr(self, k, kw[k])
            else:
                make = self.DEFAULTS.get(k)
                setattr(self, k, make() if make is not None else None)

    def begin_cycle(self, t0=_UNSET, best=_UNSET):
        """Refresh `t0` or `best`, called where the loop rebinds each.

        The sentinel, not None, marks an argument as absent, so None can be set: a "skip if
        None" default would leave the previous cycle's value in place.

        Only values assigned exactly once per pass belong here. `pred`, `up`, `status` and
        `probe_set` are written straight onto the context instead, because each is rebound
        more than once in a pass (`probe_set` starts empty and is replaced when noise probes
        are selected); a refresh call after the first assignment would leave the context
        holding the earlier value while the loop used the later one.
        """
        if t0 is not _UNSET:
            self.t0 = t0
        if best is not _UNSET:
            self.best = best
