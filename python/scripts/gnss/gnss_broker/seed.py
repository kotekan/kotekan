# The seed object (task #83, Phase 1 -- docs/CHORD_CONTROL_AUDIT.md section 2).
#
# The audit's root finding: the seed was a bare dict with ~14 writers per cycle, half of
# them REPLACING fields outright, arbitrated by ~20 mode booleans and cycle order -- and
# the arbitration policy was written nowhere. This module is where it gets written.
#
# Phase 1 is BEHAVIOR-PRESERVING: `Seed` subclasses dict, so every existing read, `in`,
# `.get`, `**splat`, and json path is unchanged and the equivalence gate
# (scripts/gnss/broker_equiv.py) holds the POST stream byte-identical across the
# migration. What it adds is ATTRIBUTION: every write carries a declared owner and the
# epoch (ref_hop) the value is valid at, recorded beside the data (never in it -- the
# provenance lives in __slots__, invisible to json.dumps and to `dict(prn=prn, **seed)`).
# An unattributed write (`seed[k] = v` from unmigrated code) is still legal and records
# its call site, so migration is field-by-field and the untouched paths lose nothing.
#
# What this buys before any behavior changes:
#   * SEEDAUDIT steps can name the writer that moved each field (the ~20-boolean
#     arbitration becomes observable per published seed, audit section 7.7);
#   * epoch skew -- a field valid at one ref_hop shipped beside a tuple valid at
#     another, which is #80's disease exactly -- becomes a measurable, loggable fact
#     instead of a code-reading exercise.
#
# THE FIELD CONTRACT (audit section 6, as data; cudaGnssChordTrack.hpp:61-77 is the
# consumer):

FIELDS = {
    "doppler_hz": "Hz, absolute; valid AT ref_hop, never at now (7cb011c50)",
    "code_phase_chips": "chips, absolute ARGUMENT back-referenced to sample 0; means "
                        "nothing unpaired from doppler_hz (the cp-currency rule)",
    "code_phase_rate": "chips/hop, RESIDUAL on the geometry propagate_seed feeds forward",
    "ref_hop": "hops, absolute; IS the tuple's epoch",
    "doppler_rate_hz_s": "Hz/s, absolute, at ref_hop",
    "code_phase_at_ref_chips": "chips mod LL, PHYSICAL phase at ref_hop's LAST sample; "
                               "the tracker PREFERS this over code_phase_chips when >= 0 "
                               "(gnssSeedTransport.cpp:325-327)",
}

# THE WRITERS (audit section 2, the reverse-engineered arbitration -- now the declared
# one). REPLACE = measurement overwrites forecast wholesale; CORRECT = forecast + bounded
# step. Owners are the strings the migration stamps at each site; a new writer must add
# itself here or its steps audit as "?file:line".
OWNERS = {
    "det":          "REPLACE  detection birth / re-detection (whole tuple from the search)",
    "dop_select":   "REPLACE  doppler source selection (pred | dr | det, last word wins)",
    "dop_model":    "REPLACE  doppler_rate_hz_s from BRDC central/forward diff or almanac",
    "dop_fit":      "REPLACE  doppler_rate_hz_s from the measured-slope fit (model-vetoed)",
    "cp_fit":       "REPLACE  cp-rate fit: code_phase_rate + ref_hop + code_phase_chips",
    "nh_lift":      "REPLACE  overlay period lift from the search's cp_long / at-ref phase",
    "cl_assist":    "REPLACE  overlay period from wall-clock CL time assist",
    "dop_force":    "REPLACE  --force-doppler-rate replay-bench override",
    "dop_clamp":    "CORRECT  single-cycle doppler step bound (dop_max_rate_hz)",
    "hold_freeze":  "REPLACE  hold-on-lock: previous tuple rides over the fresh candidate",
    "translate":    "CORRECT  currency translation: same physical phase, new doppler",
    "probe":        "REPLACE  noise probe: whole tuple from the prediction, ref_hop 0",
    "coast_retag":  "CORRECT  coast: forecast doppler, cp re-expressed at the same instant",
    "dr_birth":     "REPLACE  dead-reckon birth / re-pin: whole tuple from BRDC + clk",
    "dr_slew":      "CORRECT  dead-reckon slew: held phase + bounded step toward model",
    "la_rate":      "REPLACE  pooled (l-a) / joint clock code rate, every non-held PRN",
    "reseed":       "CORRECT  far-regime spec_tau re-seed, bounded fractional step",
    "phase_xport":  "REPLACE  #45 step 6: at-ref phase shipped beside the DR tuple",
}

import sys


class Seed(dict):
    """One (chain, PRN) seed: the dict the trackers are POSTed, plus who wrote it.

    dict-subclass on purpose: Phase 1's bar is a byte-identical POST stream, and the
    payload path splats and serializes seeds as plain mappings. Provenance rides in
    __slots__ so no serialization ever sees it; a `dict(seed)` copy simply drops it,
    which is correct -- a copy is a snapshot, not a writer.
    """

    __slots__ = ("prov",)

    def __init__(self, *a, **kw):
        dict.__init__(self, *a, **kw)
        # field -> (owner, epoch_ref_hop_or_None). Construction through plain dict
        # syntax (unmigrated sites, dict(seed) round-trips) starts unattributed.
        self.prov = {}

    @classmethod
    def born(cls, owner, epoch=None, **fields):
        """A wholesale construction with one declared owner (the REPLACE writers).

        kwargs order is the dict's insertion order (py3.7+), so a migrated literal
        keeps the original wire byte order -- state the fields in the same order the
        dict literal had.
        """
        s = cls(fields)
        for k in fields:
            s.prov[k] = (owner, epoch)
        return s

    def put(self, owner, epoch=None, **fields):
        """Attributed field writes: same values, same order, plus who and at-what-epoch."""
        for k, v in fields.items():
            dict.__setitem__(self, k, v)
            self.prov[k] = (owner, epoch)
        return self

    def __setitem__(self, k, v):
        # Unmigrated writer: record the call site so it audits as "?file:line" -- the
        # migration's own to-do list, and a tripwire for any NEW writer added without
        # declaring itself in OWNERS.
        f = sys._getframe(1)
        self.prov[k] = ("?%s:%d" % (f.f_code.co_filename.rsplit("/", 1)[-1], f.f_lineno),
                        None)
        dict.__setitem__(self, k, v)

    def pop(self, k, *default):
        self.prov.pop(k, None)
        return dict.pop(self, k, *default)

    # -- instruments (log-side only; nothing here may touch the POST payload) ----------

    def owners(self):
        """Compact attribution trail for SEEDAUDIT: 'cp0=cp_fit dop=dop_select ...'."""
        short = {"doppler_hz": "dop", "code_phase_chips": "cp0", "code_phase_rate": "rate",
                 "ref_hop": "ref", "doppler_rate_hz_s": "drate",
                 "code_phase_at_ref_chips": "aref"}
        return " ".join("%s=%s" % (short.get(k, k), self.prov[k][0])
                        for k in self if k in self.prov)

    def epoch_skew(self):
        """Fields whose recorded epoch disagrees with the tuple's shipped ref_hop.

        This is #80 stated as a measurement: hold-freeze restores (cp0, rate, ref_hop,
        dop) from the PREVIOUS tuple but code_phase_at_ref_chips keeps the DETECTION's
        epoch, so the tracker -- which prefers the at-ref phase -- reads a phase from
        one instant against an anchor from another. Returns {field: (owner, epoch)}
        for every EPOCHED field whose recorded epoch differs from the shipped ref_hop;
        empty means consistent (or not yet attributed, which the owners() trail shows).

        Only the at-epoch-valid fields are judged. code_phase_chips is a sample-0
        ARGUMENT and code_phase_rate a residual slope: their pairing with ref_hop is
        structural, not an epoch claim, and a writer like `translate` deliberately
        re-expresses them so a mixed-looking tuple is valid by construction. Judging
        them here would flag every translated seed -- a gate that cannot pass.
        """
        EPOCHED = ("doppler_hz", "doppler_rate_hz_s", "code_phase_at_ref_chips")
        ref = self.get("ref_hop")
        if ref is None:
            return {}
        out = {}
        for k in EPOCHED:
            if k not in self or k not in self.prov:
                continue
            owner, epoch = self.prov[k]
            if epoch is not None and epoch != ref:
                out[k] = (owner, epoch)
        return out
