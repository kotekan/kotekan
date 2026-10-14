# The seed object: a (chain, PRN) seed that records, per field, the writer that set it and the
# epoch (ref_hop) the value is valid at.
#
# Many writers touch a seed each cycle, some replacing fields outright; the attribution makes
# their arbitration observable. `Seed` subclasses dict, so every read, `in`, `.get`, `**splat`
# and json path behaves as for a plain dict and the POST payload is the plain dict's
# (scripts/gnss/broker_equiv.py gates the stream). The provenance lives beside the data in
# __slots__, invisible to json.dumps and to `dict(prn=prn, **seed)`.
#
# It feeds two log-side checks:
#   * SEEDAUDIT names the writer that moved each field (owners());
#   * epoch skew, a field valid at one ref_hop shipped beside a tuple valid at another, is
#     counted per cycle (epoch_skew()).


class Seed(dict):
    """One (chain, PRN) seed: the dict the trackers are POSTed, plus who wrote it.

    A dict subclass because the payload path splats and serializes seeds as plain mappings.
    Provenance rides in __slots__ so no serialization sees it; a `dict(seed)` copy drops it,
    which is correct: a copy is a snapshot, not a writer.
    """

    __slots__ = ("prov",)

    def __init__(self, *a, **kw):
        dict.__init__(self, *a, **kw)
        # field -> (owner, epoch ref_hop or None). Seed(mapping) starts unattributed.
        self.prov = {}

    @classmethod
    def born(cls, owner, epoch=None, **fields):
        """A wholesale construction with one declared owner (the writers that replace a seed).

        kwargs order becomes the dict's insertion order, which is the field order on the
        wire.
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

    def pop(self, k, *default):
        self.prov.pop(k, None)
        return dict.pop(self, k, *default)

    # -- instruments (log-side only; nothing here may touch the POST payload) ----------

    def owners(self):
        """Compact attribution trail for SEEDAUDIT: 'cp0=cp_fit dop=dop_select ...'."""
        short = {
            "doppler_hz": "dop",
            "code_phase_chips": "cp0",
            "code_phase_rate": "rate",
            "ref_hop": "ref",
            "doppler_rate_hz_s": "drate",
            "code_phase_at_ref_chips": "aref",
        }
        return " ".join(
            "%s=%s" % (short.get(k, k), self.prov[k][0]) for k in self if k in self.prov
        )

    def epoch_skew(self):
        """Fields whose recorded epoch disagrees with the tuple's shipped ref_hop.

        A skewed tuple makes the tracker, which prefers the at-ref phase, read a phase from
        one instant against an anchor from another. Returns {field: (owner, epoch)} for every
        EPOCHED field whose recorded epoch differs from the shipped ref_hop; empty means
        consistent (or not yet attributed, which the owners() trail shows).

        Only the at-epoch-valid fields are judged. code_phase_chips is a sample-0 argument
        and code_phase_rate a residual slope: their pairing with ref_hop is structural, not an
        epoch claim, and a writer like `translate` re-expresses them so a mixed-looking tuple
        is valid by construction. Judging them would flag every translated seed.
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
