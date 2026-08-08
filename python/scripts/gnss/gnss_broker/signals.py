"""Named GNSS signals for the broker (task #27 M2).

THE PROBLEM. Today a broker chain is defined by twelve loose scalars on a command line:

    --constellation E --carrier-hz 1176.45e6 --chip-rate-hz 10.23e6 --code-length 10230 \\
    --long-code-segments 100 --long-code-epoch-s 0.1 --nh-overlay-len 100 ...

Every one of them fails SILENTLY when mistyped. broker_up_extra.sh says so in its own
comments -- "getting these wrong does not error: the overlay period is computed mod the
wrong length and the seed lands in an effectively random one of the 100 periods" -- and
that is exactly the 2026-08-08 E5a failure mode, arrived at from the other direction (the
dead-reckon model reducing mod one primary period). A chain should be named, not retyped.

THE SOURCE OF TRUTH IS THE C++ HEADER, NOT THIS FILE. lib/stages/gnss/gnssSignal.hpp holds
28 `SignalDescriptor`s, and it is what the trackers' replica bank is actually built from.
Retyping those numbers here would create a second authority that can drift from the first
without either side erroring -- the same class of defect this module exists to remove. So
the header is PARSED, and a name that is not in it is a hard failure at the door.

THE CHAIN IS A PAIR, and that is the subtlety worth stating. The search acquires the
PRIMARY code; the tracker despreads the LONG one with the secondary baked in:

    gps_l5   GPS_L5_Q    10230 chips / 1 ms      GPS_L5_Q_NH    204600 / 20 ms
    gal_e5a  GAL_E5A_Q   10230 chips / 1 ms      GAL_E5A_Q_CS  1023000 / 100 ms
    gps_l2c  GPS_L2C_CM  10230 chips / 20 ms     GPS_L2C_CL     767250 / 1.5 s

so `long_code_segments` = long.code_length / primary.code_length and `long_code_epoch_s` =
long.code_period_s. Where no long descriptor exists (B1C, E1C) the overlay rides the bank's
single-sequence slot and the same two numbers come from the primary's own
`secondary_length` -- which agrees with the pair wherever both are defined. Checked, not
assumed: see `_derive` and the self-check at the bottom.

WHAT IS NOT HERE, deliberately: `hops_per_sec` (the F-engine's record geometry -- a
RECEIVER property, identical for every signal on the instrument) and `code_doppler_sign`
(a decode convention). Those are receiver-scope and move to the receiver config in M6.
"""
import os
import re

# python/scripts/gnss/gnss_broker/ -> four levels to the repo root
_HPP = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "lib", "stages", "gnss", "gnssSignal.hpp"))

# name prefix -> the single-letter constellation the broker and RINEX both use
_SYS = {"GPS": "G", "GAL": "E", "BDS": "C", "GLO": "R"}

_DESC_RX = re.compile(
    r'inline\s+constexpr\s+SignalDescriptor\s+(\w+)\s*=\s*\{\s*'
    r'"[^"]+"\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)\s*,')
_SECONDARY_RX = re.compile(r'/\*secondary_length=\*/\s*(\d+)')


def _load_descriptors(path=None):
    """{name: (carrier_hz, chip_rate_hz, code_length, code_period_s, secondary_length)}."""
    src = open(path or _HPP).read()
    out = {}
    for m in _DESC_RX.finditer(src):
        tail = src[m.end():m.end() + 400]
        sec = _SECONDARY_RX.search(tail)
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)), int(m.group(4)),
                           float(m.group(5)), int(sec.group(1)) if sec else 0)
    if not out:
        raise RuntimeError("no SignalDescriptors parsed from %s -- the header's shape "
                           "changed and this parser must be updated. Failing at the door "
                           "rather than silently serving an empty registry."
                           % (path or _HPP))
    return out


# chain key -> (primary descriptor, long/tracker descriptor or None, min_prn or None)
#
# min_prn 19 marks the BDS-3-only signals: C1-C18 are BDS-2 birds that broadcast B1I at
# 1561 MHz, 14 MHz outside the science band, so there is nothing to despread -- yet they
# are real, healthy, overhead satellites that every other gate waves through. Only
# capability excludes them (see broker_up_extra.sh).
_CHAINS = {
    "gps_l1ca":  ("GPS_L1CA",    None,             None),
    "gps_l1c":   ("GPS_L1C_P",   None,             None),
    "gps_l2c":   ("GPS_L2C_CM",  "GPS_L2C_CL",     None),
    "gps_l5":    ("GPS_L5_Q",    "GPS_L5_Q_NH",    None),
    "gal_e1c":   ("GAL_E1C",     None,             None),
    "gal_e5a":   ("GAL_E5A_Q",   "GAL_E5A_Q_CS",   None),
    "gal_e5b":   ("GAL_E5B_Q",   None,             None),
    "gal_e6":    ("GAL_E6_C",    None,             None),
    "bds_b1c":   ("BDS_B1C_P",   None,             19),
    "bds_b2a":   ("BDS_B2A_P",   "BDS_B2A_P_CS",   19),
    "bds_b2b":   ("BDS_B2B_I",   None,             19),
    "bds_b3i":   ("BDS_B3I",     None,             None),
    "glo_l3oc":  ("GLO_L3OC_P",  None,             None),
}


class SignalDef(object):
    """One named chain: constellation x band x code, frozen at construction."""

    __slots__ = ("key", "primary", "long", "constellation", "carrier_hz", "chip_rate_hz",
                 "code_length", "code_period_s", "long_code_segments",
                 "long_code_epoch_s", "nh_overlay_len", "min_prn")

    def __init__(self, key, descs=None):
        descs = descs if descs is not None else _load_descriptors()
        if key not in _CHAINS:
            raise KeyError("unknown signal %r; known: %s"
                           % (key, ", ".join(sorted(_CHAINS))))
        prim, lng, min_prn = _CHAINS[key]
        for n in (prim, lng):
            if n is not None and n not in descs:
                raise KeyError("signal %s names descriptor %s, which is not in "
                               "gnssSignal.hpp" % (key, n))
        c_hz, chip, clen, per, sec = descs[prim]
        self.key, self.primary, self.long = key, prim, lng
        self.constellation = _SYS[prim.split("_")[0]]
        self.carrier_hz, self.chip_rate_hz = c_hz, chip
        self.code_length, self.code_period_s = clen, per
        self.min_prn = min_prn
        self.long_code_segments, self.long_code_epoch_s = _derive(descs, prim, lng)
        self.nh_overlay_len = sec or self.long_code_segments

    # -- derived, so no caller recomputes them differently -------------------------
    @property
    def band(self):
        """The CODE-clock scope: cable + PFB group delay are per carrier, so a code bias
        measured on one carrier is not another's. (The CARRIER-side clock is
        receiver-scope -- see docs/CHORD_BROKER_REFACTOR.md 1.4.) Keyed by MHz so L5 and
        E5a, which really are the same 1176.45 MHz hardware, share one key."""
        return "%.2fMHz" % (self.carrier_hz / 1e6)

    # Human names, derived from the descriptor rather than typed a second time. A viewer
    # that draws "gps_l5 GPS_L5_Q" is showing its plumbing; these come from the same source
    # as every other constant, so a new signal is named correctly the day it is added.
    _SYS_NAME = {"GPS": "GPS", "GAL": "Galileo", "BDS": "BeiDou", "GLO": "GLONASS"}

    @property
    def short(self):
        """Column-header form: L5-Q, E5a-Q, B2a-P, L2C-CM, L1CA, L3OC-P.

        ⚠️ ONLY the a/b band suffix lowercases, and only after a digit: E5A->E5a, B2B->B2b.
        A blanket "first letter up, rest down" looks tidier and is wrong -- it produces L2c,
        B1c, L1ca, L3oc, none of which is how anyone writes these signals. The descriptor
        names are already correct; leave them alone except where convention differs."""
        out = []
        for p in self.primary.split("_")[1:]:
            m = re.match(r"^([A-Z]\d)([AB])$", p)
            out.append(m.group(1) + m.group(2).lower() if m else p)
        return "-".join(out)

    @property
    def label(self):
        """Table form: 'GPS L5-Q', 'Galileo E5a-Q'."""
        return "%s %s" % (self._SYS_NAME.get(self.primary.split("_")[0], "?"), self.short)

    @property
    def q_alias_hz(self):
        """Search-Doppler record-alias quantum, 1/(2*t_rec)."""
        return 0.5 * self.chip_rate_hz / self.code_length

    @property
    def long_code_chips(self):
        return self.long_code_segments * self.code_length

    @property
    def cl_seg_s(self):
        """ONE segment of the long code, in seconds -- the auto-search's step size. This
        used to be hard-coded at L2C CL's 0.020 s even after the epoch was parameterised,
        so on any other signal a correction of +-1 moved the anchor by 20 segments."""
        return self.long_code_epoch_s / self.long_code_segments

    def __repr__(self):
        return ("SignalDef(%s: %s%s, %.3f MHz, %d chips @ %.4f Mcps, %d x %.4g s)"
                % (self.key, self.primary, "/" + self.long if self.long else "",
                   self.carrier_hz / 1e6, self.code_length, self.chip_rate_hz / 1e6,
                   self.long_code_segments, self.long_code_epoch_s))


def _derive(descs, prim, lng):
    """(long_code_segments, long_code_epoch_s), from the PAIR when there is one.

    Both routes are computed wherever both exist and required to agree -- the overlay
    length and the long code's own length are two statements of the same fact, and a
    disagreement means the header is inconsistent, which is worth a crash rather than a
    coin flip between them."""
    _, _, clen, per, sec = descs[prim]
    from_sec = (sec, per * sec) if sec else None
    from_long = None
    if lng is not None:
        _, _, l_len, l_per, _ = descs[lng]
        if l_len % clen:
            raise ValueError("%s (%d chips) is not a whole number of %s periods (%d)"
                             % (lng, l_len, prim, clen))
        from_long = (l_len // clen, l_per)
    if from_sec and from_long and from_sec[0] != from_long[0]:
        raise ValueError("%s: secondary_length %d disagrees with %s/%s = %d"
                         % (prim, from_sec[0], lng, prim, from_long[0]))
    got = from_long or from_sec
    if got is None:
        # No secondary and no long code: the primary IS the whole code (GPS L1 C/A).
        return 1, per
    return got


def get(key):
    return SignalDef(key)


def names():
    return sorted(_CHAINS)


def describe():
    """One line per chain -- what `--signal help` prints."""
    descs = _load_descriptors()
    return "\n".join("  %-10s %s" % (k, SignalDef(k, descs)) for k in sorted(_CHAINS))


if __name__ == "__main__":
    print(describe())
