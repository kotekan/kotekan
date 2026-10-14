"""Named GNSS signals for the broker's --signal.

A chain's code constants (constellation, carrier, chip rate, code length, long-code segments
and epoch, overlay length) all fail silently when mistyped: with a wrong overlay length the
seed lands in an effectively random one of the overlay periods. --signal sets them from one
name.

The source of truth is lib/stages/gnss/gnssSignal.hpp, whose `SignalDescriptor`s build the
trackers' replica bank. The header is parsed rather than retyped here, so the two cannot
drift apart, and a name the header does not define is an error at startup.

A chain is a pair. The search acquires the primary code; the tracker despreads the long code
with the secondary built in:

    gps_l5   GPS_L5_Q    10230 chips / 1 ms      GPS_L5_Q_NH    204600 / 20 ms
    gal_e5a  GAL_E5A_Q   10230 chips / 1 ms      GAL_E5A_Q_CS  1023000 / 100 ms
    gps_l2c  GPS_L2C_CM  10230 chips / 20 ms     GPS_L2C_CL     767250 / 1.5 s

so `long_code_segments` = long.code_length / primary.code_length and `long_code_epoch_s` =
long.code_period_s. Where no long descriptor exists (B1C, E1C) the same two numbers come
from the primary's own `secondary_length`, which must agree with the pair wherever both are
defined (`_derive` enforces it; selftest.py cross-checks the derived values).

Receiver-scope constants are not set here: --hops-per-sec (the F-engine's record geometry,
the same for every signal on the instrument) and code_doppler_sign (a decode convention).
"""
import os
import re

# python/scripts/gnss/gnss_broker/ -> four levels to the repo root
_HPP = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "lib",
        "stages",
        "gnss",
        "gnssSignal.hpp",
    )
)

# name prefix -> the single-letter constellation the broker and RINEX both use
_SYS = {"GPS": "G", "GAL": "E", "BDS": "C", "GLO": "R"}

_DESC_RX = re.compile(
    r"inline\s+constexpr\s+SignalDescriptor\s+(\w+)\s*=\s*\{\s*"
    r'"[^"]+"\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)\s*,'
)
_SECONDARY_RX = re.compile(r"/\*secondary_length=\*/\s*(\d+)")


def _load_descriptors(path=None):
    """{name: (carrier_hz, chip_rate_hz, code_length, code_period_s, secondary_length)}."""
    src = open(path or _HPP).read()
    out = {}
    for m in _DESC_RX.finditer(src):
        tail = src[m.end() : m.end() + 400]
        sec = _SECONDARY_RX.search(tail)
        out[m.group(1)] = (
            float(m.group(2)),
            float(m.group(3)),
            int(m.group(4)),
            float(m.group(5)),
            int(sec.group(1)) if sec else 0,
        )
    if not out:
        raise RuntimeError(
            "no SignalDescriptors parsed from %s -- the header's shape "
            "changed and this parser must be updated. Failing at the door "
            "rather than silently serving an empty registry." % (path or _HPP)
        )
    return out


# chain key -> (primary descriptor, long/tracker descriptor or None, min_prn or None)
#
# min_prn 19 marks the BDS-3-only signals (B1C, B2a, B2b). A BDS-2 satellite carries none of
# them yet passes every visibility, ephemeris and health gate, so only capability excludes it.
# A min_prn also implies --signal-capability: the IGS registry's block types decide
# (prnmap.signal_incapable_prns), because some PRNs below 19 (C11-C14) are BDS-3 satellites.
# The PRN floor is the fallback when the registry is unreadable.
_CHAINS = {
    "gps_l1ca": ("GPS_L1CA", None, None),
    "gps_l1c": ("GPS_L1C_P", None, None),
    "gps_l2c": ("GPS_L2C_CM", "GPS_L2C_CL", None),
    "gps_l5": ("GPS_L5_Q", "GPS_L5_Q_NH", None),
    "gal_e1c": ("GAL_E1C", None, None),
    "gal_e5a": ("GAL_E5A_Q", "GAL_E5A_Q_CS", None),
    "gal_e5b": ("GAL_E5B_Q", "GAL_E5B_Q_CS", None),
    "gal_e6": ("GAL_E6_C", "GAL_E6_C_CS", None),
    "bds_b1c": ("BDS_B1C_P", None, 19),
    "bds_b2a": ("BDS_B2A_P", "BDS_B2A_P_CS", 19),
    "bds_b2b": ("BDS_B2B_I", None, 19),
    # B3I is broadcast by BDS-2 and BDS-3 alike, so no min_prn and no capability gate; the
    # broker's BeiDou fallback floor (PRN >= 19) still applies.
    "bds_b3i": ("BDS_B3I", "BDS_B3I_NH", None),
    "glo_l3oc": ("GLO_L3OC_P", None, None),
}


# RF band names, keyed by carrier in MHz rounded to 1 kHz. This is the column a viewer groups
# by, not the signal's own name: Galileo E5a and GPS L5 are both the L5 band (they share
# 1176.45 MHz, and so a receiver), while E5a-Q and L5-Q are different signals within it.
# Module-level so carriers_for_chains can name a carrier without a SignalDef; band_of_freq_id
# takes the (carrier, name) pairs it returns.
RF_BAND_BY_CARRIER = {
    1575.42: "L1",  # GPS L1 C/A + L1C, Galileo E1, BeiDou B1C
    1227.60: "L2",  # GPS L2C
    1176.45: "L5",  # GPS L5, Galileo E5a, BeiDou B2a
    1207.14: "E5b",  # Galileo E5b, BeiDou B2b/B2I
    1278.75: "E6",  # Galileo E6
    1268.52: "B3",  # BeiDou B3I
    1202.025: "L3",  # GLONASS L3OC
    1561.098: "B1I",  # BeiDou B1I (BDS-2, outside CHORD's band)
    1246.00: "G2",  # GLONASS L2OF
    1248.06: "G2",  # GLONASS L2OC
}

# CHORD's F-engine channel pitch (1600 MHz / 8192). freq_id is a physical bin index, so a
# channel's centre is exactly freq_id * this (channel_width_hz in config/chord_gnss_node.yaml,
# and the same constant in GnssGpuRecordAssemble.cpp).
FREQ_ID_MHZ = 0.1953125

# How far from a carrier a channel may sit and still be named for that band. The widest
# signal flown is BPSK(10), a 10.23 MHz mainlobe half-width, so 12 MHz admits the whole
# mainlobe and a little shoulder and rejects a channel that belongs to no tracked band.
_BAND_CLAIM_MHZ = 12.0


def carriers_for_chains(chain_names):
    """Sorted [(carrier MHz, band name)] for the given chain names: the declared band set.

    Carriers come from gnssSignal.hpp via the descriptor table; the config only declares
    which chains the receiver flies. Unknown names are skipped rather than fatal: this feeds
    a health panel, and a typo should cost a label, not a chain.
    """
    out = []
    for name in chain_names or []:
        key = str(name).strip().lower()
        if not key:
            continue
        try:
            sig = get(key)
        except Exception:
            continue
        mhz = round(sig.carrier_hz / 1e6, 3)
        if mhz not in [c for c, _ in out]:
            out.append((mhz, RF_BAND_BY_CARRIER.get(mhz, "%.2fMHz" % mhz)))
    return sorted(out)


def band_of_freq_id(freq_id, carriers):
    """(band name, carrier MHz) for the RF band a CHORD channel sits in; (None, None) if none.

    `carriers` must be the declared set (carriers_for_chains of the broker's --rf-bands), not
    the full RF_BAND_BY_CARRIER table. With the full table, E5b's lower shoulder (about
    1199-1202 MHz) sits nearer GLONASS L3OC's 1202.025 MHz carrier than E5b's 1207.14 and is
    named L3, a band CHORD does not fly.

    Among the candidates the nearest carrier wins, not containment: bds_b3i's BPSK(10)
    mainlobe (1268.52 MHz) reaches gal_e6's carrier at 1278.75, so the occupied bands overlap.
    Nearest splits at the midpoint and gives the same answer on every instance. A channel
    more than _BAND_CLAIM_MHZ from every candidate is unnamed.

    `freq_id` is the absolute F-engine bin index, never a local comb index: nodes hold
    different channel subsets, so a local index yields a wrong band.
    """
    if freq_id is None:
        return (None, None)
    try:
        mhz = float(freq_id) * FREQ_ID_MHZ
    except (TypeError, ValueError):
        return (None, None)
    best, best_d = None, None
    for carrier, name in carriers:
        d = abs(mhz - float(carrier))
        if best_d is None or d < best_d:
            best, best_d = (name, float(carrier)), d
    if best is None or best_d > _BAND_CLAIM_MHZ:
        return (None, None)
    return best


class SignalDef(object):
    """One named chain: constellation x band x code, frozen at construction."""

    __slots__ = (
        "key",
        "primary",
        "long",
        "constellation",
        "carrier_hz",
        "chip_rate_hz",
        "code_length",
        "code_period_s",
        "long_code_segments",
        "long_code_epoch_s",
        "nh_overlay_len",
        "min_prn",
    )

    def __init__(self, key, descs=None):
        descs = descs if descs is not None else _load_descriptors()
        if key not in _CHAINS:
            raise KeyError(
                "unknown signal %r; known: %s" % (key, ", ".join(sorted(_CHAINS)))
            )
        prim, lng, min_prn = _CHAINS[key]
        for n in (prim, lng):
            if n is not None and n not in descs:
                raise KeyError(
                    "signal %s names descriptor %s, which is not in "
                    "gnssSignal.hpp" % (key, n)
                )
        c_hz, chip, clen, per, sec = descs[prim]
        self.key, self.primary, self.long = key, prim, lng
        self.constellation = _SYS[prim.split("_")[0]]
        self.carrier_hz, self.chip_rate_hz = c_hz, chip
        self.code_length, self.code_period_s = clen, per
        self.min_prn = min_prn
        self.long_code_segments, self.long_code_epoch_s = _derive(descs, prim, lng)
        self.nh_overlay_len = sec or self.long_code_segments

    # -- derived, so no caller recomputes them differently -------------------------
    # See RF_BAND_BY_CARRIER.
    _RF_BAND = RF_BAND_BY_CARRIER

    @property
    def rf_band(self):
        """The front-end band this signal sits in -- L1, L2, L5, ... Falls back to the
        carrier in MHz so an unmapped signal is still grouped consistently rather than
        silently merged with something else."""
        mhz = round(self.carrier_hz / 1e6, 3)
        return self._RF_BAND.get(mhz, "%.2fMHz" % mhz)

    # Human names, derived from the descriptor name rather than typed a second time, so a new
    # descriptor is named correctly as soon as it is added.
    _SYS_NAME = {"GPS": "GPS", "GAL": "Galileo", "BDS": "BeiDou", "GLO": "GLONASS"}

    @property
    def short(self):
        """Column-header form: L5-Q, E5a-Q, B2a-P, L2C-CM, L1CA, L3OC-P.

        Only an A/B band suffix after a letter and a digit is lowercased (E5A -> E5a,
        B2B -> B2b); every other part keeps the descriptor's case, so L2C, B1C, L1CA and L3OC
        stay as conventionally written."""
        out = []
        for p in self.primary.split("_")[1:]:
            m = re.match(r"^([A-Z]\d)([AB])$", p)
            out.append(m.group(1) + m.group(2).lower() if m else p)
        return "-".join(out)

    @property
    def label(self):
        """Table form: 'GPS L5-Q', 'Galileo E5a-Q'."""
        return "%s %s" % (
            self._SYS_NAME.get(self.primary.split("_")[0], "?"),
            self.short,
        )

    def __repr__(self):
        return "SignalDef(%s: %s%s, %.3f MHz, %d chips @ %.4f Mcps, %d x %.4g s)" % (
            self.key,
            self.primary,
            "/" + self.long if self.long else "",
            self.carrier_hz / 1e6,
            self.code_length,
            self.chip_rate_hz / 1e6,
            self.long_code_segments,
            self.long_code_epoch_s,
        )


def _derive(descs, prim, lng):
    """(long_code_segments, long_code_epoch_s), from the pair when there is one.

    Where both the primary's secondary_length and a long descriptor exist they must agree:
    they state the same fact, so a disagreement means the header is inconsistent and raises
    ValueError rather than picking one."""
    _, _, clen, per, sec = descs[prim]
    from_sec = (sec, per * sec) if sec else None
    from_long = None
    if lng is not None:
        _, _, l_len, l_per, _ = descs[lng]
        if l_len % clen:
            raise ValueError(
                "%s (%d chips) is not a whole number of %s periods (%d)"
                % (lng, l_len, prim, clen)
            )
        from_long = (l_len // clen, l_per)
    if from_sec and from_long and from_sec[0] != from_long[0]:
        raise ValueError(
            "%s: secondary_length %d disagrees with %s/%s = %d"
            % (prim, from_sec[0], lng, prim, from_long[0])
        )
    got = from_long or from_sec
    if got is None:
        # No secondary and no long code: the primary is the whole code (GPS L1 C/A).
        return 1, per
    return got


def get(key):
    return SignalDef(key)


def describe():
    """One line per chain -- what `--signal help` prints."""
    descs = _load_descriptors()
    return "\n".join("  %-10s %s" % (k, SignalDef(k, descs)) for k in sorted(_CHAINS))


if __name__ == "__main__":
    print(describe())
