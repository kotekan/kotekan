#!/usr/bin/env python3
"""Constructed nav bits for satellites too weak to decode themselves.

`LnavPredictor` can only predict a satellite it has SYNCED, which needs 620 contiguous
parity-clean bits -- exactly what a weak satellite cannot provide. Since the 2026-07-25 floor
fix, `nobits` is the only thing still holding the peel off most of the GPS constellation.

This source needs no decode at all: subframes 1-3 ARE the ephemeris, and BRDC holds the
ephemeris for every satellite in the sky. Bit content is built by `gps_lnav_encode` (validated
113,820/113,820 bits against air); this module solves the OTHER half of the problem, which is
WHERE those bits land on the capture clock.

    ALIGNMENT.  Bit edges are 20 ms apart and the sky's propagation spread is ~19 ms
    (67-86 ms), so one satellite's anchor is NOT reusable for another -- borrowing it can be a
    whole bit out. Instead:

        rx_capture(prn, TOW T) = T*6 + range_prn/c - clk_prn + offset

    where `offset` (capture clock -> GPS week seconds) is COMMON to all satellites and is
    calibrated from any satellite the decoder HAS synced. Per-satellite geometry comes from
    BRDC, which is good to nanoseconds; the only unknown is the common offset, and one synced
    satellite pins it. Agreement between several synced satellites is the self-check, and it
    is logged -- a spread approaching a bit period means the calibration is not trustworthy
    and this source should stay quiet.

THE INTERFACE IS DELIBERATELY THE SAME as LnavPredictor.predict(prn, utc_now, horizon_s) ->
{"utc0","bit_s","bits"} | None, so the broker asks sources in priority order and a new
constellation is a new source rather than a rewrite. GAL I/NAV, BDS B-CNAV1 and GPS CNAV all
fit this shape: only the field packing and frame geometry differ, while the alignment solve,
the borrow, the known-mask and select_eph are signal-independent.
"""
import math

import numpy as np

import gps_lnav_encode as L

C_MPS = 299792458.0
SF_BITS = 300
BIT_S = 0.02
SF_S = 6.0
# Refuse to emit if the per-satellite offset estimates disagree by more than this: beyond a
# few ms the subframe/bit indexing is at risk, and wrong-but-confident bits are the one
# failure this whole path must not produce (see gps_lnav_encode.select_eph).
MAX_OFFSET_SPREAD_S = 0.002
# A calibrating satellite further than this from the median is discarded before the spread is
# judged. Half a bit: beyond it the satellite's own sync is suspect, and its offset would be
# indexing a different 20 ms slot anyway.
OUTLIER_S = 0.010
# Never construct bits for a satellite that is not actually UP. Measured live 2026-07-26: the
# broker seeds below-horizon PRNs as noise probes (--noise-probes), and handing them bits
# satisfied peel_require_bits, so the SNR gate stopped rejecting them (NOT-PEELING went to
# `none`) and the peel started "removing" satellites that were not there -- pure noise
# injection. GPS went HURTING 0-1 -> 7-8, i.e. 7-8 sats ADDING power to the band, which is the
# exact opposite of the peel's purpose. Elevations of the offenders: -84, -75, -72, -52 deg.
# 5 deg rather than 0: it matches the project's standard analysis filter, and a sat within a
# few degrees of the horizon has no usable gain to measure anyway.
MASK_DEG = 5.0


class BrdcLnavSource:
    """GPS LNAV bits constructed from broadcast ephemeris, for un-synced satellites."""

    def __init__(self, log=None):
        self._log = log or (lambda m: None)
        self.offset = None        # capture-clock UTC minus GPS seconds-of-week
        self.spread = None        # disagreement between calibrating satellites (s)
        self.n_cal = 0
        self.n_rej = 0            # calibrating sats discarded as outliers (see OUTLIER_S)
        self.n_ambig = 0          # PRNs refused for ambiguous ephemeris (rival toe)
        self.n_masked = 0         # PRNs refused for being below MASK_DEG
        self.borrow = None        # constellation-common words (TLM + sf1 reserved)
        self.week = None
        self._eph = None
        self._geom = None
        self._fleet = None        # FleetPages (Layer 2): cross-sat sf4/5 words, set by update()

    # ---- per-cycle refresh -------------------------------------------------------------
    def update(self, eph, geom, predictor):
        """Recalibrate from the decoder's synced satellites. `geom` is brdc_predict's
        {prn: (dop, rate, el, range_m, sat_clk_s)}."""
        self._eph, self._geom = eph, geom
        self._fleet = getattr(predictor, "fleet", None)
        offs = []
        borrow = None
        for prn, st in getattr(predictor, "_p", {}).items():
            if st.sync is None or st.utc_anchor is None:
                continue
            g = geom.get(prn)
            recs = eph.get(("G", prn))
            if not g or not recs:
                continue
            slot0, tow0, _sfid0 = st.sync
            aslot, autc = st.utc_anchor
            rx_capture = autc + (slot0 - aslot) * st.bit_s
            # st.sync's tow is the HOW field, which carries the count of the NEXT subframe --
            # so the subframe STARTING at slot0 is (tow0 - 1). Getting this wrong slides the
            # whole calibration by one subframe, i.e. 6 s.
            rx_gps_sow = (tow0 - 1) * SF_S + g[3] / C_MPS - g[4]
            offs.append(rx_capture - rx_gps_sow)
            self.week = int(recs[-1]["week"])
            if borrow is None and 1 in st.sf_data:
                w = st.sf_data[1]
                borrow = {"tlm": w[0],
                          "sf1_resv": [w[3], w[4], w[5], np.asarray(w[6])[:16]]}
        if not offs:
            self.offset, self.spread, self.n_cal = None, None, 0
            self.n_rej = 0
            return
        offs.sort()
        med = offs[len(offs) // 2]                  # median: robust to a bad sync
        # ...and the GATE has to be as robust as the estimator. Gating on max-min meant ONE
        # satellite whose sync sat a bit off disabled the whole source: measured live
        # 2026-07-25, spread read 19.1 ms (~one bit period) across 8 satellites and refused to
        # emit, having been 0.31-0.90 ms minutes earlier. Drop the outliers, then judge the
        # survivors -- one satellite should not be able to veto seven that agree.
        keep = [o for o in offs if abs(o - med) <= OUTLIER_S]
        self.n_rej = len(offs) - len(keep)
        if not keep:
            keep = offs
        self.offset = keep[len(keep) // 2]
        self.spread = keep[-1] - keep[0]
        self.n_cal = len(keep)
        if borrow is not None:
            self.borrow = borrow                     # sticky: survives losing every sync

    def ready(self):
        # n_cal >= 2: a single calibrating satellite has NO cross-check -- at startup its own
        # sync can be half-baked and the whole clock offset lands wrong, which shifts every
        # constructed table and scored 64-76% against the air (2026-07-26, the fossilized
        # startup transient that first exposed the absorbing-veto flaw). Two sats must agree.
        return (self.offset is not None and self.borrow is not None
                and self.week is not None and self.n_cal >= 2
                and self.spread <= MAX_OFFSET_SPREAD_S)

    def why_not(self):
        if self.offset is None:
            return "no synced satellite to calibrate the clock offset"
        if self.n_cal < 2:
            return "only one calibrating satellite (no cross-check on the clock offset)"
        if self.borrow is None:
            return "no decoded sf1 to borrow the reserved words from"
        if self.n_cal >= 2 and self.spread > MAX_OFFSET_SPREAD_S:
            return (f"offset spread {self.spread * 1e3:.1f} ms across {self.n_cal} sats"
                    f" ({self.n_rej} outliers already dropped)")
        return ""

    # ---- the supplier contract ---------------------------------------------------------
    def predict(self, prn, utc_now, horizon_s=4.0):
        """{"utc0","bit_s","bits"} for [utc_now, +horizon), or None. Same contract as
        LnavPredictor.predict, so the broker can treat sources uniformly."""
        if not self.ready():
            return None
        g = (self._geom or {}).get(prn)
        recs = (self._eph or {}).get(("G", prn))
        if not g or not recs:
            return None
        if g[2] < MASK_DEG:          # below the mask: not up, nothing to peel (see MASK_DEG)
            self.n_masked += 1
            return None
        # rx_capture(T) = T*6 + range/c - clk + offset  ->  invert for the subframe index
        delay = g[3] / C_MPS - g[4]
        t_sf = (utc_now - self.offset - delay) / SF_S
        sf0 = int(math.floor(t_sf))
        # capture-clock time of that subframe's first bit
        utc0 = sf0 * SF_S + delay + self.offset
        n_sf = int(math.ceil((horizon_s + (utc_now - utc0)) / SF_S)) + 1
        out = []
        for k in range(n_sf):
            T = sf0 + k
            sfid = (T % 5) + 1
            if sfid not in L.ENCODERS:
                # sf4/5: not constructible from BRDC -- but the FLEET may know this
                # supercycle page (Layer 2: unanimity across other satellites' decodes).
                # TLM is borrowed (constellation-common), HOW is computed for this TOW, and
                # only the CHAIN-VALID prefix is emitted: an unknown word breaks the D30
                # chain, so later words are untransmittable even when their data is known.
                fl = self._fleet.lookup(sfid, (T + 1) % 125) if self._fleet else None
                if fl is None:
                    out.extend([0] * SF_BITS)
                    continue
                fw, fk = fl
                # assemble_subframe takes the EIGHT data words (TLM via kwarg, HOW from
                # tow_next); fleet arrays are 10-slot with 0-1 = TLM/HOW placeholders.
                pm10 = self._fleet.prefix_mask([True, True] + fk[2:])
                known8 = [fk[w] and pm10[w] for w in range(2, 10)]
                bits, kn = L.assemble_subframe(T + 1, sfid, fw[2:],
                                               tlm=self.borrow["tlm"], known29=known8)
                sf = [int(1 - 2 * int(bits[i])) if (kn[i] and pm10[i // 30]) else 0
                      for i in range(SF_BITS)]
                self._fleet.n_fill += 1
                self._fleet.n_word += sum(fk[2:])
                out.extend(sf)
                continue
            t_gps = T * SF_S + self.week * 604800.0
            e = L.select_eph(recs, t_gps)
            # AMBIGUOUS EPHEMERIS -> refuse (see gps_lnav_encode.eph_ambiguous). A rival record
            # at effectively the same toe means we cannot know which data set is on air, and a
            # satellite needing constructed bits is one we cannot decode to find out. Wrong-
            # record bits are CONFIDENTLY wrong and would be subtracted at the wrong sign on
            # ~40% of records; no bits merely means no subtraction.
            if L.eph_ambiguous(recs, e):
                self.n_ambig += 1
                return None
            words = L.ENCODERS[sfid](e)
            known = [True] * 8
            if sfid == 1:
                r = self.borrow["sf1_resv"]
                words[1], words[2], words[3] = r[0], r[1], r[2]
                words[4] = L._w(list(r[3][:16]), list(np.asarray(words[4])[16:]))
            if sfid == 2:
                known[7] = False                     # fit + AODO are not in RINEX
            bits, kn = L.assemble_subframe(T + 1, sfid, words,
                                           tlm=self.borrow["tlm"], known29=known)
            out.extend(int(np.where(kn[i], 1 - 2 * int(bits[i]), 0)) for i in range(SF_BITS))
        # trim to the requested window
        i0 = int(round((utc_now - utc0) / BIT_S))
        i0 = max(0, min(i0, len(out)))
        out = out[i0:i0 + int(math.ceil(horizon_s / BIT_S)) + 2]
        if not any(out):
            return None
        return {"utc0": utc0 + i0 * BIT_S, "bit_s": BIT_S, "bits": out}

    # ---- health ------------------------------------------------------------------------
    def verify(self, prn, predictor):
        """Score constructed bits against a SYNCED satellite's own received bits. This is the
        live version of the offline 113,820/113,820 test and the only thing that can catch a
        bad ephemeris selection or a drifting offset, both of which produce confidently WRONG
        bits. Returns (n_compared, n_agree) or None.

        ⚠️ READ IT ON STRONG SATELLITES. The comparison is against RECEIVED bits, errors and
        all, so a weak satellite scores low because the AIR is wrong, not the construction --
        measured hold-one-out 2026-07-25: G05/G15/G18/G20 100.00%, G24 65.6%, G29 80.3%, and
        the same G24 subframes score 100% when restricted to those that passed parity. A drop
        on a STRONG satellite is the alarm; a low score on a weak one is the point of this
        module."""
        st = getattr(predictor, "_p", {}).get(prn)
        if st is None or st.sync is None or st.utc_anchor is None or not st.hist:
            return None
        aslot, autc = st.utc_anchor
        slots = sorted(st.hist)
        t0 = autc + (slots[0] - aslot) * st.bit_s
        span = (slots[-1] - slots[0]) * st.bit_s
        nb = self.predict(prn, t0, horizon_s=min(span, 60.0))
        if nb is None:
            return None
        # Score in SUBFRAME-SIZED BLOCKS, each at its better polarity. A single global polarity
        # is the wrong instrument here: st.hist carries the predictor's CHAINED polarity, which
        # can flip mid-history (that is why _try_sync has a suffix-flip repair search), so one
        # flip inside the window drags a perfect construction down to 50-67% -- measured live
        # 2026-07-25 at 53-67% on satellites that score 100% offline. Blocks confine a flip to
        # the block it happens in. Global polarity remains free either way: the peel de-bits
        # and re-applies with the same signs, so only self-consistency matters.
        n = ag = 0
        blk_n = blk_ag = 0
        for j, b in enumerate(nb["bits"]):
            if j and j % SF_BITS == 0:
                n += blk_n; ag += max(blk_ag, blk_n - blk_ag)
                blk_n = blk_ag = 0
            if b == 0:
                continue
            slot = aslot + int(round((nb["utc0"] + j * nb["bit_s"] - autc) / st.bit_s))
            v = st.hist.get(slot)
            if v is None:
                continue
            blk_n += 1
            blk_ag += int(v == b)
        n += blk_n; ag += max(blk_ag, blk_n - blk_ag)
        return (n, ag) if n else None
