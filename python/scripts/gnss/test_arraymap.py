#!/usr/bin/env python3
"""Tests for gnss_arraymap: the resolver, the refusals, and the two numberings.

    /home/kvand/gnss/venv/bin/python -m unittest discover -s python/scripts/gnss -p 'test_*.py'

Most of these are about what the module REFUSES. A resolver that answers every question is
worse than none: the failure this exists to prevent is a beam map quietly summing two different
arrays, and that only shows up as a slightly wrong answer nobody can trace.
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_arraymap as am  # noqa: E402


def write_table(epochs, pointings=None):
    """A minimal two-element table, so the tests do not ride on the shipped one."""
    d = {"schema": 1,
         "pointings": pointings or {"pA": {"dec_deg": 40.73, "bore_az_deg": 180.0,
                                           "bore_el_deg": 81.41}},
         "epochs": epochs}
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(d, fh)
    fh.close()
    return am.JsonBackend(fh.name)


def ep(name, frm, to, planes=("A", "B"), pointing="pA", cube_order=(0, 64), verified=True):
    return {"name": name, "valid_from": frm, "valid_to": to, "pointing": pointing,
            "cube_order": list(cube_order), "verified": verified, "provenance": "test",
            "elements": {
                "0": {"dish": 0, "dish_name": "A01", "slot": 0, "plane": planes[0],
                      "plane_confidence": "measured", "enu_m": [0.0, 0.0, 0.0],
                      "position_confidence": "measured"},
                "64": {"dish": 0, "dish_name": "A01", "slot": 1, "plane": planes[1],
                       "plane_confidence": "measured", "enu_m": [0.0, 0.0, 0.0],
                       "position_confidence": "measured"}}}


TWO = [ep("e1", "2026-08-29T00:00:00Z", "2026-09-15T00:00:00Z", ("A", "B")),
       ep("e2", "2026-09-16T00:00:00Z", None, ("B", "A"))]


class TestTime(unittest.TestCase):
    def test_forms_agree(self):
        want = datetime(2026, 9, 8, tzinfo=timezone.utc)
        for t in ("20260908", "2026-09-08T00:00:00Z", "2026-09-08T00:00:00+00:00",
                  want.timestamp(), want):
            self.assertEqual(am.as_utc(t), want, t)

    def test_naive_datetime_is_utc_not_local(self):
        self.assertEqual(am.as_utc(datetime(2026, 9, 8)), datetime(2026, 9, 8, tzinfo=timezone.utc))

    def test_day_span_is_a_whole_day(self):
        t0, t1 = am.day_span("20260908")
        self.assertEqual((t1 - t0).total_seconds(), 86400.0)


class TestResolve(unittest.TestCase):
    def setUp(self):
        self.be = write_table(TWO)

    def test_resolves_inside(self):
        self.assertEqual(am.at("20260908", self.be).name, "e1")
        self.assertEqual(am.at("20260920", self.be).name, "e2")

    def test_open_ended_epoch_covers_the_far_future(self):
        self.assertEqual(am.at("20301231", self.be).name, "e2")

    def test_gap_refuses(self):
        # The day of the recabling: both neighbours exist, the hour does not. Refusing is the
        # answer, and it is the whole point of the hook.
        with self.assertRaises(am.NoEpoch):
            am.at("20260915", self.be)

    def test_before_the_first_epoch_refuses(self):
        with self.assertRaises(am.NoEpoch):
            am.at("20260101", self.be)

    def test_boundary_is_half_open(self):
        self.assertEqual(am.at("2026-09-14T23:59:59Z", self.be).name, "e1")
        with self.assertRaises(am.NoEpoch):
            am.at("2026-09-15T00:00:00Z", self.be)


class TestSpan(unittest.TestCase):
    def setUp(self):
        self.be = write_table(TWO)

    def test_single_epoch_span(self):
        e = am.require_single("20260901", "20260910", self.be)
        self.assertEqual(e.name, "e1")

    def test_straddle_refuses(self):
        # Adjacent epochs, no gap between them: the refusal must be Straddle specifically.
        be = write_table([ep("a", "2026-08-01T00:00:00Z", "2026-09-15T00:00:00Z"),
                          ep("b", "2026-09-15T00:00:00Z", None)])
        with self.assertRaises(am.Straddle):
            am.require_single("20260908", "20260920", be)

    def test_a_span_crossing_both_a_change_and_a_gap_still_refuses(self):
        # Which subclass fires depends on where the gap falls, so callers catch the base.
        with self.assertRaises(am.Refused):
            am.require_single("20260908", "20260920", self.be)

    def test_span_reaching_into_a_gap_refuses(self):
        # 09-14 -> 09-16 touches e1 and then the gap; it must not come back as "just e1".
        with self.assertRaises(am.NoEpoch):
            am.span("20260914", "20260916", self.be)

    def test_span_past_the_last_epoch_is_fine_when_open(self):
        self.assertEqual([e.name for e in am.span("20260920", "20261001", self.be)], ["e2"])

    def test_empty_span_is_a_programming_error(self):
        with self.assertRaises(ValueError):
            am.span("20260910", "20260910", self.be)

    def test_for_day_uses_the_whole_day(self):
        self.assertEqual(am.for_day("20260908", self.be).name, "e1")
        with self.assertRaises(am.NoEpoch):
            am.for_day("20260915", self.be)

    def test_for_day_catches_a_boundary_inside_the_day(self):
        # An epoch that changes at noon makes that day's master two arrays, not one.
        be = write_table([ep("a", "2026-09-01T00:00:00Z", "2026-09-08T12:00:00Z"),
                          ep("b", "2026-09-08T12:00:00Z", None)])
        with self.assertRaises(am.Straddle):
            am.for_day("20260908", be)


class TestNumbering(unittest.TestCase):
    """The two axes. Crossing them is the trap this module exists to remove."""

    def setUp(self):
        self.e = am.epochs()[0]          # the shipped table

    def test_cube_axis_is_not_the_correlator_axis(self):
        self.assertEqual(self.e.element(cube=9).index, 9)
        self.assertEqual(self.e.element(cube=25).index, 73)      # NOT 25
        self.assertEqual(self.e.element(cube=25).dish, 9)
        self.assertEqual(self.e.element(cube=25).slot, 1)

    def test_correlator_index_is_pol64_plus_dish(self):
        for ci, el in self.e.cube_elements():
            self.assertEqual(el.index, el.slot * 64 + el.dish)
            self.assertEqual(ci, el.slot * 16 + el.dish)

    def test_must_give_exactly_one_axis(self):
        with self.assertRaises(ValueError):
            self.e.element()
        with self.assertRaises(ValueError):
            self.e.element(index=9, cube=9)

    def test_dead_element_is_a_KeyError_not_a_guess(self):
        with self.assertRaises(KeyError):
            self.e.element(index=32)       # live ranges are 0-15 and 64-79


class TestKey(unittest.TestCase):
    def test_key_tracks_content_not_just_name(self):
        a = write_table([ep("same", "2026-01-01T00:00:00Z", None, ("A", "B"))]).epochs()[0]
        b = write_table([ep("same", "2026-01-01T00:00:00Z", None, ("B", "A"))]).epochs()[0]
        self.assertEqual(a.name, b.name)
        self.assertNotEqual(a.key(), b.key())

    def test_key_is_stable_across_loads(self):
        be = write_table(TWO)
        self.assertEqual(be.epochs()[0].key(), write_table(TWO).epochs()[0].key())


class TestCompatibility(unittest.TestCase):
    def setUp(self):
        self.be = write_table(TWO)
        self.k1 = self.be.epochs()[0].key()
        self.k2 = self.be.epochs()[1].key()

    def test_same_epoch_passes(self):
        e, notes = am.assert_compatible(
            [{"day": "20260906", "array_epoch_key": self.k1},
             {"day": "20260910", "array_epoch_key": self.k1}], self.be)
        self.assertEqual(e.name, "e1")
        self.assertEqual(notes, [])

    def test_mixed_epochs_refuse(self):
        with self.assertRaises(am.Straddle):
            am.assert_compatible([{"day": "20260906", "array_epoch_key": self.k1},
                                  {"day": "20260920", "array_epoch_key": self.k2}], self.be)

    def test_unstamped_masters_are_placed_by_day_and_reported(self):
        e, notes = am.assert_compatible([{"day": "20260906"}, {"day": "20260910"}], self.be)
        self.assertEqual(e.name, "e1")
        self.assertTrue(any("no array_epoch stamp" in n for n in notes))

    def test_unstamped_masters_from_different_epochs_still_refuse(self):
        with self.assertRaises(am.Straddle):
            am.assert_compatible([{"day": "20260906"}, {"day": "20260920"}], self.be)

    def test_stale_stamp_refuses_rather_than_matching_by_name(self):
        with self.assertRaises(am.Straddle):
            am.assert_compatible([{"day": "20260906", "array_epoch_key": "e1:deadbeef"}], self.be)

    def test_pointing_disagreement_refuses(self):
        with self.assertRaises(am.Straddle):
            am.assert_compatible([{"day": "20260906", "pointing": "p_other"}], self.be)

    def test_unverified_epoch_is_flagged_not_refused(self):
        _, notes = am.assert_compatible(
            [{"day": "20260920"}],
            write_table([ep("e1", "2026-08-29T00:00:00Z", "2026-09-15T00:00:00Z"),
                         ep("e2", "2026-09-16T00:00:00Z", None, verified=False)]))
        self.assertTrue(any("NOT verified" in n for n in notes))

    def test_unplaceable_artifact_refuses(self):
        with self.assertRaises(am.Straddle):
            am.assert_compatible([{"nside": 64}], self.be)


class TestCheck(unittest.TestCase):
    def test_shipped_table_is_valid(self):
        self.assertEqual(am.check(), [])

    def test_overlap_is_caught(self):
        be = write_table([ep("a", "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z"),
                          ep("b", "2026-02-01T00:00:00Z", None)])
        self.assertTrue(any("overlap" in b for b in am.check(be, node_config="/nonexistent")))

    def test_a_dish_with_one_plane_in_both_slots_is_caught(self):
        be = write_table([ep("a", "2026-01-01T00:00:00Z", None, ("A", "A"))])
        self.assertTrue(any("both slots" in b for b in am.check(be, node_config="/nonexistent")))

    def test_cube_order_naming_a_missing_element_is_caught(self):
        be = write_table([ep("a", "2026-01-01T00:00:00Z", None, cube_order=(0, 64, 65))])
        self.assertTrue(any("no record" in b for b in am.check(be, node_config="/nonexistent")))


class TestShippedTable(unittest.TestCase):
    """The facts the table is meant to carry, asserted against the real file."""

    def test_the_polswap_moves_exactly_six_elements(self):
        a, b = am.at("20260908"), am.at("20260916")
        moved = sorted(i for i in a.elements
                       if a.elements[i].plane != b.elements[i].plane)
        self.assertEqual(moved, [9, 10, 14, 73, 74, 78])

    def test_the_moved_dishes_are_A06_A07_B07(self):
        a = am.at("20260908")
        self.assertEqual(sorted({a.elements[i].dish_name for i in (9, 10, 14, 73, 74, 78)}),
                         ["A06", "A07", "B07"])

    def test_element_9_changes_plane_across_the_swap(self):
        self.assertEqual(am.at("20260908").element(index=9).plane, "B")
        self.assertEqual(am.at("20260916").element(index=9).plane, "A")

    def test_unclassified_dishes_carry_no_plane(self):
        a = am.at("20260908")
        for dish in (4, 5, 7, 8, 12, 15):
            for el in a.dish_elements(dish):
                self.assertIsNone(el.plane, "dish %d slot %d" % (dish, el.slot))
                self.assertFalse(el.known)

    def test_every_five_day_master_we_hold_is_one_epoch(self):
        # The maps already built (20260906..10) must not have straddled anything.
        e = am.assert_compatible([{"day": d} for d in
                                  ("20260906", "20260907", "20260908", "20260909", "20260910")])[0]
        self.assertEqual(e.name, "2026-08-29-baseline")

    def test_pointing_is_carried_and_matches_the_memo(self):
        self.assertEqual(am.at("20260908").pointing.boresight(), (180.0, 81.41))


if __name__ == "__main__":
    unittest.main()
