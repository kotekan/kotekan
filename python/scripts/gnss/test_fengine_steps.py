#!/usr/bin/env python3
"""Tests for fengine_steps: the steps between two starts from the table, and fit_steps on
synthetic phases.

    /home/kvand/gnss/venv-ft/bin/python test_fengine_steps.py

A product made under one start is turned by the sum of the steps of every start since, so a
start with no row or no measurement must refuse, not pass for zero. fit_steps must find the steps
from a handful of bands, each with a constant of its own, and give no step where the bands cannot
tell one step from another: across the L1 bands alone, steps 2 samples apart look the same.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fengine_steps as fs  # noqa: E402

TS_NS = 0.3125
# Carriers of bands the trackers hold element models for, Hz.
GNSS_HZ = [1176.45e6, 1207.14e6, 1227.6e6, 1248.06e6, 1268.52e6, 1575.42e6]
L1_HZ = [1561.098e6, 1575.42e6, 1575.42e6]
# The bands whose models had settled half an hour after the 2026-11-26 start, Hz.
SETTLED_1126_HZ = (
    [1176.45e6] * 3
    + [1202.025e6, 1207.14e6, 1207.14e6, 1246.0e6, 1268.52e6, 1278.75e6, 1561.098e6]
    + [1575.42e6] * 3
)


def table():
    """Three starts of eight inputs; the second and the third are measured."""
    t = {"adc_sample_ns": TS_NS, "n_inputs": 8, "rows": []}
    for frame0 in (100, 200, 300):
        fs.add_start(t, frame0, "utc%d" % frame0)
    t["rows"][1].update(steps=[0, 1, None, -2, 0, 0, 4, 0], pol_step=0)
    t["rows"][2].update(steps=[0, 0, 3, 1, 0, -1, 0, 0], pol_step=2)
    return t


class TestStepsBetween(unittest.TestCase):
    def test_steps_add_up_over_the_starts_between(self):
        t = table()
        self.assertEqual(fs.steps_between(t, 100, 200), [0, 1, None, -2, 0, 0, 4, 0])
        self.assertEqual(fs.steps_between(t, 100, 300), [0, 1, None, -1, 0, -1, 4, 0])

    def test_data_older_than_the_product_reverses_the_sign(self):
        self.assertEqual(
            fs.steps_between(table(), 300, 100), [0, -1, None, 1, 0, 1, -4, 0]
        )

    def test_the_same_start_is_no_step(self):
        self.assertEqual(fs.steps_between(table(), 200, 200), [0] * 8)

    def test_pol_step_is_added_to_the_second_half(self):
        self.assertEqual(
            fs.steps_between(table(), 200, 300, pol_step=True),
            [0, 0, 3, 1, 2, 1, 2, 2],
        )

    def test_a_start_that_was_not_measured_is_refused(self):
        t = table()
        t["rows"][1]["steps"] = None
        with self.assertRaises(ValueError) as e:
            fs.steps_between(t, 100, 300)
        self.assertIn("steps of start 200", str(e.exception))
        t = table()
        t["rows"][2]["pol_step"] = None
        fs.steps_between(t, 100, 300)
        with self.assertRaises(ValueError) as e:
            fs.steps_between(t, 100, 300, pol_step=True)
        self.assertIn("pol_step of start 300", str(e.exception))

    def test_a_start_without_a_row_is_refused(self):
        t = table()
        del t["rows"][1]
        with self.assertRaises(ValueError) as e:
            fs.steps_between(t, 100, 300)
        self.assertIn("a start in between has no row", str(e.exception))
        with self.assertRaises(ValueError):
            fs.steps_between(table(), 100, 400)


class TestTable(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.path = os.path.join(self.d, "steps.json")
        fs.save(table(), self.path)

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_a_saved_table_reads_back(self):
        self.assertEqual(fs.load(self.path), table())
        self.assertEqual(os.listdir(self.d), ["steps.json"])

    def test_a_row_that_names_it_half_step_is_read(self):
        t = table()
        for row in t["rows"]:
            row["half_step"] = row.pop("pol_step")
        with open(self.path, "w") as fh:
            json.dump(t, fh)
        self.assertEqual(fs.load(self.path), table())

    def test_a_new_start_follows_the_last_row(self):
        t = table()
        fs.add_start(t, 400, "utc400")
        self.assertEqual(
            t["rows"][-1],
            {
                "frame0_int": 400,
                "start_utc": "utc400",
                "prev_frame0_int": 300,
                "steps": None,
                "pol_step": None,
            },
        )

    def test_set_steps_and_pol_step(self):
        steps = [0, 0, None, 2, 0, 0, 0, -1]
        src = os.path.join(self.d, "measured.json")
        with open(src, "w") as fh:
            json.dump(steps, fh)
        fs.set_steps(
            argparse.Namespace(table=self.path, frame0=200, file=src, source="by hand")
        )
        fs.set_pol_step(argparse.Namespace(table=self.path, frame0=200, n=-1))
        row = fs.load(self.path)["rows"][1]
        self.assertEqual(
            (row["steps"], row["pol_step"], row["source"]), (steps, -1, "by hand")
        )
        with open(src, "w") as fh:
            json.dump(steps[:4], fh)
        with self.assertRaises(SystemExit):
            fs.set_steps(
                argparse.Namespace(table=self.path, frame0=200, file=src, source="x")
            )


def gained(rng, steps, freqs_hz, noise_deg=5.0):
    """The phase each input gains in each band for `steps`: plus a constant per band and noise
    per input."""
    p = np.array([fs.phase(steps, f, TS_NS) for f in freqs_hz])
    p += rng.uniform(-np.pi, np.pi, (len(freqs_hz), 1))
    return p + np.radians(noise_deg) * rng.standard_normal(p.shape)


class TestFitSteps(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1126)
        self.steps = [0] * 32
        self.steps[4:8] = [4] * 4
        self.steps[12:16] = [-3] * 4
        self.steps[20:22] = [8, -8]
        self.steps[30] = -13

    def test_steps_are_recovered(self):
        steps, rms = fs.fit_steps(gained(self.rng, self.steps, GNSS_HZ), GNSS_HZ, TS_NS)
        self.assertEqual(steps, self.steps)
        self.assertLess(rms, 10.0)

    def test_an_input_too_few_bands_read_gets_none(self):
        p = gained(self.rng, self.steps, GNSS_HZ)
        p[:, 5] = np.nan
        p[:4, 9] = np.nan  # two bands are left: fewer than MIN_BANDS
        steps, _ = fs.fit_steps(p, GNSS_HZ, TS_NS)
        self.assertEqual(
            steps, [None if i in (5, 9) else s for i, s in enumerate(self.steps)]
        )

    def test_a_noisy_band_is_left_out(self):
        p = gained(self.rng, self.steps, GNSS_HZ)
        p[3] += np.radians(45.0) * self.rng.standard_normal(32)
        steps, rms = fs.fit_steps(p, GNSS_HZ, TS_NS)
        self.assertEqual(steps, self.steps)
        self.assertLess(rms, 10.0)

    def test_every_band_noisy_still_fits(self):
        """2026-11-26, pol 0: every band's models read 12-34 deg; leaving out each band above
        MAX_BAND_RMS_DEG would leave the L5 and E5b bands alone, which cannot tell the steps."""
        p = gained(self.rng, self.steps, SETTLED_1126_HZ, noise_deg=22.0)
        steps, _ = fs.fit_steps(p, SETTLED_1126_HZ, TS_NS)
        got = [(i, s) for i, s in enumerate(steps) if s is not None]
        self.assertGreaterEqual(len(got), 0.75 * len(steps))
        for i, s in got:
            self.assertEqual(s, self.steps[i])

    def test_one_band_off_does_not_split_the_group(self):
        """An input group must sit together in most bands, not in every one: one band off by
        40-80 deg on random inputs would otherwise leave a handful of inputs as the group."""
        p = gained(self.rng, self.steps, GNSS_HZ)
        p[2] += np.radians(self.rng.uniform(40, 80, 32)) * self.rng.choice([-1, 1], 32)
        self.assertGreaterEqual(fs._group(p).sum(), 16)
        steps, _ = fs.fit_steps(p, GNSS_HZ, TS_NS)
        self.assertEqual(steps, self.steps)

    def test_dark_inputs_are_never_the_group(self):
        p = gained(self.rng, self.steps, GNSS_HZ)
        p[:, 16:30] = np.nan
        self.assertFalse(fs._group(p)[16:30].any())

    def test_the_l1_bands_alone_cannot_tell_the_steps(self):
        with self.assertRaises(ValueError):
            fs.fit_steps(gained(self.rng, self.steps, L1_HZ), L1_HZ, TS_NS)


class TestRanges(unittest.TestCase):
    def test_runs_of_inputs_are_named_together(self):
        self.assertEqual(
            fs.ranges([0, 4, 4, 4, 0, None, -1, 0]), "1-3 +4, 6 -1; unread 5"
        )
        self.assertEqual(fs.ranges([0, 0]), "none moved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
