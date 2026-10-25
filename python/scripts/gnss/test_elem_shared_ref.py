#!/usr/bin/env python3
"""Tests for elem_shared_ref.band_consensus on synthetic polls.

    /home/kvand/gnss/venv-ft/bin/python test_elem_shared_ref.py

An instance whose band has no reference pins to its own first model, so its global phase is
arbitrary while its shape is the fleet's. The consensus must form from such a band, phase-aligned,
and still leave out a model whose shape it does not describe. On a band that is already pinned the
reference keeps the pinned instances' phase, and an instance off that phase is listed, not dropped.

A posted reference carries the zeros the instance's generated config holds, so a re-post does
not put an input the generator took out back at full weight.
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elem_shared_ref as esr  # noqa: E402

N_HALF = 16  # elements per pol
NODES = ("cx19", "cx27", "cx42", "cx43", "cx44", "cx51")


def true_shape(rng):
    """One pol's element model: amplitudes 0.5-1.5, random phases, unit norm."""
    v = rng.uniform(0.5, 1.5, N_HALF) * np.exp(1j * rng.uniform(-np.pi, np.pi, N_HALF))
    return v / np.linalg.norm(v)


def model(rng, shape, phase_deg, noise=0.03):
    """An instance's half: the shape at a global phase, plus per-element noise."""
    g = shape * np.exp(1j * np.radians(phase_deg))
    return g + noise * (rng.standard_normal(N_HALF) + 1j * rng.standard_normal(N_HALF))


def noise_model(rng):
    """A model that learned nothing: unrelated to the fleet's shape."""
    return rng.standard_normal(N_HALF) + 1j * rng.standard_normal(N_HALF)


def poll(halves, band="l3oc"):
    """A poll result: one warm, unfrozen assembler per (pol-0 half, pol-1 half)."""
    res = {}
    for i, (h0, h1) in enumerate(halves):
        stage = "gnss%d_%s_n2assemble" % (i % 2, band)
        g = np.concatenate([h0, h1])
        res[(NODES[i // 2], stage)] = {
            "shared_warm": True,
            "shared_frozen": False,
            "g_shared": [[float(x.real), float(x.imag)] for x in g],
        }
    return res


def ref_halves(entry):
    F = np.array([complex(a, b) for a, b in entry["ref"]])
    return F[:N_HALF], F[N_HALF:]


def offset_deg(ref, g):
    return float(np.degrees(np.angle(np.vdot(ref, g / np.linalg.norm(g)))))


class TestBandConsensus(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1017)
        self.shape = (true_shape(self.rng), true_shape(self.rng))

    def consensus(self, halves):
        return esr.band_consensus(poll(halves), "l3oc", esr.EXCLUDE_DEG, esr.MIN_SIM)

    def test_unpinned_band_forms_a_consensus(self):
        # ten instances spread over the circle in global phase, as an unpinned band holds them
        p0 = (0, 40, 75, 110, 150, 200, 240, 280, 310, 345)
        p1 = (-170, -120, -60, -20, 10, 55, 95, 130, 165, 175)
        halves = [
            (model(self.rng, self.shape[0], a), model(self.rng, self.shape[1], b))
            for a, b in zip(p0, p1)
        ]
        e = self.consensus(halves)
        self.assertEqual(e["n"], [10, 10])
        self.assertEqual(e["excluded"], [{}, {}])
        for pol, f in enumerate(ref_halves(e)):
            self.assertAlmostEqual(float(np.linalg.norm(f)), 1.0, places=5)
            self.assertGreater(abs(np.vdot(self.shape[pol], f)), 0.99)
            self.assertGreater(e["sim_median"][pol], 0.98)
        # the instances are where they were: most of them are listed as off the reference
        self.assertGreaterEqual(min(len(x) for x in e["off"]), 6)
        self.assertLess(max(e["R"]), 0.5)

    def test_a_model_of_another_shape_is_left_out(self):
        halves = [
            (model(self.rng, self.shape[0], a), model(self.rng, self.shape[1], -a))
            for a in (0, 60, 130, 200, 260, 320)
        ]
        halves += [(noise_model(self.rng), noise_model(self.rng)) for _ in range(2)]
        e = self.consensus(halves)
        self.assertEqual(e["n"], [6, 6])
        for pol in (0, 1):
            self.assertEqual(sorted(e["excluded"][pol]), ["cx43/0", "cx43/1"])
            self.assertTrue(
                all(s < esr.MIN_SIM for _, s in e["excluded"][pol].values()),
                e["excluded"][pol],
            )
            self.assertGreater(abs(np.vdot(self.shape[pol], ref_halves(e)[pol])), 0.99)

    def test_pinned_band_keeps_its_phase(self):
        # eight instances pinned within a few degrees of 0, two with the fleet's shape far off
        p = (-4, -2, -1, 0, 1, 2, 3, 4, 120, 100)
        halves = [
            (model(self.rng, self.shape[0], a), model(self.rng, self.shape[1], a))
            for a in p
        ]
        e = self.consensus(halves)
        self.assertEqual(e["n"], [10, 10])
        self.assertEqual(e["excluded"], [{}, {}])
        for pol, f in enumerate(ref_halves(e)):
            pinned = [offset_deg(f, h[pol]) for h in halves[:8]]
            self.assertLess(abs(np.mean(pinned)), 1.0, pinned)
            self.assertLess(abs(offset_deg(f, self.shape[pol])), 2.0)
            self.assertEqual(sorted(e["off"][pol]), ["cx44/0", "cx44/1"])
            self.assertAlmostEqual(e["off"][pol]["cx44/0"][0], 120.0, delta=6.0)
            self.assertAlmostEqual(e["off"][pol]["cx44/1"][0], 100.0, delta=6.0)

    def test_fewer_than_four_shapes_is_no_consensus(self):
        # three models of the fleet's shape, pinned together, and four with none: in a
        # consensus that includes them the four score above MIN_SIM
        halves = [
            (model(self.rng, self.shape[0], 0), model(self.rng, self.shape[1], 0))
            for _ in range(3)
        ]
        halves += [(noise_model(self.rng), noise_model(self.rng)) for _ in range(4)]
        with self.assertRaises(ValueError):
            self.consensus(halves)


    def test_two_groups_take_the_phase_of_the_larger(self):
        # The 2026-10-20 b2b pol-1 split, eleven instances of one shape: six near -34 deg,
        # five near +38 deg (one at +61). Every instance keeps (same shape); the reference
        # takes the larger group's phase; the minority is listed as off, not excluded.
        p1 = (-29, -31, -30, -35, -37, -39, 37, 35, 38, 40, 61)
        halves = [
            (model(self.rng, self.shape[0], 0.0), model(self.rng, self.shape[1], p)) for p in p1
        ]
        e = self.consensus(halves)
        self.assertEqual(e["n"], [11, 11])
        self.assertEqual(e["excluded"], [{}, {}])
        self.assertEqual(e["off"][0], {})
        self.assertEqual(len(e["off"][1]), 5)
        for o, _s in e["off"][1].values():
            self.assertGreater(abs(o), esr.EXCLUDE_DEG)
        # the reference's pol-1 phase is the majority group's mean: a model at that phase
        # reads an offset of ~0
        _r0, r1 = ref_halves(e)
        maj = float(np.degrees(np.angle(np.mean([np.exp(1j * np.radians(p)) for p in p1[:6]]))))
        self.assertLess(abs(offset_deg(r1, self.shape[1] * np.exp(1j * np.radians(maj)))), 2.0)


class FakeNode:
    """urllib.request.urlopen for set_elem_sum_shared_ref: records (url, body) per POST."""

    def __init__(self):
        self.posts = []

    def __call__(self, req, timeout=None):
        self.posts.append((req.full_url, json.loads(req.data)))

        class _R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _R(json.dumps({"mode": "live", "staged": {}}).encode())


class TestPostZeros(unittest.TestCase):
    """Every POST zeroes the entries the instance's generated config holds at 0."""

    REF = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]

    def setUp(self):
        self.d = tempfile.mkdtemp()
        flat = [0.1, 0.2, 0.0, 0.0, 0.5, 0.6, 0.0, 0.0]
        with open(os.path.join(self.d, "chord_gnss_cx51_multi.yaml"), "w") as fh:
            fh.write(
                "gnss0_n2assemble:\n  elem_positions_epoch: E\n  elem_sum_shared_ref:\n"
                + "".join("  - %s\n" % x for x in flat)
                + "  elem_sum_shared_ref_mode: live\n"
                "gnss1_e5a_n2assemble:\n  elem_positions_epoch: E\n  elem_sum: true\n"
            )
        self.snap = os.path.join(self.d, "ref.json")
        with open(self.snap, "w") as fh:
            bands = {"l5": {"ref": self.REF}, "e5a": {"ref": self.REF}}
            json.dump({"epoch": "E", "bands": bands}, fh)
        self.real = (esr.GEN, esr.urllib.request.urlopen)
        esr.GEN = os.path.join(self.d, "chord_gnss_%s_multi.yaml")
        self.node = esr.urllib.request.urlopen = FakeNode()

    def tearDown(self):
        esr.GEN, esr.urllib.request.urlopen = self.real
        shutil.rmtree(self.d)

    def test_config_zeros(self):
        self.assertEqual(esr.config_zeros("cx51", "gnss0_n2assemble"), {1, 3})
        self.assertEqual(esr.config_zeros("cx51", "gnss1_e5a_n2assemble"), set())

    def test_post_applies_the_configs_zeros(self):
        with contextlib.redirect_stdout(io.StringIO()):
            esr.post(argparse.Namespace(file=self.snap, mode="live"))
        got = {url.split("/")[3]: body for url, body in self.node.posts}
        self.assertEqual(
            got["gnss0_n2assemble"]["ref"],
            [[0.1, 0.2], [0.0, 0.0], [0.5, 0.6], [0.0, 0.0]],
        )
        self.assertEqual(got["gnss0_n2assemble"]["mode"], "live")
        # a config without a reference has no zeros to apply
        self.assertEqual(got["gnss1_e5a_n2assemble"]["ref"], self.REF)

    def test_watch_posts_apply_them_too(self):
        ok, _ = esr.post_note("cx51", "gnss0_n2assemble", self.REF, "live", "test")
        self.assertTrue(ok)
        self.assertEqual(self.node.posts[0][1]["ref"][1], [0.0, 0.0])


class TestRelabel(unittest.TestCase):
    REF = [[0.6, 0.0], [0.0, 0.8], [1.0, 0.0], [0.0, 0.0]]

    def relabel(self, swap=""):
        d = tempfile.mkdtemp()
        try:
            inp, out = os.path.join(d, "in.json"), os.path.join(d, "out.json")
            with open(inp, "w") as fh:
                json.dump({"epoch": "OLD", "bands": {"l5": {"ref": self.REF}}}, fh)
            a = argparse.Namespace(inp=inp, out=out, epoch="NEW", swap=swap)
            with contextlib.redirect_stdout(io.StringIO()):
                esr.relabel(a)
            with open(out) as fh:
                return json.load(fh)
        finally:
            shutil.rmtree(d)

    def test_relabel_keeps_the_vector(self):
        r = self.relabel()
        self.assertEqual(r["epoch"], "NEW")
        self.assertEqual(r["epoch_relabelled_from"], "OLD")
        self.assertEqual(r["bands"]["l5"]["ref"], self.REF)
        self.assertNotIn("relabel_swap", r)

    def test_swap_across_the_pols_renormalises_each_half(self):
        r = self.relabel("1:2")
        F = np.array([complex(a, b) for a, b in r["bands"]["l5"]["ref"]])
        # pol 0 now (0.6, 1.0), pol 1 (0.8j, 0): each half back to unit norm
        pol0 = np.array([0.6, 1.0]) / np.hypot(0.6, 1.0)
        np.testing.assert_allclose(F[:2], pol0, 1e-5)
        np.testing.assert_allclose(F[2:], [1j, 0.0], atol=1e-6)
        self.assertEqual(r["relabel_swap"], "1:2")
        self.assertEqual(r["epoch_relabelled_from"], "OLD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
