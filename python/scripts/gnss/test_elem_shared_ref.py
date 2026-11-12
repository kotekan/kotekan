#!/usr/bin/env python3
"""Tests for elem_shared_ref.band_consensus on synthetic polls.

    /home/kvand/gnss/venv-ft/bin/python test_elem_shared_ref.py

An instance whose band has no reference pins to its own first model, so its global phase is
arbitrary while its shape is the fleet's. The consensus must form from such a band, phase-aligned,
and still leave out a model whose shape it does not describe. On a band that is already pinned the
reference keeps the pinned instances' phase, and an instance off that phase is listed, not dropped.

A posted reference carries the zeros the instance's generated config holds, so a re-post does
not put an input the generator took out back at full weight.

A band that has assemblers and no entry in the reference (a new chain) is one whose instances
each hold their own pin. The watch takes its first reference from the fleet consensus once the
band has gone without one for MIN_RUN_S, unless the band is held.
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

    def test_gps_and_sbas_l5_on_one_gpu_are_two_instances(self):
        # Both chains' assemblers are band l5. Named by node/GPU alone they collided, so the
        # consensus saw half the band while the re-take quorum counted all of it (n 16 of 22).
        res = {}
        for i, a in enumerate((0, 50, 100, 150, 200, 250)):
            for chain in ("gps_l5", "sbas_l5"):
                g = np.concatenate(
                    [
                        model(self.rng, self.shape[0], a),
                        model(self.rng, self.shape[1], -a),
                    ]
                )
                res[(NODES[i // 2], "gnss%d_%s_n2assemble" % (i % 2, chain))] = {
                    "shared_warm": True,
                    "shared_frozen": False,
                    "g_shared": [[float(x.real), float(x.imag)] for x in g],
                }
        e = esr.band_consensus(res, "l5", esr.EXCLUDE_DEG, esr.MIN_SIM)
        self.assertEqual(e["n"], [12, 12])
        self.assertEqual(
            esr.inst_name("cx27", "gnss1_sbas_l5_n2assemble"), "cx27/1:sbas"
        )
        self.assertEqual(esr.inst_name("cx27", "gnss1_gps_l5_n2assemble"), "cx27/1")

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
            (model(self.rng, self.shape[0], 0.0), model(self.rng, self.shape[1], p))
            for p in p1
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
        maj = float(
            np.degrees(np.angle(np.mean([np.exp(1j * np.radians(p)) for p in p1[:6]])))
        )
        self.assertLess(
            abs(offset_deg(r1, self.shape[1] * np.exp(1j * np.radians(maj)))), 2.0
        )


class TestBandOf(unittest.TestCase):
    def test_both_namings(self):
        """The snapshot's band of an assembler, named by its broker chain key or by the band
        tag that came before it: GPS and SBAS L5 share l5, as before."""
        for stage, band in (
            ("gnss0_gps_l5_n2assemble", "l5"),
            ("gnss0_n2assemble", "l5"),
            ("gnss1_sbas_l5_n2assemble", "l5"),
            ("gnss1_l5_n2assemble", "l5"),
            ("gnss0_gal_e5a_n2assemble", "e5a"),
            ("gnss0_e5a_n2assemble", "e5a"),
            ("gnss1_gw_1202_n2assemble", "1202"),
        ):
            self.assertEqual(esr.band_of(stage), band, stage)


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


class TestWatchFirstTake(unittest.TestCase):
    """watch on a fleet whose reference has no entry for one of its bands."""

    def setUp(self):
        self.rng = np.random.default_rng(1112)
        self.shape = (true_shape(self.rng), true_shape(self.rng))
        self.d = tempfile.mkdtemp()
        for n in NODES:
            with open(os.path.join(self.d, "chord_gnss_%s_multi.yaml" % n), "w") as fh:
                fh.write("gnss0_gps_l1ca_n2assemble:\n  elem_positions_epoch: E\n")
        self.snap = os.path.join(self.d, "ref.json")
        flat = [[1.0, 0.0]] * (2 * N_HALF)
        with open(self.snap, "w") as fh:
            json.dump(
                {
                    "epoch": "E",
                    "taken_utc": "2026-01-01T00:00:00Z",
                    "bands": {"l5": {"ref": flat}},
                },
                fh,
            )
        self.out = os.path.join(self.d, "elem_ref")
        self.real = (esr.GEN, esr.urllib.request.urlopen, esr.poll)
        esr.GEN = os.path.join(self.d, "chord_gnss_%s_multi.yaml")
        self.node = esr.urllib.request.urlopen = FakeNode()
        self.set_fleet(n_noise=0)

    def tearDown(self):
        esr.GEN, esr.urllib.request.urlopen, esr.poll = self.real
        shutil.rmtree(self.d)

    def set_fleet(self, n_noise):
        """Ten gps_l1ca assemblers at scattered global phases; n_noise of them learned nothing."""
        halves = [
            (model(self.rng, self.shape[0], a), model(self.rng, self.shape[1], -a))
            for a in range(0, 360, 36)
        ]
        for i in range(n_noise):
            halves[i] = (noise_model(self.rng), noise_model(self.rng))
        res = poll(halves, band="gps_l1ca")
        esr.poll = lambda: res

    def watch(self, act=True, hold=""):
        a = argparse.Namespace(
            file=self.snap, dir=self.out, mode="live", act=act, hold=hold
        )
        with contextlib.redirect_stdout(io.StringIO()):
            esr.watch(a)
        with open(os.path.join(self.out, "current.json")) as fh:
            return json.load(fh)

    def age(self):
        """Move the band's no-reference timer past MIN_RUN_S."""
        path = os.path.join(self.out, "state.json")
        with open(path) as fh:
            st = json.load(fh)
        st["stale"]["l1ca"] -= esr.MIN_RUN_S + 1.0
        with open(path, "w") as fh:
            json.dump(st, fh)

    def auto(self):
        try:
            with open(os.path.join(self.out, "auto_ref.json")) as fh:
                return json.load(fh)
        except OSError:
            return None

    def test_first_reference_is_taken_after_min_run(self):
        cur = self.watch()
        self.assertEqual((self.node.posts, self.auto()), ([], None))
        self.assertEqual(cur["alerts"], [])
        self.age()
        cur = self.watch()
        self.assertEqual(len(self.node.posts), 10)
        self.assertEqual({body["mode"] for _url, body in self.node.posts}, {"live"})
        auto = self.auto()
        self.assertEqual(sorted(auto["bands"]), ["l1ca", "l5"])
        self.assertIn("l1ca", auto["auto"])
        self.assertTrue(
            any("taken from the fleet consensus" in n for n in cur["acted"])
        )
        for pol, f in enumerate(ref_halves(auto["bands"]["l1ca"])):
            self.assertGreater(abs(np.vdot(self.shape[pol], f)), 0.99)
        # the band is in the reference now: the next pass takes nothing
        del self.node.posts[:]
        self.watch()
        self.assertEqual(self.node.posts, [])

    def test_held_band_is_reported_and_left(self):
        cur = self.watch(hold="1202,l1ca")
        self.assertEqual((self.node.posts, self.auto()), ([], None))
        self.assertEqual(
            cur["alerts"], ["l1ca: 10 assemblers and no reference (held, not taken)"]
        )
        with open(os.path.join(self.out, "state.json")) as fh:
            self.assertNotIn("l1ca", json.load(fh)["stale"])

    def test_without_act_it_is_reported(self):
        self.watch(act=False)
        self.age()
        cur = self.watch(act=False)
        self.assertEqual((self.node.posts, self.auto()), ([], None))
        self.assertEqual(len(cur["alerts"]), 1)
        self.assertIn("no reference for", cur["alerts"][0])
        self.assertIn("take a snapshot", cur["alerts"][0])

    def test_no_consensus_is_no_reference(self):
        self.set_fleet(n_noise=5)  # 5 of 10 agree; two thirds are needed
        self.watch()
        self.age()
        cur = self.watch()
        self.assertEqual((self.node.posts, self.auto()), ([], None))
        self.assertEqual(len(cur["alerts"]), 1)
        self.assertIn("the fleet has no consensus", cur["alerts"][0])


class TestActMode(unittest.TestCase):
    """--act in live mode keeps an instance in shape mode (re-posts and re-takes alike)."""

    def test_shape_is_kept(self):
        live = argparse.Namespace(mode="live")
        self.assertEqual(esr.act_mode(live, {"mode": "shape"}), "shape")
        self.assertEqual(esr.act_mode(live, {"mode": "off"}), "live")
        self.assertEqual(esr.act_mode(live, None), "live")
        self.assertEqual(
            esr.act_mode(argparse.Namespace(mode="off"), {"mode": "shape"}), "off"
        )


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
