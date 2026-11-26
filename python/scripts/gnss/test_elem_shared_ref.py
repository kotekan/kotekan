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

The F-engine starts with some inputs whole samples earlier or later than before. The watch gives
a start in the frame0 file a row in the step table, measures its steps from the pass kept before
it and a pass after it, and turns a reference entry made under an earlier start by them, unless
the models then fit the entry worse. An entry it replaces is kept in DIR/replaced.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elem_shared_ref as esr  # noqa: E402
import fengine_steps  # noqa: E402

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

    def test_l5_chains_on_one_gpu_are_separate_instances(self):
        # Both chains' assemblers are band l5. Under one name per node/GPU the consensus sees
        # one chain's model of each GPU while the re-take quorum counts both.
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
        self.assertEqual(
            esr.inst_name("cx27", "gnss1_qzss_l5_n2assemble"), "cx27/1:qzss"
        )
        self.assertEqual(esr.inst_name("cx27", "gnss1_gps_l5_n2assemble"), "cx27/1")

    def test_chains_that_share_a_band_have_their_own_names(self):
        # every chain of the chains table: a new chain on a band that has one needs a suffix
        names = {}
        for r in esr.gnss_chain_table.load():
            name = esr.inst_name("cx27", "gnss1_%s_n2assemble" % r.key)
            names.setdefault(r.shared_ref_band, []).append(name)
        for band, of_band in names.items():
            self.assertEqual(len(of_band), len(set(of_band)), (band, of_band))

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

    def test_the_band_is_the_chains_row(self):
        """The band comes from the chain's row in the chains table, the one the generator copies
        into the assembler, not from the chain key; a chain without a row is refused."""
        rows = [
            r._replace(shared_ref_band="l1ca") if r.key == "sbas_l1" else r
            for r in esr.gnss_chain_table.load()
        ]
        saved = dict(esr._BANDS)
        esr._BANDS.clear()
        esr._BANDS.update({r.key: r.shared_ref_band for r in rows})
        try:
            self.assertEqual(esr.band_of("gnss0_sbas_l1_n2assemble"), "l1ca")
            del esr._BANDS["gal_e5a"]
            with self.assertRaises(SystemExit) as e:
                esr.band_of("gnss0_gal_e5a_n2assemble")
            self.assertIn("gal_e5a has no row", str(e.exception))
        finally:
            esr._BANDS.clear()
            esr._BANDS.update(saved)


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
                "  kotekan_stage: GnssGpuRecordAssemble\n"
                "gnss1_e5a_n2assemble:\n  elem_positions_epoch: E\n  elem_sum: true\n"
                "  kotekan_stage: GnssGpuRecordAssemble\n"
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

    def test_rows_of_a_record_host(self):
        """The same from the rows of gnss<g>_records, rewritten in place of the stages."""
        path = esr.GEN % "cx51"
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        host = {
            "gnss%s_records"
            % k[4]: {
                "kotekan_stage": "GnssGpuRecordAssemble",
                "elem_positions_epoch": "E",
                "chains": {k[: -len("_n2assemble")]: v},
            }
            for k, v in cfg.items()
        }
        for v in cfg.values():
            del v["kotekan_stage"], v["elem_positions_epoch"]
        with open(path, "w") as fh:
            yaml.safe_dump(host, fh)
        self.assertEqual(
            esr.endpoints(),
            [("cx51", "gnss0_n2assemble"), ("cx51", "gnss1_e5a_n2assemble")],
        )
        self.assertEqual(esr.config_zeros("cx51", "gnss0_n2assemble"), {1, 3})
        self.assertEqual(esr.config_zeros("cx51", "gnss1_e5a_n2assemble"), set())

    def test_a_record_host_is_asked_by_chain(self):
        """A row of gnss<g>_records named by its chain is asked at its host, by chain; the body
        it posts is the row's own."""
        with open(esr.GEN % "node-a", "w") as fh:
            yaml.safe_dump(
                {
                    "gnss0_records": {
                        "kotekan_stage": "GnssGpuRecordAssemble",
                        "elem_positions_epoch": "E",
                        "chains": {"gnss0_gps_l5": {"elem_sum_shared_ref": [0.0] * 8}},
                    }
                },
                fh,
            )
        self.assertEqual(
            esr.request("node-a", "gnss0_gps_l5_n2assemble", "get_elem_cal")[0],
            "http://node-a:12048/gnss0_records/get_elem_cal?chain=gps_l5",
        )
        ok, _ = esr.post_note(
            "node-a", "gnss0_gps_l5_n2assemble", self.REF, "live", "test"
        )
        url, body = self.node.posts[0]
        self.assertEqual(
            url, "http://node-a:12048/gnss0_records/set_elem_sum_shared_ref"
        )
        self.assertEqual(body["chain"], "gps_l5")
        self.assertEqual(body["body"]["ref"], [[0.0, 0.0]] * 4)

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
                fh.write(
                    "gnss0_gps_l1ca_n2assemble:\n  elem_positions_epoch: E\n"
                    "  kotekan_stage: GnssGpuRecordAssemble\n"
                )
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

    def test_a_committed_copy_of_the_watchs_file_lists_what_changed_since(self):
        with open(self.snap) as fh:
            snap = json.load(fh)
        snap["auto"] = {"l5": "2025-12-31T00:00:00Z"}
        with open(self.snap, "w") as fh:
            json.dump(snap, fh)
        self.watch()
        self.age()
        cur = self.watch()
        self.assertEqual(sorted(self.auto()["auto"]), ["l1ca", "l5"])
        self.assertTrue(any("(re-taken: l1ca)" in line for line in cur["alerts"]))

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


def utc(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


class TestWatchSteps(unittest.TestCase):
    """watch across an F-engine start that stepped some inputs by whole samples."""

    BANDS = {
        "gps_l5": "l5",
        "gps_l2c": "l2c",
        "glo_l2oc": "l2oc",
        "bds_b3i": "b3i",
        "gps_l1ca": "l1ca",
    }
    TS_NS = 0.3125
    DARK = 7  # an input with no signal in any band: no step can be read for it

    def setUp(self):
        self.rng = np.random.default_rng(1126)
        self.freqs = esr.band_freqs()
        self.shape = {}
        for band in self.BANDS.values():
            pol0 = true_shape(self.rng)
            pol0[self.DARK] = 0.0
            self.shape[band] = np.concatenate(
                [pol0 / np.linalg.norm(pol0), true_shape(self.rng)]
            )
        self.moved = [0] * (2 * N_HALF)
        self.moved[2:6] = [4] * 4
        self.moved[8:10] = [-8] * 2
        self.moved[N_HALF + 3 : N_HALF + 7] = [-3] * 4
        self.moved[N_HALF + 10] = 8
        self.read = list(self.moved)
        self.read[self.DARK] = None
        # The start was long enough ago that the models hold nothing from before it.
        self.start = int(time.time() - esr.AFTER_START_S) - 100
        self.prev = self.start - 100000
        self.d = tempfile.mkdtemp()
        for n in NODES:
            with open(os.path.join(self.d, "chord_gnss_%s_multi.yaml" % n), "w") as fh:
                fh.write(
                    "gnss0_gps_l5_n2assemble:\n  elem_positions_epoch: E\n"
                    "  kotekan_stage: GnssGpuRecordAssemble\n"
                )
        self.snap = os.path.join(self.d, "ref.json")
        self.frame0 = os.path.join(self.d, "frame0.json")
        self.steps = os.path.join(self.d, "sample_steps.json")
        self.out = os.path.join(self.d, "elem_ref")
        with open(self.frame0, "w") as fh:
            json.dump(
                [
                    {"frame0_int": t, "start_utc": utc(t)}
                    for t in (self.prev, self.start)
                ],
                fh,
            )
        self.set_ref(self.shape, self.start - 3600)
        self.real = (esr.GEN, esr.urllib.request.urlopen, esr.poll)
        esr.GEN = os.path.join(self.d, "chord_gnss_%s_multi.yaml")
        self.node = esr.urllib.request.urlopen = FakeNode()

    def tearDown(self):
        esr.GEN, esr.urllib.request.urlopen, esr.poll = self.real
        shutil.rmtree(self.d)

    def stepped(self, steps):
        """Each band's shape with its inputs `steps` samples later."""
        return {
            band: v
            * np.exp(-1j * fengine_steps.phase(steps, self.freqs[band], self.TS_NS))
            for band, v in self.shape.items()
        }

    def set_ref(self, shapes, taken):
        """The reference file: an entry per band, taken at `taken`."""
        self.ref = {
            band: [[float(x.real), float(x.imag)] for x in v]
            for band, v in shapes.items()
        }
        bands = {band: {"ref": ref} for band, ref in self.ref.items()}
        with open(self.snap, "w") as fh:
            json.dump({"epoch": "E", "taken_utc": utc(taken), "bands": bands}, fh)

    def set_table(self, steps):
        """The step table: the start before, then this one with `steps`; with "no row", the
        start before alone."""
        table = {"adc_sample_ns": self.TS_NS, "n_inputs": 2 * N_HALF, "rows": []}
        fengine_steps.add_start(table, self.prev, utc(self.prev))
        if steps != "no row":
            fengine_steps.add_start(table, self.start, utc(self.start))
            table["rows"][-1]["steps"] = steps
        fengine_steps.save(table, self.steps)

    def fleet(self, shapes):
        """A poll of ten assemblers per band that all hold the band's shape."""
        res = {}
        for chain, band in self.BANDS.items():
            v = shapes[band]
            halves = [
                (model(self.rng, v[:N_HALF], 0.0), model(self.rng, v[N_HALF:], 0.0))
                for _ in range(10)
            ]
            res.update(poll(halves, band=chain))
        return res

    def set_fleet(self, shapes):
        res = self.fleet(shapes)
        esr.poll = lambda: res

    def watch(self, act=True):
        a = argparse.Namespace(
            file=self.snap,
            dir=self.out,
            mode="live",
            act=act,
            hold="",
            frame0=self.frame0,
            steps=self.steps,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            esr.watch(a)
        with open(os.path.join(self.out, "current.json")) as fh:
            return json.load(fh)

    def auto(self):
        try:
            with open(os.path.join(self.out, "auto_ref.json")) as fh:
                return json.load(fh)
        except OSError:
            return None

    def replaced(self):
        out = {}
        for path in glob.glob(os.path.join(self.out, "replaced", "*.json")):
            with open(path) as fh:
                kept = json.load(fh)
            out[kept["band"]] = kept
        return out

    def test_a_new_start_gets_a_row(self):
        self.set_table("no row")
        self.set_fleet(self.stepped(self.moved))
        cur = self.watch()
        row = fengine_steps.load(self.steps)["rows"][-1]
        self.assertEqual(
            (row["frame0_int"], row["prev_frame0_int"], row["steps"]),
            (self.start, self.prev, None),
        )
        # no pass was kept before the start, so its steps cannot be measured, and no entry
        # can be turned by them
        self.assertEqual((self.node.posts, self.auto()), ([], None))
        self.assertEqual(len(cur["alerts"]), 2)
        self.assertIn("are not measured: 0 bands", cur["alerts"][0])
        self.assertIn("b3i, l1ca, l2c, l2oc, l5: made under", cur["alerts"][1])
        self.assertIn("were not measured", cur["alerts"][1])

    def test_steps_are_measured_and_the_reference_turned(self):
        self.set_table("no row")
        before = self.start - 60
        esr.save_models(self.out, before, utc(before), self.fleet(self.shape))
        after = self.stepped(self.moved)
        self.set_fleet(after)
        cur = self.watch()
        row = fengine_steps.load(self.steps)["rows"][-1]
        self.assertEqual(row["steps"], self.read)
        self.assertIn("element models of b3i l1ca l2c l2oc l5", row["source"])
        self.assertTrue(
            any("the steps of the F-engine start" in n for n in cur["acted"])
        )
        self.assertEqual(len(self.node.posts), 50)
        auto = self.auto()
        self.assertEqual(sorted(auto["auto"]), sorted(self.BANDS.values()))
        for band, entry in auto["bands"].items():
            self.assertEqual(entry["frame0"], self.start)
            for pol, f in enumerate(ref_halves(entry)):
                want = after[band][pol * N_HALF : (pol + 1) * N_HALF]
                self.assertGreater(abs(np.vdot(want, f)), 0.99)
        kept = self.replaced()
        self.assertEqual(sorted(kept), sorted(self.BANDS.values()))
        self.assertEqual(kept["l5"]["entry"]["ref"], self.ref["l5"])
        self.assertEqual(kept["l5"]["taken_utc"], utc(self.start - 3600))
        # the entries are of this start now: the next pass turns nothing
        del self.node.posts[:]
        self.watch()
        self.assertEqual(self.node.posts, [])

    def test_without_act_the_turn_is_reported(self):
        self.set_table(self.read)
        self.set_fleet(self.stepped(self.moved))
        cur = self.watch(act=False)
        self.assertEqual((self.node.posts, self.auto()), ([], None))
        self.assertEqual(len(cur["alerts"]), 5)
        for line in cur["alerts"]:
            self.assertIn("(--act turns it)", line)

    def test_a_turn_the_models_fit_worse_is_refused(self):
        self.set_table(self.read)
        self.set_fleet(self.shape)  # the models did not move
        cur = self.watch()
        self.assertEqual((self.node.posts, self.auto()), ([], None))
        self.assertEqual(len(cur["alerts"]), 5)
        for line in cur["alerts"]:
            self.assertIn("left as it is", line)

    def test_an_entry_taken_after_the_start_is_left(self):
        """The frame0 file can trail a start, so an entry is of the last start before the time
        it was taken, whatever start it records."""
        self.set_table(self.read)
        after = self.stepped(self.moved)
        self.set_ref(after, self.start + 60)
        self.set_fleet(after)
        cur = self.watch()
        self.assertEqual((self.node.posts, self.auto()), ([], None))
        self.assertEqual(cur["alerts"], [])

    def test_made_under(self):
        table = {"rows": [{"frame0_int": 1000}, {"frame0_int": 2000}]}
        self.assertEqual(esr.made_under(table, {}, utc(1500)), 1000)
        self.assertEqual(esr.made_under(table, {"frame0": 1000}, utc(2500)), 2000)
        self.assertEqual(esr.made_under(table, {"frame0": 2000}, utc(2500)), 2000)
        self.assertIsNone(esr.made_under(table, {}, utc(500)))

    def test_a_retaken_entry_is_kept_and_the_new_one_records_the_start(self):
        self.set_table([0] * (2 * N_HALF))
        signs = np.where(np.arange(2 * N_HALF) % 2, 1.0, -1.0)
        self.set_ref(
            {band: v * signs for band, v in self.shape.items()}, self.prev - 60
        )
        self.set_fleet(self.shape)
        self.watch()
        path = os.path.join(self.out, "state.json")
        with open(path) as fh:
            st = json.load(fh)
        for band in st["stale"]:
            st["stale"][band] -= esr.MIN_RUN_S + 1.0
        with open(path, "w") as fh:
            json.dump(st, fh)
        self.watch()
        self.assertEqual(len(self.node.posts), 50)
        kept = self.replaced()
        self.assertEqual(sorted(kept), sorted(self.BANDS.values()))
        self.assertEqual(kept["l5"]["entry"]["ref"], self.ref["l5"])
        for entry in self.auto()["bands"].values():
            self.assertEqual(entry["frame0"], self.start)

    def test_snapshot_records_the_start(self):
        self.set_fleet(self.stepped(self.moved))
        out = os.path.join(self.d, "new.json")
        with contextlib.redirect_stdout(io.StringIO()):
            esr.snapshot(argparse.Namespace(out=out, frame0=self.frame0))
        with open(out) as fh:
            for entry in json.load(fh)["bands"].values():
                self.assertEqual(entry["frame0"], self.start)


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
