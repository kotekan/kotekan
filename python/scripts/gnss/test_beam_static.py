#!/usr/bin/env python3
"""Tests for gnss_beam_static.py: the element gains, the map's orientation and the co-add guard.

    /home/kvand/gnss/venv/bin/python test_beam_static.py

Needs healpy (the patch projection and pixel centres), so it runs in the venv, not venv-ft.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_arraymap as am  # noqa: E402
import gnss_beam_static as bs  # noqa: E402

BASELINE_DAY, POLSWAP_DAY = (
    "20260901",
    "20261001",
)  # one day in each of the first two epochs


def write_master(d, day, key):
    """A one-chain, 16-element master at nside 8 with every pixel sampled, stamped `key`."""
    import healpy as hp

    nside, E = 8, 16
    npix = 12 * nside * nside
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    th = bs.angsep_deg(
        np.degrees(phi), 90.0 - np.degrees(theta), bs.BORE_AZ, bs.BORE_EL
    )
    n = np.full((1, E, npix), 10, np.uint32)
    s1 = (
        n
        * np.linspace(1.0, 2.0, E)[None, :, None]
        * 10.0 ** (-th / 20.0)[None, None, :]
    )
    chain = {
        "chain": "gps_l5",
        "sys": "G",
        "n_sub": 1,
        "n_elem": E,
        "n_pix": npix,
        "freq_ids": [[5988, 5988]],
    }
    meta = {
        "nside": nside,
        "day": day,
        "source": "l0",
        "units": "pedestal",
        "pointing": "p0_dec40p73",
        "array_epoch_key": key,
        "chains": [chain],
    }
    path = os.path.join(d, "cube_%s_nside8.npz" % day)
    np.savez(
        path,
        meta=json.dumps(meta),
        pix_0=np.arange(npix, dtype=np.int32),
        n_0=n,
        s1_0=s1,
    )
    return path


class TestSvdLogGains(unittest.TestCase):
    def test_sign_follows_the_data(self):
        # S = N * g_e * pattern_p * (1 + 5% noise): the recovered log-gains must rise with the
        # true ones for every seed, never come back mirrored.
        E, P = 16, 200
        flipped = []
        for seed in range(40):
            rng = np.random.default_rng(seed)
            g = 10.0 ** (rng.normal(0.0, 2.0, E) / 10.0)
            pattern = 10.0 ** (rng.uniform(-20.0, 0.0, P) / 10.0)
            N = np.full((E, P), 50.0)
            S = (
                N
                * g[:, None]
                * pattern[None, :]
                * (1.0 + 0.05 * rng.normal(size=(E, P)))
            )
            got, live, _, _ = bs.elem_gains(
                N, S, "svd-log", 8, np.ones(P, bool), quiet=True
            )
            if np.corrcoef(np.log(got[live]), np.log(g[live]))[0, 1] <= 0.0:
                flipped.append(seed)
        self.assertEqual(flipped, [], "svd-log gains mirrored for seed(s) %s" % flipped)


class TestCompass(unittest.TestCase):
    """The compass letters draw_map places must sit where patch() puts that direction."""

    def hot_x(self, az, el, centre, radius):
        import healpy as hp

        nside = 64
        vals = np.full(12 * nside * nside, np.nan)
        vals[hp.ang2pix(nside, np.radians(90.0 - el), np.radians(az))] = 1.0
        img, ax = bs.patch(vals, nside, radius, 201, centre=centre)
        rows, cols = np.nonzero(img == 1.0)
        self.assertTrue(len(cols), "hot pixel not in the patch")
        return float(np.mean(ax[cols]))

    def test_east_of_boresight(self):
        x = self.hot_x(160.0, bs.BORE_EL, (bs.BORE_AZ, bs.BORE_EL), 25.0)
        self.assertEqual(np.sign(x), np.sign(bs.COMPASS["E"][0]), "x = %.2f" % x)

    def test_east_on_the_hemisphere(self):
        x = self.hot_x(90.0, 45.0, (180.0, 90.0), 90.0)
        self.assertEqual(np.sign(x), np.sign(bs.COMPASS["E"][0]), "x = %.2f" % x)


class TestMap(unittest.TestCase):
    """`map` end to end on synthetic masters, with draw_map replaced by a recorder."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.drawn, self.prepared = [], []
        self._saved = (bs.draw_map, bs.prepare, sys.argv)
        prepare = bs.prepare
        bs.draw_map = lambda *a, **k: self.drawn.append(k) or a[4]
        bs.prepare = (
            lambda args: self.prepared.append(prepare(args)) or self.prepared[-1]
        )

    def tearDown(self):
        bs.draw_map, bs.prepare, sys.argv = self._saved
        shutil.rmtree(self.d)

    def run_map(self, masters, *extra):
        out = os.path.join(self.d, "m.png")
        sys.argv = (
            ["gnss_beam_static.py", "map"] + masters + ["--out", out, "--size", "41"]
        )
        sys.argv += list(extra)
        bs.main()
        return self.drawn[-1]

    def polswap_master(self, day=POLSWAP_DAY):
        return write_master(self.d, day, am.for_day(POLSWAP_DAY).key())

    def test_wide_boresight_map_is_not_a_sky_map(self):
        k = self.run_map([self.polswap_master()], "--radius", "60")
        self.assertFalse(k["sky"], "a boresight-centred map labelled as zenith-centred")
        self.assertEqual(list(k["rings"]), [5, 10, 20])

    def test_zenith_map_is_a_sky_map(self):
        k = self.run_map(
            [self.polswap_master()], "--radius", "90", "--centre", "zenith"
        )
        self.assertTrue(k["sky"])
        self.assertEqual(list(k["rings"]), [30, 60, 90])

    def test_two_array_epochs_are_refused(self):
        a = write_master(self.d, BASELINE_DAY, am.for_day(BASELINE_DAY).key())
        with self.assertRaises(SystemExit) as cm:
            self.run_map([a, self.polswap_master()])
        self.assertIn("array configurations", str(cm.exception.code))

    def test_one_array_epoch_is_summed(self):
        k = self.run_map([self.polswap_master(), self.polswap_master("20261002")])
        self.assertFalse(k["sky"])
        self.assertEqual(
            sorted(self.prepared[-1]["info"]["days"]), [POLSWAP_DAY, "20261002"]
        )

    def test_boresight_comes_from_the_epoch(self):
        # The same table with the pointing's boresight moved to el 70. The key digests the
        # pointing's name, not its boresight, so the master's stamp still resolves.
        with open(am.CONFIG) as fh:
            table = json.load(fh)
        table["pointings"]["p0_dec40p73"]["bore_el_deg"] = 70.0
        path = os.path.join(self.d, "epochs.json")
        with open(path, "w") as fh:
            json.dump(table, fh)
        backend = am.JsonBackend
        am.JsonBackend = lambda: backend(path)
        # The boresight the chain-alignment annulus and the projection centre are given.
        seen = {"offsets": [], "patch": []}
        offsets, patch = bs.chain_offsets_from, bs.patch
        bs.chain_offsets_from = lambda *a: seen["offsets"].append(a[5]) or offsets(*a)
        bs.patch = lambda *a: seen["patch"].append(a[5]) or patch(*a)
        try:
            k = self.run_map([self.polswap_master()])
        finally:
            am.JsonBackend = backend
            bs.chain_offsets_from, bs.patch = offsets, patch
        import healpy as hp

        th = self.prepared[-1]["th"]
        at70, at81 = (hp.ang2pix(8, np.radians(90.0 - e), np.pi) for e in (70.0, 81.41))
        self.assertLess(th[at70], th[at81], "angles measured from el 81.41, not el 70")
        self.assertEqual(tuple(k["bore"]), (180.0, 70.0))
        self.assertEqual([tuple(b) for b in seen["offsets"]], [(180.0, 70.0)])
        self.assertEqual([tuple(c) for c in seen["patch"]], [(180.0, 70.0)])


if __name__ == "__main__":
    unittest.main()
