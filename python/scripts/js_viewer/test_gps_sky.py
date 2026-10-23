#!/usr/bin/env python3
"""GpsSkyResource (/gps_sky) keeps the sky plot whole when one of its sources fails.

  * A TLE group that fails to load is retried, and a loaded group is reloaded once a day;
    load_gps_satellites(max_age_s=...) re-downloads an old cache file and keeps it when the
    download fails.
  * A failing GLONASS TLE merge leaves the broker's G/E/C sky in the reply.
  * The broker's bind address (0.0.0.0) is connected to as the loopback.
  * The GLONASS filter stands down when any running R signal's capability lookup fails,
    rather than filtering to the signals whose lookups succeeded.

Needs the viewer's interpreter (twisted, autobahn, skyfield):

    ~/gnss/venv/bin/python test_gps_sky.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gnss"))
sys.path.insert(0, HERE)
import gps_beamtrack as bt
import livebeam_server as lb


def _sky(lat, lon, alt, _sats=None):
    return {p: (10.0, 45.0) for p in _sats}


class TestTleReload(unittest.TestCase):
    def setUp(self):
        self.r = lb.GpsSkyResource(43.0, -79.0, 0.0, 0.0)
        self.clock = [1000.0]
        self.calls = []
        self.down = set()  # TLE groups whose load raises
        patches = [
            mock.patch.object(bt, "load_gps_satellites", side_effect=self._load),
            mock.patch.object(bt, "predict_skypos", side_effect=_sky),
            mock.patch.object(lb.time, "monotonic", side_effect=lambda: self.clock[0]),
            mock.patch.object(self.r, "_glonass_capable_set", return_value=None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _load(self, url, max_age_s=None):
        self.calls.append((url, max_age_s))
        if any(g in url for g in self.down):
            raise IOError("network down")
        return {1: object(), 2: object()}

    def _tle(self, want):
        return [s["prn"] for s in self.r._tle_skypos(want)]

    def test_failed_group_is_retried(self):
        self.down = {"glo-ops"}
        self.assertEqual(self._tle(["R"]), [])
        self.down = set()
        self.clock[0] += 60.0
        self.assertEqual(self._tle(["R"]), [])  # inside the retry interval
        self.clock[0] += 600.0
        self.assertEqual(self._tle(["R"]), [1, 2])

    def test_loaded_group_is_reloaded_daily(self):
        self._tle(["R"])
        n = len(self.calls)
        self.clock[0] += 3600.0
        self._tle(["R"])
        self.assertEqual(len(self.calls), n)
        self.clock[0] += 86400.0
        self._tle(["R"])
        self.assertEqual(len(self.calls), 2 * n)
        self.assertTrue(all(a == 86400.0 for _u, a in self.calls))


class TestTleCacheAge(unittest.TestCase):
    def setUp(self):
        self._home = os.environ.get("HOME")
        self.d = tempfile.mkdtemp()
        os.environ["HOME"] = self.d
        cache = os.path.join(self.d, ".cache", "kotekan_gps")
        os.makedirs(cache)
        self.path = os.path.join(cache, "tle_glo-ops.txt")
        open(self.path, "w").close()

    def tearDown(self):
        if self._home is not None:
            os.environ["HOME"] = self._home
        shutil.rmtree(self.d, ignore_errors=True)

    def _load(self, age_s, fail_reload=False, **kw):
        t = time.time() - age_s
        os.utime(self.path, (t, t))
        reloads = []

        class Loader:
            def __init__(self, directory, verbose=False):
                pass

            def tle_file(self, url, reload=False, filename=None):
                reloads.append(reload)
                if reload and fail_reload:
                    raise IOError("cannot download")
                return []

        with mock.patch("skyfield.api.Loader", Loader):
            bt.load_gps_satellites(bt.GLONASS_TLE_URL, **kw)
        return reloads

    def test_old_cache_is_reloaded(self):
        self.assertEqual(self._load(2 * 86400.0, max_age_s=86400.0), [True])

    def test_fresh_cache_is_kept(self):
        self.assertEqual(self._load(3600.0, max_age_s=86400.0), [False])

    def test_failed_reload_keeps_the_old_copy(self):
        self.assertEqual(
            self._load(2 * 86400.0, fail_reload=True, max_age_s=86400.0), [True, False]
        )

    def test_no_max_age_never_reloads(self):
        self.assertEqual(self._load(100 * 86400.0), [False])


class TestBrokerSky(unittest.TestCase):
    ROWS = [
        {"prn": 3, "const": "G", "az": 10.0, "el": 50.0},
        {"prn": 5, "const": "G", "az": 200.0, "el": 30.0},
    ]

    def test_tle_failure_keeps_the_broker_sky(self):
        r = lb.GpsSkyResource(43.0, -79.0, 0.0, 0.0, broker=("127.0.0.1", 12060))
        with mock.patch.object(
            r, "_brdc_skypos", return_value=list(self.ROWS)
        ), mock.patch.object(
            r, "_tle_skypos", side_effect=ImportError("gps_beamtrack")
        ):
            out = r._compute()
        self.assertTrue(out["ok"])
        self.assertEqual(out["sats"], self.ROWS)

    def test_sbas_rows_map_to_the_s_tag(self):
        # The broker's /get_sky serves the SBAS GEOs as const "S" (BRDC S records).
        r = lb.GpsSkyResource(43.0, -79.0, 0.0, 0.0, broker=("127.0.0.1", 12060))
        rows = [
            {"prn": 131, "const": "S", "az": 176.6, "el": 33.4},
            {"prn": 3, "const": "G", "az": 10.0, "el": 50.0},
        ]
        body = mock.MagicMock()
        body.__enter__.return_value.read.return_value = json.dumps(rows).encode()
        with mock.patch("urllib.request.urlopen", return_value=body):
            got = r._brdc_skypos(["G", "S"])
        self.assertIn({"prn": 131, "const": "S", "az": 176.6, "el": 33.4}, got)

    def test_sbas_is_asked_of_the_broker_never_of_tle(self):
        # S has no TLE group: it is wanted from the broker, and the TLE paths (merge and
        # whole-sky fallback) never see it.
        r = lb.GpsSkyResource(43.0, -79.0, 0.0, 0.0, broker=("127.0.0.1", 12060))
        seen = {}

        def brdc(want):
            seen["brdc"] = list(want)
            return [{"prn": 131, "const": "S", "az": 176.6, "el": 33.4}]

        def tle(want=None):
            seen.setdefault("tle", []).append(list(want or []))
            return []

        with mock.patch.object(r, "_brdc_skypos", side_effect=brdc), mock.patch.object(
            r, "_tle_skypos", side_effect=tle
        ):
            out = r._compute()
        self.assertIn("S", seen["brdc"])
        self.assertTrue(all("S" not in w for w in seen["tle"]))
        self.assertIn(131, [x["prn"] for x in out["sats"] if x["const"] == "S"])
        with mock.patch.object(r, "_brdc_skypos", return_value=None), mock.patch.object(
            r, "_tle_skypos", side_effect=tle
        ):
            r._compute()
        self.assertNotIn("S", seen["tle"][-1])

    def test_bind_address_connects_to_loopback(self):
        r = lb.GpsSkyResource(43.0, -79.0, 0.0, 0.0, broker=("0.0.0.0", 12060))
        urls = []

        def urlopen(url, timeout=None):
            urls.append(url)
            raise OSError("connection refused")

        with mock.patch("urllib.request.urlopen", side_effect=urlopen):
            self.assertIsNone(r._brdc_skypos(["G"]))
        self.assertEqual(urls, ["http://127.0.0.1:12060/get_sky"])


class TestGlonassFilter(unittest.TestCase):
    def _capable(self, sigids, fdma, registry, kset=frozenset()):
        r = lb.GpsSkyResource(43.0, -79.0, 0.0, 0.0, glonass_sigids=sigids)
        fake_bt = types.SimpleNamespace(
            glonass_fdma_trackable=lambda: set(fdma),
            signal_capable_prns=lambda sig, url: set(kset),
            GLONASS_TLE_URL="glo-ops",
        )
        fake_prnmap = types.SimpleNamespace(registry_capable=lambda sig: registry)
        fake_pkg = types.ModuleType("gnss_broker")
        fake_pkg.prnmap = fake_prnmap
        mods = {
            "gps_beamtrack": fake_bt,
            "gnss_broker": fake_pkg,
            "gnss_broker.prnmap": fake_prnmap,
        }
        with mock.patch.dict(sys.modules, mods):
            return r._glonass_capable_set()

    def test_union_of_running_signals(self):
        got = self._capable(["GLO_L2OF", "GLO_L3OC_P"], {1, 2, 3, 4}, {1, 2})
        self.assertEqual(got, {1, 2, 3, 4})

    def test_failed_fdma_lookup_stands_the_filter_down(self):
        self.assertIsNone(self._capable(["GLO_L2OF", "GLO_L3OC_P"], set(), {1, 2}))

    def test_failed_cdma_lookup_stands_the_filter_down(self):
        self.assertIsNone(self._capable(["GLO_L2OF", "GLO_L3OC_P"], {1, 2, 3}, None))

    def test_k_marker_answers_for_the_registry(self):
        got = self._capable(["GLO_L3OC_P"], set(), None, kset={7, 9})
        self.assertEqual(got, {7, 9})


if __name__ == "__main__":
    unittest.main()
