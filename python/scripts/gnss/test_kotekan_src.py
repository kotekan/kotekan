#!/usr/bin/env python3
"""Tests for kotekan_src: where the kotekan source tree is found, in the order the module
docstring gives, and that nothing found is an error at use, never at import.

    python3 python/scripts/gnss/test_kotekan_src.py

Each case copies kotekan_src.py into a checkout under a fresh temp dir and imports the copy
with a controlled environment, so neither this checkout's tree nor the caller's KOTEKAN_SRC
can answer for it.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.join(HERE, "kotekan_src.py")
MARK = os.path.join("lib", "stages", "gnss")


class TestResolve(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="kotekan_src-"))
        self.addCleanup(shutil.rmtree, self.tmp)
        self.checkout = os.path.join(self.tmp, "gnss-broker")
        self.gnss_py = os.path.join(self.checkout, "python", "scripts", "gnss")
        os.makedirs(self.gnss_py)
        shutil.copy(MODULE, self.gnss_py)
        self.submodule = os.path.join(self.checkout, "kotekan")
        self.sibling = os.path.join(self.tmp, "kotekan")
        self.elsewhere = os.path.join(self.tmp, "elsewhere")

    def tree(self, root):
        """Make `root` hold a kotekan tree, as far as the resolver looks."""
        os.makedirs(os.path.join(root, MARK))
        return root

    def load(self, env=None):
        """The copied kotekan_src, imported with os.environ = `env`."""
        spec = importlib.util.spec_from_file_location(
            "kotekan_src_under_test", os.path.join(self.gnss_py, "kotekan_src.py")
        )
        mod = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, env or {}, clear=True):
            spec.loader.exec_module(mod)
        return mod

    def test_env_wins(self):
        for root in (self.checkout, self.submodule, self.sibling, self.elsewhere):
            self.tree(root)
        k = self.load({"KOTEKAN_SRC": self.elsewhere})
        self.assertEqual(k.SRC, self.elsewhere)

    def test_env_is_taken_as_it_is(self):
        self.tree(self.checkout)
        missing = os.path.join(self.tmp, "no-such-tree")
        k = self.load({"KOTEKAN_SRC": missing + os.sep})
        self.assertEqual(k.SRC, missing)
        self.assertEqual(k.gnss_cpp("x.hpp"), os.path.join(missing, MARK, "x.hpp"))

    def test_empty_env_is_unset(self):
        self.tree(self.checkout)
        self.assertEqual(self.load({"KOTEKAN_SRC": ""}).SRC, self.checkout)

    def test_this_checkout_first(self):
        for root in (self.checkout, self.submodule, self.sibling):
            self.tree(root)
        self.assertEqual(self.load().SRC, self.checkout)

    def test_submodule_before_sibling(self):
        self.tree(self.submodule)
        self.tree(self.sibling)
        self.assertEqual(self.load().SRC, self.submodule)

    def test_uninitialised_submodule_is_passed_over(self):
        os.makedirs(self.submodule)
        self.tree(self.sibling)
        self.assertEqual(self.load().SRC, self.sibling)

    def test_sibling(self):
        self.tree(self.sibling)
        self.assertEqual(self.load().SRC, self.sibling)

    def test_paths(self):
        self.tree(self.checkout)
        k = self.load()
        self.assertEqual(
            k.src("config", "chord", "pathfinder.j2"),
            os.path.join(self.checkout, "config", "chord", "pathfinder.j2"),
        )
        self.assertEqual(
            k.gnss_cpp("gnssSignal.hpp"),
            os.path.join(self.checkout, MARK, "gnssSignal.hpp"),
        )
        self.assertEqual(k.gnss_cpp(), os.path.join(self.checkout, MARK))

    def test_nothing_found_raises_at_use_not_import(self):
        os.makedirs(self.submodule)
        os.makedirs(self.sibling)
        k = self.load()
        self.assertIsNone(k.SRC)
        with self.assertRaises(k.NotFound):
            k.src("config")
        with self.assertRaises(k.NotFound) as cm:
            k.gnss_cpp()
        msg = str(cm.exception)
        self.assertIn("KOTEKAN_SRC", msg)
        for place in (self.checkout, self.submodule, self.sibling):
            self.assertIn(place, msg)
        self.assertTrue(issubclass(k.NotFound, RuntimeError))

    def test_plain_import_without_a_tree(self):
        """The import a reader makes, in a process of its own: no tree, no KOTEKAN_SRC."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("KOTEKAN_SRC", "PYTHONPATH")
        }
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        p = subprocess.run(
            [sys.executable, "-c", "import kotekan_src; print(kotekan_src.SRC)"],
            cwd=self.gnss_py,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "None")


if __name__ == "__main__":
    unittest.main(verbosity=2)
