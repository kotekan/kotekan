#!/usr/bin/env python3
"""Tests for the ranging-code checks that compile the C++: a broken C++ module fails them.

    /home/kvand/gnss/venv-ft/bin/python test_code_check_cpp.py

Each check is pointed at a temporary copy of its C++ source, either with one table entry
changed (the copy compiles, and its codes differ from the python mirror) or with a syntax error
(the copy does not compile). Both must make main() exit non-zero; only a missing compiler may
skip the bit-exact comparison, and not with GNSS_REQUIRE_CXX set. Needs a C++ compiler on PATH
for all but TestNoCompiler.
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kotekan_src  # noqa: E402
import b2b_code_check  # noqa: E402
import e5b_code_check  # noqa: E402
import glonass_ca_code_check  # noqa: E402
import l1cd_code_check  # noqa: E402
import l2oc_code_check  # noqa: E402
import l3oc_code_check  # noqa: E402

LIB = kotekan_src.gnss_cpp()

# (module, C++ source, the getter main() calls, a table entry and the typo that replaces it).
# Each typo is in PRN 2, which no fingerprint covers, so only the C++ comparison can catch it.
CHECKS = (
    (
        b2b_code_check,
        "beidouB2bCode.cpp",
        "_cpp_codes",
        "0b1000000100101, 0b1000000110100",
        "0b1000000100101, 0b1000000110101",
    ),
    (
        e5b_code_check,
        "galileoE5bCode.cpp",
        "_cpp_codes",
        "0x025C, 0x390D",
        "0x025C, 0x390C",
    ),
    (l1cd_code_check, "gpsL1CCode.cpp", "_cpp_codes", "{5097, 5110", "{5097, 5111"),
    (l3oc_code_check, "glonassL3OCCode.cpp", "_cpp_codes", None, None),
    (l2oc_code_check, "glonassL2OCCode.cpp", "_cpp_codes", None, None),
    (glonass_ca_code_check, "glonassCACode.cpp", "cpp_code", None, None),
)


def run_main(mod):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            rc = mod.main()
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue()


# GNSS_REQUIRE_CXX (the test runner sets it): no compiler fails here rather than skips, since
# each check's main() then fails too.
@unittest.skipIf(
    not any(shutil.which(c) for c in ("g++-12", "g++", "clang++"))
    and not os.environ.get("GNSS_REQUIRE_CXX"),
    "no C++ compiler",
)
class TestCodeCheckCpp(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.saved = [
            (m, g, getattr(m, g, None), getattr(m, "CPP", None))
            for m, _, g, _, _ in CHECKS
        ]

    def tearDown(self):
        for mod, getter, fn, cpp in self.saved:
            setattr(mod, getter, fn)
            mod.CPP = cpp
        self.td.cleanup()

    def broken_copy(self, mod, name, old=None, new=None):
        """Point mod at a copy of its C++ with `old` replaced by `new`, or, with no `old`,
        with a line appended that does not compile."""
        with open(os.path.join(LIB, name)) as f:
            src = f.read()
        if old is None:
            src += "\nthis is not C++;\n"
        else:
            self.assertEqual(src.count(old), 1, "%s: %r is not unique" % (name, old))
            src = src.replace(old, new)
        path = os.path.join(self.td.name, name)
        with open(path, "w") as f:
            f.write(src)
        mod.CPP = path

    def test_shipped_cpp_passes(self):
        for mod, *_ in CHECKS:
            with self.subTest(mod.__name__):
                rc, out = run_main(mod)
                self.assertEqual(rc, 0, out)

    def test_table_typo_fails(self):
        for mod, name, _, old, new in CHECKS:
            if old is None:
                continue
            with self.subTest(mod.__name__):
                self.broken_copy(mod, name, old, new)
                rc, out = run_main(mod)
                self.assertNotEqual(rc, 0, out)
                self.assertIn("DIFFER from the mirror", out)

    def test_compile_failure_fails(self):
        for mod, name, *_ in CHECKS:
            with self.subTest(mod.__name__):
                self.broken_copy(mod, name)
                rc, out = run_main(mod)
                self.assertNotEqual(rc, 0, out)
                self.assertIn("compile failed", out)


class TestNoCompiler(unittest.TestCase):
    """main() when the getter finds no compiler: a skip, unless GNSS_REQUIRE_CXX is set. Needs
    no compiler itself."""

    def setUp(self):
        self.saved = [(m, g, getattr(m, g)) for m, _, g, _, _ in CHECKS]

    def tearDown(self):
        for mod, getter, fn in self.saved:
            setattr(mod, getter, fn)

    def run_without_compiler(self, mod, getter, require):
        setattr(mod, getter, lambda: None)
        with mock.patch.dict(os.environ):
            os.environ.pop("GNSS_REQUIRE_CXX", None)
            if require:
                os.environ["GNSS_REQUIRE_CXX"] = "1"
            return run_main(mod)

    def test_no_compiler_skips(self):
        for mod, _, getter, *_ in CHECKS:
            with self.subTest(mod.__name__):
                rc, out = self.run_without_compiler(mod, getter, False)
                self.assertEqual(rc, 0, out)
                self.assertIn("SKIPPED", out)

    def test_no_compiler_fails_when_required(self):
        for mod, _, getter, *_ in CHECKS:
            with self.subTest(mod.__name__):
                rc, out = self.run_without_compiler(mod, getter, True)
                self.assertNotEqual(rc, 0, out)
                self.assertIn("GNSS_REQUIRE_CXX is set", out)
                self.assertNotIn("SKIPPED", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
