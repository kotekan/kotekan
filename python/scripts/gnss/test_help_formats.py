#!/usr/bin/env python3
"""argparse help strings must survive percent-formatting.

WHY THIS EXISTS. argparse expands every help string through percent-format at --help time,
so one bare `%` anywhere makes --help itself die with an exception -- and nothing else. The
broker runs perfectly; only the way anyone DISCOVERS a flag is broken. That is a silent
failure of the worst kind: it hides new flags precisely when someone is looking for them.

It happened four times in this file in one day (2026-08-11): `%Y/%m/%d` in the #25 spectrum
archive help killed --help that morning and nobody noticed for eight hours, then three more
turned up (`30-36%`, `67-82%`, `2%`) once it was fixed -- and then the sentence added to
EXPLAIN the escaping contained `%-format`, which reads as a left-justified float and broke it
again. Escaping by inspection does not converge; testing does.

Run: python3 test_help_formats.py
"""
import argparse
import os
import runpy
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
# argparse substitutes a dict of the action's own attributes; the VALUES do not matter, only
# that a well-formed string consumes them by NAME (%(default)s) rather than positionally.
PARAMS = {"default": 0, "prog": "p", "choices": "c", "type": "t",
          "const": 0, "metavar": "m", "dest": "d", "nargs": 1}


def misformats(h):
    """Why this help string is unsafe, or None.

    TWO failure modes, and only the first is loud. A bare `%Y` or `%-format` RAISES, killing
    --help outright. But a bare `%` followed by something that happens to be a valid
    conversion -- "50% sign" is `% s` -- does NOT raise: it silently consumes the parameter
    dict and renders it into the text. The help still displays, mangled, which is why
    crash-only detection is not enough. A string with no intentional %(name)s substitution
    must come back through the formatter completely UNCHANGED.
    """
    try:
        out = h % PARAMS
    except Exception as e:
        return "%s: %s" % (type(e).__name__, e)
    # The invariant is NOT "unchanged": a correctly escaped %% legitimately becomes %.
    # It is that formatting does nothing BEYOND that unescaping. Anything else means a bare
    # % consumed a parameter and the rendered help differs from what was written.
    if "%(" not in h and out != h.replace("%%", "%"):
        return "silently mangled (a bare %% consumed the parameter dict)"
    return None


def collect_bad(script):
    bad = []
    orig = argparse.ArgumentParser.add_argument

    def patched(self, *a, **kw):
        h = kw.get("help")
        if isinstance(h, str):
            why = misformats(h)
            if why:
                bad.append(((a[0] if a else "?"), why))
        return orig(self, *a, **kw)

    argparse.ArgumentParser.add_argument = patched
    argv = sys.argv
    try:
        sys.argv = [script, "--a-flag-that-does-not-exist"]
        try:
            runpy.run_path(script, run_name="__main__")
        except SystemExit:
            pass            # argparse exits on the bad flag -- by then every add_argument ran
    finally:
        argparse.ArgumentParser.add_argument = orig
        sys.argv = argv
    return bad


class TestHelpFormats(unittest.TestCase):

    def test_broker_help_strings_are_formattable(self):
        bad = collect_bad(os.path.join(HERE, "gps_distributed_broker.py"))
        self.assertEqual(bad, [], "help strings argparse cannot format (double the %% in "
                                  "any literal percent sign): " + repr(bad))

    def test_the_detector_would_catch_a_regression(self):
        """A test for the test: a bare %% must actually be reported, or this file is a gate
        that cannot fail -- which is the whole reason the original bug survived a day."""
        ap = argparse.ArgumentParser()
        for h, should_flag in (("plain text", False),
                               ("default is %(default)s", False),
                               ("literal 50%% escaped", False),
                               ("bare 50% sign", True),        # mangles, does NOT raise
                               ("a %Y date", True),            # raises
                               ("percent-%-format", True)):    # raises
            self.assertEqual(bool(misformats(h)), should_flag,
                             "misjudged help string %r -> %r" % (h, misformats(h)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
