#!/usr/bin/env python3
"""RECORD_FLOATS / ELEM_FLOATS, read from the C++ header that defines them.

WHY THIS FILE EXISTS. The record width is a C++ constant (gnssRecord.hpp) but the frame is
sized in yaml (`n_prn * record_floats * sizeof_float`), and until 2026-08-07 nothing linked
the two. RECORD_FLOATS went 24 -> 26 on the CHORD branch and config/gnss_node.yaml kept
saying 24, so 34 airspy stages FATAL'd at construction on a frame 256 B short per PRN. The
guard that caught it is good (it prints the number to use), but the drift should not be
possible in the first place -- so both generators now READ the header instead of restating
it. The CHORD generator had the same latent bug in a different disguise: a Python literal
`26 + n_elem * 12` whose comment claimed the value was "kept in one place".

Parsed, not imported: these are plain `constexpr int` one-liners, and a regex over them is
cheaper and more robust than any build-time codegen. If the header stops matching, this
raises rather than silently returning a stale default.
"""

import os
import re

HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "lib", "stages", "gnss", "gnssRecord.hpp")
TELEM_HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "lib", "stages", "gnss", "gnssTelem.hpp")


def _read(name, header=None):
    path = header or HEADER
    with open(path) as fh:
        text = fh.read()
    m = re.search(r"^\s*constexpr\s+int\s+%s\s*=\s*(\d+)\s*;" % name, text, re.M)
    if not m:
        raise SystemExit("%s: could not parse `constexpr int %s` -- the header's shape "
                         "changed and config/gnss_record_layout.py needs updating" % (path, name))
    return int(m.group(1))


def record_floats(header=None):
    """Per-PRN record header width, floats (gnss::RECORD_FLOATS)."""
    return _read("RECORD_FLOATS", header)


def elem_floats(header=None):
    """Per-element block width, floats (gnss::ELEM_FLOATS)."""
    return _read("ELEM_FLOATS", header)


def record_stride(n_elem, header=None):
    """Full per-PRN stride: RECORD_FLOATS + n_elem * ELEM_FLOATS."""
    return record_floats(header) + n_elem * elem_floats(header)


def telem_header_bytes():
    """gnss::TELEM_HEADER_BYTES -- the task #59 wire header (gnssTelem.hpp)."""
    return _read("TELEM_HEADER_BYTES", TELEM_HEADER)


def telem_max_chan():
    """gnss::TELEM_MAX_CHAN -- comb columns reserved per wire row."""
    return _read("TELEM_MAX_CHAN", TELEM_HEADER)


def chan_floats():
    """gnss::CHAN_FLOATS -- floats per comb column (prompt re, im, energy)."""
    return _read("CHAN_FLOATS")


def telem_row_floats():
    """gnss::TELEM_ROW_FLOATS -- the record header PLUS the reserved comb columns."""
    return record_floats() + telem_max_chan() * chan_floats()


def telem_frame_bytes(n_rec, n_prn):
    """Bytes of one telemetry wire frame -- the SAME expression as gnss::telem_frame_bytes.

    ⚠️ Every sender's out_buf AND the gather's receive buffer must be sized from this, with the
    same (n_rec, n_prn). bufferRecv compares frame_size on the wire against its own buffer and
    CLOSES THE CONNECTION on a mismatch, so a disagreement here does not corrupt data -- it
    silently delivers none, which is its own kind of expensive.
    """
    return telem_header_bytes() + n_rec * n_prn * telem_row_floats() * 4


if __name__ == "__main__":
    print("RECORD_FLOATS %d  ELEM_FLOATS %d  TELEM_HEADER_BYTES %d"
          % (record_floats(), elem_floats(), telem_header_bytes()))
