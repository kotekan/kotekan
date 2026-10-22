#!/usr/bin/env python3
"""Record and wire-frame layout constants (RECORD_FLOATS, ELEM_FLOATS, ...), read from the C++
headers that define them.

The widths are C++ constants (gnssRecord.hpp, gnssTelem.hpp, gnssGpuChain.hpp) but kotekan
frames are sized in the generated yaml (`n_prn * record_floats * sizeof_float32`). The generator
reads the constants from the headers instead of restating them, so they cannot drift. The
frame-size expressions below (telem_frame_bytes, cube_frame_bytes) restate their C++
counterparts and must change with them.

Parsed, not imported: these are plain `constexpr int` one-liners, and a regex over them is
cheaper than any build-time codegen. If a header stops matching, this raises rather than
silently returning a stale default.
"""

import os
import re
import sys

# kotekan_src (python/scripts/gnss) says where the kotekan source tree is.
_GNSS_PY = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "python", "scripts", "gnss"
    )
)
if _GNSS_PY not in sys.path:
    sys.path.append(_GNSS_PY)
import kotekan_src  # noqa: E402

HEADER = kotekan_src.gnss_cpp("gnssRecord.hpp")
TELEM_HEADER = kotekan_src.gnss_cpp("gnssTelem.hpp")
CHAIN_HEADER = kotekan_src.gnss_cpp("gnssGpuChain.hpp")


def _read(name, path=HEADER):
    with open(path) as fh:
        text = fh.read()
    m = re.search(r"^\s*constexpr\s+int\s+%s\s*=\s*(\d+)\s*;" % name, text, re.M)
    if not m:
        raise SystemExit(
            "%s: could not parse `constexpr int %s` -- the header's shape "
            "changed and config/gnss_record_layout.py needs updating" % (path, name)
        )
    return int(m.group(1))


def record_floats():
    """Per-PRN record header width, floats (gnss::RECORD_FLOATS)."""
    return _read("RECORD_FLOATS")


def elem_floats():
    """Per-element block width, floats (gnss::ELEM_FLOATS)."""
    return _read("ELEM_FLOATS")


def record_stride(n_elem):
    """Full per-PRN stride: RECORD_FLOATS + n_elem * ELEM_FLOATS."""
    return record_floats() + n_elem * elem_floats()


def prnctl_bytes():
    """sizeof(gnss_gpu::PrnCtl) -- parsed from its static_assert in gnssGpuChain.hpp.

    The generator sizes the epl and ctl frames with it (gnss_gpu::frame_bytes, restated in
    Python with this one term read). A literal copy goes stale when PrnCtl grows, and every node
    then refuses to start on a frame that is (new - old) * 16 * n_prn bytes short.

    Parsed from the static_assert rather than a constexpr because the size is the compiler's
    to decide; the assert is the one place the expected value is written down, and every build
    checks it.
    """
    with open(CHAIN_HEADER) as fh:
        text = fh.read()
    m = re.search(r"static_assert\s*\(\s*sizeof\s*\(\s*PrnCtl\s*\)\s*==\s*(\d+)", text)
    if not m:
        raise SystemExit(
            "%s: could not parse the sizeof(PrnCtl) static_assert -- the header's "
            "shape changed and config/gnss_record_layout.py needs updating"
            % CHAIN_HEADER
        )
    return int(m.group(1))


def telem_header_bytes():
    """gnss::TELEM_HEADER_BYTES -- the telemetry wire frame's header (gnssTelem.hpp)."""
    return _read("TELEM_HEADER_BYTES", TELEM_HEADER)


def telem_max_chan():
    """gnss::TELEM_MAX_CHAN -- the most comb columns a wire row may carry (not the stride)."""
    return _read("TELEM_MAX_CHAN", TELEM_HEADER)


def chan_floats():
    """gnss::CHAN_FLOATS -- floats per comb column (re, im, energy for prompt, early, late)."""
    return _read("CHAN_FLOATS")


def telem_row_floats(n_cols):
    """gnss::telem_row_floats -- the record header plus this sender's own comb columns."""
    return record_floats() + n_cols * chan_floats()


def telem_frame_bytes(n_rec, n_prn, n_cols):
    """Bytes of one telemetry wire frame -- the same expression as gnss::telem_frame_bytes.

    The shape is per sender: an instance ships the comb columns it despreads and its chain's PRN
    row count. The gather's bufferRecv accepts any frame up to its buffer (`allow_short_frames`)
    and closes the connection of a sender whose frame exceeds it, so the generator sizes the
    gather for the widest possible sender on each axis: telem_rows_max() rows and TELEM_MAX_CHAN
    columns.
    """
    return telem_header_bytes() + n_rec * n_prn * telem_row_floats(n_cols) * 4


def cube_chain_chars():
    """gnss::CUBE_CHAIN_CHARS -- "<hostname>/<stage>" field width in the beam-cube frame."""
    return _read("CUBE_CHAIN_CHARS")


def cube_header_bytes():
    """gnss::CUBE_HEADER_BYTES.

    Parsed from the header's own expression rather than restated here: it is written as
    `104 + CUBE_CHAIN_CHARS`, and a literal copy would go stale the first time a field is
    added -- silently, because bufferRecv closes a frame of the wrong size with a message about
    frame_size and nothing about this file.
    """
    with open(HEADER) as fh:
        text = fh.read()
    m = re.search(
        r"^\s*constexpr\s+size_t\s+CUBE_HEADER_BYTES\s*=\s*"
        r"(\d+)\s*\+\s*CUBE_CHAIN_CHARS\s*;",
        text,
        re.M,
    )
    if not m:
        raise SystemExit(
            "%s: could not parse `constexpr size_t CUBE_HEADER_BYTES = N + "
            "CUBE_CHAIN_CHARS` -- the header's shape changed and "
            "config/gnss_record_layout.py needs updating" % HEADER
        )
    return int(m.group(1)) + cube_chain_chars()


def cube_frame_bytes(max_prn, max_bins, n_elem):
    """Bytes of one beam-cube wire frame -- the same expression as gnss::cube_frame_bytes.

    Uniform across every sender, unlike the telemetry frame: one bufferRecv serves the whole
    fleet and closes any connection whose frame_size disagrees with its buffer. The frame is
    sized for the maxima (the generator's CUBE_MAX_PRN slots and CUBE_MAX_BINS channels); a
    chain with fewer sends zero rows, which the offline converter strips.
    """
    return (
        cube_header_bytes()
        + max_bins * 2 * 4  # freq_id lo/hi (int32)
        + max_prn * 3 * 4  # prn, n_rec, n_reanchor (int32)
        + max_prn * 8  # phi0 (double)
        + max_prn * max_bins * 2 * 4  # w, energy (float)
        + max_prn * max_bins * n_elem * 3 * 4
    )  # coh_re, coh_im, incoh (float)


if __name__ == "__main__":
    print(
        "RECORD_FLOATS %d  ELEM_FLOATS %d  TELEM_HEADER_BYTES %d"
        % (record_floats(), elem_floats(), telem_header_bytes())
    )
    print(
        "CUBE_HEADER_BYTES %d  cube_frame_bytes(32, 8, 32) %d"
        % (cube_header_bytes(), cube_frame_bytes(32, 8, 32))
    )
