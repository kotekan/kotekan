#!/usr/bin/env python3
"""Record and wire-frame layout constants (RECORD_FLOATS, ELEM_FLOATS, ...), from the interface
golden.

The widths are C++ constants (gnssRecord.hpp, gnssTelem.hpp, gnssGpuChain.hpp) but kotekan
frames are sized in the generated yaml (`n_prn * record_floats * sizeof_float32`). The generator
takes the constants from the interface document kotekan builds from those headers
(gnss_broker/iface.py, the golden the broker also speaks) instead of restating them, so they
cannot drift. The frame-size expressions below (telem_frame_bytes, cube_frame_bytes) restate
their C++ counterparts and must change with them; the golden's frame_bytes samples check them
(config/test_record_layout.py).
"""

import os
import sys

# python/scripts/gnss, for the interface golden (gnss_broker/iface.py).
_GNSS_PY = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "python", "scripts", "gnss"
    )
)
if _GNSS_PY not in sys.path:
    sys.path.append(_GNSS_PY)
from gnss_broker import iface  # noqa: E402


def _const(section, name):
    return iface.golden().const(section, name)


def record_floats():
    """Per-PRN record header width, floats (gnss::RECORD_FLOATS)."""
    return _const("record.row", "RECORD_FLOATS")


def elem_floats():
    """Per-element block width, floats (gnss::ELEM_FLOATS)."""
    return _const("record.elem", "ELEM_FLOATS")


def record_stride(n_elem):
    """Full per-PRN stride: RECORD_FLOATS + n_elem * ELEM_FLOATS."""
    return record_floats() + n_elem * elem_floats()


def prnctl_bytes():
    """sizeof(gnss_gpu::PrnCtl), the size of the golden's PrnCtl table (gnssGpuChain.hpp).

    The generator sizes the epl and ctl frames with it (gnss_gpu::frame_bytes, restated in
    Python with this one term read). A literal copy goes stale when PrnCtl grows, and every node
    then refuses to start on a frame that is (new - old) * 16 * n_prn bytes short.

    The document takes the size from the compiler (sizeof), and its field table must tile it.
    """
    return iface.golden().sections["gpu_chain"]["PrnCtl"]["size"]


def telem_header_bytes():
    """gnss::TELEM_HEADER_BYTES -- the telemetry wire frame's header (gnssTelem.hpp)."""
    return _const("telem", "TELEM_HEADER_BYTES")


def telem_max_chan():
    """gnss::TELEM_MAX_CHAN -- the most comb columns a wire row may carry (not the stride)."""
    return _const("telem", "TELEM_MAX_CHAN")


def chan_floats():
    """gnss::CHAN_FLOATS -- floats per comb column (re, im, energy for prompt, early, late)."""
    return _const("record.chan", "CHAN_FLOATS")


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
    return _const("cube", "CUBE_CHAIN_CHARS")


def cube_header_bytes():
    """gnss::CUBE_HEADER_BYTES: the fixed fields, then the CUBE_CHAIN_CHARS chain string.

    A stale copy fails silently: bufferRecv closes a frame of the wrong size with a message
    about frame_size and nothing about this file.
    """
    return _const("cube", "CUBE_HEADER_BYTES")


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
