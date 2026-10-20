"""Where the kotekan source tree is, for everything the GNSS Python reads out of it: the C++
headers that the telemetry, record and signal tables are parsed from, the stock config templates
the generator renders, and the build trees whose binaries the tools run.

$KOTEKAN_SRC names it. Unset, it is the checkout this file is in, which is where kotekan and
the GNSS Python live together today. This is the one place that default is written down, so
that the Python can move out of the kotekan tree by setting the variable (or changing this
default) rather than by editing every reader.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

#: The kotekan source tree.
SRC = os.path.abspath(
    os.environ.get("KOTEKAN_SRC") or os.path.join(_HERE, "..", "..", "..")
)


def src(*parts):
    """A path inside the kotekan source tree."""
    return os.path.join(SRC, *parts)


def gnss_cpp(*parts):
    """A path inside lib/stages/gnss, where the GNSS stages and their headers are."""
    return src("lib", "stages", "gnss", *parts)
