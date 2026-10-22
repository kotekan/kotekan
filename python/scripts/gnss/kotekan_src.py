"""Where the kotekan source tree is, for everything the GNSS Python reads out of it: the C++
headers that the telemetry, record and signal tables are parsed from, the stock config templates
the generator renders, and the build trees whose binaries the tools run.

The first of these that applies, with <checkout> the checkout this file is in, as a physical
path (scripts/gnss/site/kotekan_src.sh takes the same one):

    $KOTEKAN_SRC            when set, taken as it is
    <checkout>              kotekan and the GNSS Python in one tree
    <checkout>/kotekan      a kotekan submodule
    <checkout>/../kotekan   a kotekan checkout beside this one (~/gnss/kotekan on the gnss VM)

A candidate counts only if it holds lib/stages/gnss. When none does, SRC is None and src() and
gnss_cpp() raise NotFound, naming KOTEKAN_SRC and the places tried. Importing this module never
raises: only asking for a path in a tree that is not there does. This is the one place the
search is written down, so that a reader asks here rather than climbing out of its own
directory.
"""

import os

_HERE = os.path.dirname(os.path.realpath(__file__))
_CHECKOUT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_MARK = os.path.join("lib", "stages", "gnss")


class NotFound(RuntimeError):
    """No kotekan source tree: KOTEKAN_SRC is unset and no default place holds one."""


def _resolve(environ, checkout):
    """(the kotekan tree or None, the places tried)."""
    env = environ.get("KOTEKAN_SRC")
    if env:
        return os.path.abspath(env), [env]
    tried = [
        checkout,
        os.path.join(checkout, "kotekan"),
        os.path.join(os.path.dirname(checkout), "kotekan"),
    ]
    for path in tried:
        if os.path.isdir(os.path.join(path, _MARK)):
            return path, tried
    return None, tried


#: The kotekan source tree, or None when there is none to be found.
SRC, _TRIED = _resolve(os.environ, _CHECKOUT)


def src(*parts):
    """A path inside the kotekan source tree."""
    if SRC is None:
        raise NotFound(
            "no kotekan source tree: set KOTEKAN_SRC to a kotekan checkout "
            "(none of %s holds %s)" % (", ".join(_TRIED), _MARK)
        )
    return os.path.join(SRC, *parts)


def gnss_cpp(*parts):
    """A path inside lib/stages/gnss, where the GNSS stages and their headers are."""
    return src("lib", "stages", "gnss", *parts)
