# The GNSS checkout the site scripts sit in, and the kotekan source tree it runs against.
# Sourced, never executed:
#
#     . "$(dirname "${BASH_SOURCE[0]}")/kotekan_src.sh"
#     K=$GNSS_CHECKOUT
#     KOTEKAN_SRC=$(gnss_kotekan_src "$K")
#
# GNSS_CHECKOUT is the checkout this file is in, found from its own path, so that cron, systemd
# and ssh, which start a script by its path in a bare environment, need nothing set. It is the
# physical path, as kotekan_src.py's must be: through a symlink, a logical and a physical path
# look for ../kotekan beside different directories.
#
# gnss_kotekan_src <checkout> prints the kotekan tree for that checkout, by the rule
# python/scripts/gnss/kotekan_src.py writes down: the first of these that applies,
#
#     $KOTEKAN_SRC            when set, taken as it is (made absolute)
#     <checkout>              kotekan and the GNSS Python in one tree
#     <checkout>/kotekan      a kotekan submodule
#     <checkout>/../kotekan   a kotekan checkout beside this one (~/gnss/kotekan on the gnss VM)
#
# where a candidate counts only if it holds lib/stages/gnss. When none does it prints nothing,
# says why on stderr and returns 1. The Python and the shell must find the same tree, so a
# change to the rule goes in both; test_site_scripts.py runs the two over the same layouts.
#
# Nothing here is exported: a script exports KOTEKAN_SRC to the programs that read kotekan.

GNSS_CHECKOUT=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)

gnss_kotekan_src() {
    local c=${1:?usage: gnss_kotekan_src <checkout>} p
    if [ -n "${KOTEKAN_SRC:-}" ]; then
        realpath -ms -- "$KOTEKAN_SRC"
        return
    fi
    c=$(realpath -m -- "$c")    # absolute and physical: the dirname of a relative . is .
    for p in "$c" "$c/kotekan" "$(dirname -- "$c")/kotekan"; do
        if [ -d "$p/lib/stages/gnss" ]; then
            printf '%s\n' "$p"
            return 0
        fi
    done
    echo "no kotekan source tree: set KOTEKAN_SRC to a kotekan checkout (none of $c," \
         "$c/kotekan, $(dirname -- "$c")/kotekan holds lib/stages/gnss)" >&2
    return 1
}
