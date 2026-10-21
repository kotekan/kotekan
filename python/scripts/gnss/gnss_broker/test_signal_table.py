"""Every descriptor in gnssSignal.hpp resolves through gnss::signal_by_name.

    python3 -m gnss_broker.test_signal_table

The node stages (cudaGnssChordTrack, GnssChannelizedSearch) look a config's signal name up in
ALL_SIGNALS through signal_by_name; a descriptor defined in the header but left out of that
list compiles, passes the config CI batches (none of which names it) and FATAL_ERRORs the node
at its first start on a config that does (SBAS_L5_Q_MC, 2026-10-20, cx51 down four launches).
The broker's own registry parses the header's descriptors, so it sees the new name and starts
the chain against a node that cannot. This test holds the two lists equal.

@author Keith Vanderlinde
"""

import re
import sys

from gnss_broker import signals

_TABLE_RX = re.compile(r"ALL_SIGNALS\[\] = \{(.*?)\};", re.S)


def main():
    src = open(signals._HPP).read()
    defined = set(signals._load_descriptors())
    m = _TABLE_RX.search(src)
    if not m:
        print("FAIL: ALL_SIGNALS not found in %s" % signals._HPP)
        return 1
    listed = set(re.findall(r"&([A-Z0-9_]+)", m.group(1)))
    missing = sorted(defined - listed)
    extra = sorted(listed - defined)
    for n in missing:
        print(
            "  descriptor %s is defined but not in ALL_SIGNALS: a node config naming it dies"
            % n
        )
    for n in extra:
        print("  ALL_SIGNALS lists %s, which no descriptor defines" % n)
    print("%d descriptors, %d in the table" % (len(defined), len(listed)))
    print("\n%s (%d check(s) failed)" % ("FAIL" if (missing or extra) else "PASS", len(missing) + len(extra)))
    return 1 if (missing or extra) else 0


if __name__ == "__main__":
    sys.exit(main())
