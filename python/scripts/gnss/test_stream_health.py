#!/usr/bin/env python3
"""#8+ stream health: SK RFI reduction and drop-counter parsing.

    python3 python/scripts/gnss/test_stream_health.py

sk_summary reduces the per-element SK array to a flag count against a band that SCALES with
the estimator's accumulation; parse_drops pulls the right counters out of a /metrics dump.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_broker.fleet import sk_summary, sk_url_for, parse_drops  # noqa: E402


def main():
    fails = []

    # 1. sk_url_for derives the sibling SK endpoint per GPU.
    if sk_url_for("http://cx43:12048/gnss0_srch_tap") != "http://cx43:12048/rfi_sk_metrics/sk_metrics_0/sk":
        fails.append("sk_url_for gnss0 wrong")
    if sk_url_for("http://cx43:12048/gnss1_srch_tap") != "http://cx43:12048/rfi_sk_metrics/sk_metrics_1/sk":
        fails.append("sk_url_for gnss1 wrong")
    if sk_url_for("http://cx43:12048/gnss0_combine") is not None:
        fails.append("sk_url_for should reject non-tap urls")

    # 2. clean SK (all ~1) flags nothing.
    s = sk_summary({"ema_frames": 256, "sk": [1.0, 1.05, 0.98, 1.1, 0.95]})
    if s["sk_flagged"] != 0:
        fails.append("clean SK flagged %d (want 0)" % s["sk_flagged"])

    # 3. a jammed channel (SK far from 1) is flagged, and named as worst.
    s = sk_summary({"ema_frames": 256, "sk": [1.0, 1.0, 5.0, 1.0]})  # element 2 jammed
    if s["sk_flagged"] != 1 or s["sk_worst"] != 2:
        fails.append("jam: flagged=%d worst=%d (want 1, 2)" % (s["sk_flagged"], s["sk_worst"]))

    # 4. the flag band TIGHTENS with more accumulation (more frames -> smaller sigma).
    lo256 = sk_summary({"ema_frames": 256, "sk": [1.0]})["sk_lo"]
    lo64 = sk_summary({"ema_frames": 64, "sk": [1.0]})["sk_lo"]
    if not (lo256 > lo64):     # 256 frames -> band closer to 1 than 64 frames
        fails.append("band should tighten with frames: lo256=%.3f lo64=%.3f" % (lo256, lo64))

    # 4b. null (unpopulated) channels are dropped, not crashed on or counted.
    s = sk_summary({"ema_frames": 256, "sk": [1.0, None, 5.0, None, 1.0]})
    if s is None or s["sk_flagged"] != 1 or s["sk_n"] != 3 or s["sk_worst"] != 2:
        fails.append("null channels mishandled: %r" % s)

    # 5. empty SK -> None, never a crash or a false 'clean'.
    if sk_summary({"ema_frames": 256, "sk": []}) is not None:
        fails.append("empty SK should be None")

    # 6. parse_drops pulls this GPU's stages + node-level dpdk (max over ports).
    mt = ('kotekan_buffer_send_dropped_frame_count{stage_name="/gnss0_srch_send"} 80537\n'
          'kotekan_buffer_send_dropped_frame_count{stage_name="/gnss1_srch_send"} 80566\n'
          'kotekan_buffer_send_dropped_frame_count{stage_name="/gnss0_telem_send"} 92488\n'
          'kotekan_dpdk_nic_rx_missed_total{stage_name="/dpdk",port="0"} 40323\n'
          'kotekan_dpdk_nic_rx_missed_total{stage_name="/dpdk",port="1"} 44490\n'
          'kotekan_dpdk_distributor_ring_full_dropped_packets_total{stage_name="/d",port="0"} 0\n')
    d0 = parse_drops(mt, 0)
    if d0.get("srch_send_drops") != 80537 or d0.get("telem_send_drops") != 92488:
        fails.append("gpu0 drops wrong: %r" % d0)
    if d0.get("dpdk_missed") != 44490:   # max over ports
        fails.append("dpdk_missed should be max over ports: %r" % d0)
    d1 = parse_drops(mt, 1)
    if d1.get("srch_send_drops") != 80566 or "telem_send_drops" in d1:  # gpu1 has no telem line here
        fails.append("gpu1 drops wrong: %r" % d1)

    if fails:
        print("FAIL:\n  " + "\n  ".join(fails))
        return 1
    print("ok: sk_url_for, sk_summary (clean/jam/band/empty), parse_drops (per-gpu + dpdk max)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
