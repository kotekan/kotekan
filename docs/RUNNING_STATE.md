# RUNNING STATE at 2026-07-31 20:47 UTC -- captured for the compaction handoff

## Processes
     416821 Thu Jul 30 23:15:17 2026 /home/kvand/gnss/venv/bin/python -u livebeam_server.py --no-power-stream --http-port 8080 --ws-port 8539 --kotekan-rest-port 12050 --lat 49.32075144444 --lon -119.62081125 --alt 545 --gps-search-stage agg_search --gps-combiner-stage agg_search --gps-constellations G
     519837 Fri Jul 31 13:39:30 2026 ./build/kotekan/kotekan --config config/generated/chord_gnss_agg.yaml --bind-address 0.0.0.0:12050
     520351 Fri Jul 31 14:02:37 2026 /home/kvand/gnss/venv/bin/python -u python/scripts/gnss/gps_distributed_broker.py --rest-url http://localhost:12049 --detectors http://localhost:12050/gps_search --trackers  --combiner gnss0_combine --almanac --almanac-source brdc --dead-reckon --narrow-search --time0-endpoint telescope/time0_ns --dr-clock-chips 0.0 --constellation G --carrier-hz 1176.45e6 --code-length 10230 --hops-per-sec 195312.5 --lat 49.32075144444 --lon -119.62081125 --alt 545 --mask-deg 0 --interval 2 --search-margin-wide-hz 150 --search-margin-hz 100
     762993 Fri Jul 31 20:11:50 2026 sudo ./build/kotekan/kotekan --config config/generated/chord_gnss_cx19.yaml --bind-address 0.0.0.0:12049
     763483 Fri Jul 31 20:12:21 2026 sudo ./build/kotekan/kotekan --config config/generated/chord_gnss_cx19.yaml --bind-address 0.0.0.0:12049
     763484 Fri Jul 31 20:12:21 2026 ./build/kotekan/kotekan --config config/generated/chord_gnss_cx19.yaml --bind-address 0.0.0.0:12049

## Tracker build check (must show channel_ids or the node predates 51b1ca034)
    localhost:
      gnss0_gpu: signal=GPS_L5_Q_NH code_trim=True channel_ids=[5972, 5988, 6004, 6020, 6036, 6052, 6068]
      gnss1_gpu: signal=GPS_L5_Q_NH code_trim=True channel_ids=[5980, 5996, 6012, 6028, 6044, 6060, 6076]
    cx27:
      gnss0_gpu: signal=GPS_L5_Q_NH code_trim=True channel_ids=[5984, 6000, 6016, 6032, 6048, 6064]
      gnss1_gpu: signal=GPS_L5_Q_NH code_trim=True channel_ids=[5976, 5992, 6008, 6024, 6040, 6056, 6072]

## Ports
    12048 production (choco owns; generator refuses)
    12049 node instances (cx19, cx27)
    12050 aggregator search (stage name gps_search)
    8080/8539 viewer

## Latest search passes
    INFO: /gps_search: GnssChannelizedSearch[/gps_search]: pass best snr 1084.25 (PRN 32, nh 12/20), threshold 5.50, pure-noise ceiling ~4.84
    INFO: /gps_search: GnssChannelizedSearch[/gps_search]: pass best snr 158.25 (PRN 32, nh 17/20), threshold 5.50, pure-noise ceiling ~4.84
    INFO: /gps_search: GnssChannelizedSearch[/gps_search]: pass best snr 152.19 (PRN 32, nh 19/20), threshold 5.50, pure-noise ceiling ~4.84
