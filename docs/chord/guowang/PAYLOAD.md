# GuoWang 1202.025 MHz: the message content

Status: 2026-11-05 (the high-rate message decoded as a per-GNSS-satellite ephemeris from the
overnight track-beam stream; the 11-04 sections below were decoded during the 11-04 site outage
from stored gw_reduce products, no raw baseband). The signal is in `SIGNAL.md`. Products: cf05 `/var/tmp/kvand-gw1202/reduce/wNNN/<sat>.npz`
(per-frame `car_ph`, `data_bits`, `code_err`, `theta`, `n` = UTC frame number); scripts under
`/home/kvand/gnss/builds/gw1202/`.

Two channels share every 2 ms frame: the sign of the ranging half (low rate) and the cyclic shift
of the CSK half (high rate). ITU-R M.1787-6 files them as 125 / 1500 bit/s; measured 125 and
1488 (1464 data + 24 CRC).

## The low-rate BPSK channel

- Symbols: one sign per frame pair, pairs starting on even UTC frame numbers: 4 ms, 250 sym/s.
  Read from `car_ph` frame to frame with the 10 ms carrier pattern modelled per class n mod 5 (a
  smoothed absolute reference gives 10 % sign errors); `slow_bits2.py` pair agreement 0.9998.
- Code: rate-1/2, K = 7 convolutional, G1 = 171, G2 = 133 (octal, GPS convention, G1 first),
  found by an exhaustive parity-check search (`slow_code_test.py`: 97-98 % of checks, the only
  pair). Hard-decision Viterbi in `slow_bits.py`. Odd-weight generators leave the bit polarity
  ambiguous per gap segment: both are tried per subframe.
- Subframe = 3 s = 375 bits (125 bit/s) on the UTC 3 s grid. Layout (`parse_slow.py`): 12 bits
  outside the CRC, then [3-bit page 1..4][20-bit BDT SOW = the subframe's own start second]
  [payload][CRC-24Q]. The script's CRC covers 339 bits (page + SOW + 316-bit payload);
  12 + 339 + 24 = 375. Its docstring and the notes quote 328 and 351, which count the 12 bits
  twice. Pages 1-4 cycle every 12 s. Every clean subframe's SOW matched its UTC start with
  BDT = UTC + 4 s. Clean 18/28 and 25/55 subframes on two passes (`slow_decode.py`).
- Page 1 payload: [4 bits = 0010][8-bit satellite ID: 139 = H24-4, 126 = H84-1, the high-rate
  header's numbering][17 bits 0x21fa|1, common][10-bit per-satellite field: 481 / 812][~50 bits
  common: ...bfffea88000002][a 24-28-bit per-page word with ~5 satellite-specific bits][filler].
  Pages 2-4: filler only, the byte 0x2d + 8 x (page - 1) repeated. Payloads are constant over a
  pass. Placeholder content; no ephemeris. **That holds for the two inferred-orbit satellites
  decoded first (H24-4, H84-1). Catalogued satellites do not send filler (11-05): DIGUI-121,
  -139 and -142 carry changing content in all four pages, 20-36 of 316 bits per 12 s cycle over
  a union of 30-120 positions per page, 25-35 % differing between satellites at other times.
  What it encodes is open; it is where an ephemeris or clock would sit.**
- CN115685267A (Qixin receiver) describes this pilot channel as convolutional (Viterbi) + CRC
  with week number and SOW, 6 s subframes in its example; ours are 3 s.
- Overnight 11-05 (`builds/gw1202/slow_dataset.py`: the loop's per-frame sign products through
  `gw_nav_live.NavPages`; 6680 CRC-valid subframes from 76 satellites, cf05
  `/var/tmp/kvand-trackbeam/stream/slow_night.npz`; `slow_study.py`, `slow_fields.py`,
  `slow_angles.py`): on the catalogued satellites 23-29 of the 316 payload bits change per 12 s
  cycle, in fixed regions per page (page 1: 35-41, 59-69, 77-91, 118-138, 149-190, 197-232,
  237-291; page 2: 24-32, 47-53, 89-110, 125-141, 171-210, 259-314; page 3: 94-107, 137-166,
  183-202, 216-314; page 4: 101-128, 139-154, 160-167, 169-190, 193-206, 215-221, 223-232,
  236-289). Page 1 bits 26-40 are a 15-bit counter of 12 s cycles (BDT seconds / 12, minus 1);
  page 2 bits 47-52 count every ~48 s. The other changing fields are per-satellite numbers,
  never equal on two satellites at the same time, refreshed each cycle with no simple slope. No
  field tracks the sender's own sqrtA, e, i, node, argument of latitude, latitude, longitude, or
  ECEF position/velocity at any power-of-two scale, so the pages do not carry the sender's orbit
  in the fast channel's form or as a state vector; what they carry is open. For the sign wipe
  this means the changing bits are not forecastable from geometry; only the counters are.
- **11-09 (`slow_dig*.py`, session scratchpad; data `slow_night.npz`): the changing content is a
  ROTATING TABLE, not the sender's state.** Page 1 bits 237-243 are an index k that steps -2 per
  12 s cycle; the 50 deg senders cycle k = 1..108 and the polar senders k = 22..82, i.e. the
  almanac IDs of the sender's OWN shell. The four pages of one cycle carry, bit for bit, the
  268-bit fast-channel almanac entry of satellite k (same-day issue: 98-100 % of indices agree):
  page 2 bits 232-271 = entry 0-39 (header, @28/12), page 4 232-258 = entry 40-66 (e), page 2
  289-313 = 67-91 (delta-i), page 3 232-291 = 92-151 (node, omega), page 4 259-288 = 152-181
  (M0), page 1 244-265 = 182-203, page 3 292-313 = 204-225, page 2 272-288 = 226-242, page 1
  266-290 = 243-267 (Omega-dot, below); bit 243 of page 1 = k parity = the entry's flag. The
  per-sender time series of these bits could never fit the sender's own orbit. What remains
  per sender: the counters, the own-theta field, a per-page record (page 1 bits ~118-232 and
  similar blocks on pages 2-4) rewritten in one go at BDT 5-minute boundaries + 12 s on every
  satellite at once (content open), and two modular ramps (page 1 @59/10 and @77/14, rate
  varying by pass; open). No IEEE floats anywhere; no correlation with the sender's geometry.

## The high-rate CSK message

- Raw: 500 six-bit symbols per UTC second (3000 bit/s), the shift k of each frame's CSK code.
  Slots 0-3: sync word 56 36 55 40, every satellite, every second. Early folds (`csk_fold.py`)
  saw shift 0 about 5x over-represented (fill).
- Coding (`csk_ldpc_probe.py`, 100 % zero syndrome over 139k checks): slots 4-203 LDPC(200,100)
  over GF(64) (BeiDou B1C subframe-2 code), 204-403 LDPC(200,100), 404-499 LDPC(96,48) (BeiDou
  B2a code). p(x) = x^6 + x + 1; H tables from Pocket SDR `sdr_ldpc.c` (copy in
  `itu_refs/pocketsdr/`); the CSK shift is the GF(64) symbol directly; systematic, info first.
  Decoder `gw_ldpc.py`: GF(64) belief propagation with WHT check nodes, LLR = |R_u| x pk_f /
  var_off; recovers 25 % erasures and 20 % weak errors.
- Message: 1464 data bits + CRC-24Q over the info of all three codewords (sync excluded), at the
  end of the third codeword's info. Archive (`csk_extract.py` -> `csk_streams/msgs/<pass>.npz`,
  1488 bits per UTC second with a CRC flag): 4684 of 4930 complete seconds verified (95 %, 63
  passes), later 5986 verified messages from 80 passes.
- Header, bits 0-48: shell + ID 12 bits (slot 4 bit 3 = shell flag, set for 50 deg, clear for
  polar; number = (slot 4 & 3) x 64 + slot 5; DIGUI-87..91 = 20, 19, 21, 22, 23; DIGUI-H54-3 =
  128; code-mates differ by 2); a 4-bit constant (slot 6 high bits); BDT SOW 20 bits (slot 6 low 2
  bits + slots 7-9; the rollover past 262144 at 11-04 00:49Z was predicted and seen); BDT week
  13 bits (slots 10-11 read 8, 31).
- Body: ONE constellation-wide stream. The per-bit change profile repeats with a 490-bit record
  period from bit 49 (records 49-538, 539-1028, 1029-...), ~30-bit entries inside. The 15-bit
  fields at 61 and 551 = BDT SOW / 20 of the DATA EPOCH; the 16-bit fields at 349 and 839 = BDT
  seconds-of-day / 10 of the same epoch. Each satellite lags the epoch by 9-115 s (uplink
  latency). At the same (epoch, page) the payload beyond the header is identical across
  satellites: blocks 98 %, tail/slow/table 99.6-100 % over 48979 cross-satellite pairs. No
  per-satellite data except the header ID.
  - Superseded (11-04 morning): record boundary at 76/566 with 49-75 a "27-bit preamble"
    (sub-fields 37, 28, 34, 35, 85, 40, 30, 30, 16, ~155 bits); the field at 349 as a per-satellite
    issue counter stepping every 9 s; a per-satellite tail and "1309-1415 slow data" (satellites
    compared at different epochs).
- Change rates: blocks 6-7 % per 10 s epoch, 35 % per page. The change pattern has a 9 s
  self-similarity (best lag 9 s in 58/63 passes; ~165 of 1464 bits per cycle vs ~455 between
  consecutive seconds); the decode loop keys its archive by (epoch at bit 349, page = SOW mod 9).
  36-bit blocks recur identically on up to 34 satellites. No page-ID field; no PPP-B2b mask at a
  fixed offset.
- Tail 1056-1308: 40 % / 32 % change, fast, clock-correction-like. 1309-1415: all zero but its
  last 7 bits. 1416-1463, from flip co-occurrence over 5529 transitions (not a snapshot): relative
  bits 0, 5, 10, 15, 29, 34, 39, 44-47 never flip; 1-4 = 6-9 = 11-14 (one 4-bit value sent three
  times), 25-28 = 30-33 = 35-38 = 40-43 (another, four times); 16-24 a separate ~9-bit field; not
  a counter. (The earlier "7 x 5-bit + 14-bit table" was a snapshot label.)
- Variant satellites (DIGUI-169/170/176/177, the inferred H24-x and H54-x; 556 of 5986 messages)
  send 0x6565 or zeros in the epoch fields; H84-1 and H24-1 sent zeros at 11-04 13:10Z.
- Interpretation (11-04, superseded 11-05): live GNSS augmentation corrections at ~10 s epochs.

### Decoded 2026-11-05: a Keplerian ephemeris per GNSS satellite

The overnight track-beam stream (43 051 CRC-valid messages, 87 senders, 11-04 23:27Z to 11-05
14:45Z; tools `builds/gw1202/stream_*.py`, data cf05 `/var/tmp/kvand-trackbeam/stream/`) settled
the body. The 11-04 "one stream, identical at (epoch, page)" was half right: the content is common
to every sender, but each sender carries its own subset, so same-second messages differ.

- Layout of the 1464-bit second: [49 header][record 1, bits 49-538][record 2, 539-1028][table
  entry, 1029-1308][zeros 1309-1402][record 1's id 1403-1411][1111][0 1111 0 1111 0 1111 0]
  [record 2's id 1432-1440][111 0 1111 0 1111 0 1111 0000]. Header bits 36-48 = BDT week (1087).
- A record = the orbit and clock of ONE GNSS satellite. Offsets (start/width; angles in
  semicircles; `layout.json`): 0/9 id = [3-bit system][6-bit PRN], system 0 BeiDou, 1 GPS, 2
  Galileo; 12/15 toe = BDT second-of-week / 20 (the epoch, exact to < 0.1 s; all 18 records of a
  superframe share it; it steps by 60 s each minute); 29/35 M0 x 2^-34; 64/28 e x 2^-33; 92/34
  sqrtA x 2^-21; 126/35 i0 x 2^-34; 176/35 Omega0 x 2^-34 (BeiDou week convention: the
  -omega_e * toe term takes the BDT second of week); 211/35 omega x 2^-34; 246/20 Cus x 2^-33 rad;
  266/20 signed Cuc x 2^-33; 300/16 BDT second-of-day / 10 of the superframe; 316/30 signed af0
  x 2^-36 s; 350/12 signed af1 x 2^-44; 362-441 per-satellite constants updated every few hours
  (a 6-bit changing field at 364; the rest unassigned: not dn, Omegadot, IDOT, Cic/Cis, Crc/Crs or
  TGD at any 2^k scale against the broadcast); 161-175, 286-299, 442-489 zeros and page flags.
- How it was found: rank correlation of every candidate bit field with the day's broadcast
  ephemeris (BKG `BRDC00WRD_S` day 309, `builds/gw1202/brdc/`) propagated to the record epoch
  (`stream_ref.py`), power-of-two scale fits (`stream_eph.py`), the harmonic terms from the
  residual's sin 2u / cos 2u structure (`stream_harm.py`), positions through the broker's
  `gnss_ephemeris.sat_pos_clk` (`stream_verify.py`).
- Verification against the broadcast ephemerides: at the epoch BeiDou |dp| median 0.5 m (radial
  -0.5, along/cross MAD 0.1 m), Galileo 12.9 m, GPS 11.7 m (a ~10 m cos u cross-track term for
  GPS and Galileo only: open); propagated as a plain Kepler set with the two harmonics and NO
  secular terms, within 1 m (BeiDou) / 12 m (GPS, Galileo) at +-5 min and 11-22 m at +-15 min.
  Clock: record af0 = broadcast clock - 433 ns (BeiDou, MAD 13), - 416 (Galileo, MAD 0.7), - 413
  (GPS, MAD 1.5): one time scale for all three systems.
- Which satellites: the pool satellites above ~22 deg elevation at the sender's ground sub-point
  (entry median 22.8 deg, exit 21.4; a 22 deg threshold gives precision 0.95, recall 0.96); 4-8
  per system, 18 per 9 s superframe, the set changing about every two minutes. The slot order is
  stable while the set is unchanged (92 % of consecutive superframes), newcomers are appended at
  the end, and the list is re-permuted at each 60 s epoch step by a key that is not elevation,
  azimuth, bearing, PRN or system (within a system it follows the ephemeris age). Forecast value:
  within an epoch a sender's next superframe repeats its records verbatim; the first superframe
  after an epoch step carries re-fitted values no receiver can predict.
- The table entry (1029-1308 = [11-bit index][flag][268 bits]) is the GUOWANG ALMANAC (11-05
  evening, `table_umatch.py`, `table_decode*.py`, `almanac_map.json`). Each entry is one GuoWang
  satellite: satellite ID = 2 x index + flag, with index bit 8 (256) the shell bit (1 = 50 deg
  shell, 0 = polar), the same bit as bit 2 of the message header's ID word; 80 of 80 decoded
  senders agree. The 50 deg-shell senders cycle 108 entries (IDs 1-108, nine per plane in the
  system's plane order, one entry per second, 54 s per flag); the polar senders cycle the polar
  shell's 62 (IDs 22-83, including the two silent planes). Entry fields after a 28-bit table header
  (0x3b7afc4 / 0x3b7a7be): @92/30 x 360/2^30 = node minus GMST at the reference epoch (residual
  0.000 deg), @122/30 = omega and @152/30 = M0 (x 360/2^30; their sum, the argument of latitude,
  matches the TLE to 0.003 deg, the two trading off between re-fits), @67/25 = inclination offset
  at 2^-19 semicircles from a per-shell reference, @40/27 holds the eccentricity (~2^-20), @28/12 and
  @182/22 open; @204/64 = [@204/22 tiny signed][@226/17 signed, drag-like][@243/25 signed = Omega-dot
  in 2^-44 semicircles/s: -3.59 deg/day for the 50 deg shell, -0.34 polar, the J2 nodal rate to
  the TLE's precision (11-09, via the slow channel's copy of the table)]. The reference epoch is 00:00:00 BDT of the day; the table is
  re-issued about every 6 h (00:20, 06:25, 12:20 UTC) at the same epoch. It covers 170
  satellites: the nine extra per plane in planes 0, 1, 2, 11 (the inferred H sets among them) are
  not in it. Not a GNSS almanac, not code biases (both tested).
- Propagating the almanac (`builds/gw1202/table_prop2.py`): it is an SGP4-style mean-element set
  (a plain Kepler conversion is 190 km off, exactly as it is for the TLE's own mean elements).
  Build a pseudo-TLE: epoch = toa (header bits 0-14 = BDT day number, 00:00 BDT); inclination =
  shell reference + @67/25 (signed) x 2^-29 semicircles, reference 50.017 deg for the 50 deg
  shell (index bit 8 set) and 86.50 for polar; RAAN = @92/30 x 360/2^30 + GMST(toa); omega =
  @122/30 and M0 = @152/30 (x 360/2^30); e = bits 45-66 x 2^-32 for the polar shell (the 50 deg
  shell's e field is elsewhere; e ~ 1e-4, so 0 costs < 1.5 km); mean motion = the shell's nominal
  Kozai value (the within-shell spread costs < 1 km/day); bstar = 0; run sgp4. Against the real
  TLEs over 15 h: 3D median 0.86 km (50 deg shell) and 0.93 km (polar), i.e. the TLE's own
  precision, so the almanac is a TLE-equivalent orbit source for its 170 satellites with a 6 h
  refresh. Consecutive daily issues give each satellite's own rates of argument of latitude and
  node, agreeing with SGP4 to 0.002 and 0.0002 deg/day.

## What is not in it

- No LEO ephemeris or almanac. No field tracks the sender's orbit (raw angles, ECEF or ECI
  position or velocity: `orbit_field_scan.py`) or its theta (`theta_field_probe.py`, best |rho|
  0.36). The 12-bit words at 1286 and 1052 that looked like satellite IDs are not almanac keys
  (`almanac_probe.py`). Low-rate pages 2-4 are filler; an ephemeris would appear there (~1000 bits
  per 12 s) if it is ever broadcast. Orbits come from TLEs plus the measured theta.

## Predictability

What a symbol wipe can know in advance, per frame (measured on 11-04: the low-rate signs are
predicted at 98.3 % from time and the page templates, `PER_ELEMENT.md` section 5):

- Deterministic from UTC and the satellite identity. Low-rate: subframe (3 s) and symbol (4 ms)
  boundaries, the page (12 s cycle), the SOW field, pages 2-4 entirely, and the encoder from bits
  to signs. High-rate: the second boundary, the sync word, the header constant, SOW and week, and
  the LDPC parity and CRC given the info symbols.
- Learned per satellite, constant over a pass: the page-1 payload (ID, the 10-bit field, the
  per-page word), the high-rate header ID and shell flag, the CSK code (from the ranging code
  through the fixed table), theta.
- Learned per second from another satellite: the body is constellation-wide, so the message at
  (epoch, page) is the same from every satellite; but each satellite's uplink lag (9-115 s)
  selects the epoch and is only known from its own message.
- Unknown: the 12 bits outside the low-rate CRC; the live correction content (6-7 % of block bits
  per 10 s epoch); the CSK polarity (~54 % positive; the patent assigns it to authentication);
  the variant satellites' format.
- Measured 11-06 for the wipe (`builds/gw1202/nav_predict_eval.py` over the 11-05 night's 6680
  CRC-valid subframes, the node's predictor reproduced): on the catalogued satellites the "pages
  2-4 entirely" above does not hold. Every page has ~25 live payload bits per 12 s cycle, so an
  echo of the same page k cycles earlier gets 85 % of symbols right at 12 s, 81 % at 1 min,
  78 % at 2 min, 69 % at 5 min. A 12-s echo still has ~39 wrong bits of 375: 24 in the payload, 12
  in the CRC those scramble, 3 in the trailer. A record's coherent gain 5(2p-1)^2 against
  unpaged is +4.0 dB at 12 s and +2.0 dB at 2 min, and falls below unpaged at ~3.5 min. The table
  therefore ships live pages for 3 min after their transmit time. The H satellites' pages do
  repeat exactly. Counters: page 1 payload bits 26-40 = (BDT SOW / 12 - 1) mod 2^15 on 94 % of
  catalogued page-1 subframes (the H satellites send a constant there, H24-4 22407); page 2 bits
  47-52 step every 10 s at a per-satellite phase (91 %). Advancing both adds only ~0.7 points
  (the node advances page 1's, kv/gw-nav-counter). Trailer bits 0-5 vary with no repeat, bits
  6-11 are nearly always 0.
- Readability off boresight (`~/gnss/bb1030/csk_bb_test.py`, 10-30 baseband): with per-channel
  spatial prewhitening, DIGUI-94 (43 deg off) and DIGUI-112 (70 deg off) read 419/419 frames
  with 0 symbol errors, ~35 dB per symbol, while DIGUI-26 railed the quantizer; without
  prewhitening the readout fails. DIGUI-125 at 0.6 deg elevation: not detected.

### The live page chain

How a decoded page reaches the trackers' wipe, with the latencies measured on the night of
11-05/06 before the change (page's last symbol on air to its use on the node):

| stage | mechanism | latency |
|---|---|---|
| node file | `rawFileWrite`, 240 frames = 10.07 s per file; the loop pulls a file once the next one has opened | mean +5 s |
| pull | `track_beam_loop.py` on cf05 rsyncs the 11 GPUs' files every `--period` (60 s) | mean +30 s |
| decode | 7-9 s per (10-s block, PRN), 12 workers, depth 4; not CPU-bound (load 5 of 64 cores) | +8-15 s |
| measured, loop | decode stamp minus block end, 185 blocks | p10 80, p50 101, p90 129 s |
| table | `gw_ephem_table.py` cron every 2 min reads `nav_pages_live.json` (180-s cap) | mean +60 s |
| broker pickup | table mtime polled every 2-s cycle | +2 s |
| broker to seed | pages attached at a seed's slew or re-birth (`gw_nav_fields`): every 2 s for a locked seed, every `--dr-repin-s` (10 s) otherwise (SEEDAUDIT dt) | +1-5 s |
| push | seeds posted every cycle; the node applies them on the next frame | +2 s |

Two polling periods stacked on a 10-s file: 2.5-3.5 min at use, past the template's
break-even, and under the 180-s cap 2 of 390 table records carried pages. The 11-06 changes:

- the loop pulls every 10 s and, after each page write, rsyncs the file to the VM beside the
  table (`/var/tmp/gnss-guowang/nav_pages_live.json`): the broker then polls a local file, where
  an `os.stat` on the NFS home can lag a write by the attribute cache's 60 s. First hour: loop
  lag after block end p50 34 s, a page decoded 24 s after its last symbol.
- the broker (kv/gw-pages-live, `gnss_broker/deadreckon.py`, live 01:13:32Z) polls that file's
  mtime every cycle, so the table cron drops out of the chain; `dr_nav_pages` also refreshes
  seeds that are held between re-births. Pages are looked up by the covering table record's
  satellite name, never from another record of the same PRN (that was a code-mate's pages), and
  aged by their transmit time (`--gw-live-pages-max-age-s`, 180 s); the table's own pages are
  the fallback.

Measured at 01:18Z: pages 33-42 s old at use (the 24-60 s rows of the echo table), and the
loop's own check of the template one block old against the next decoded periods: 0.86-0.89 of
symbols right. The 10-s node file is what separates that from the 12-s row:
`num_frames_per_file` 240 -> 48 and a loop that decodes a period as soon as its frames are in.

Coverage, not latency, is now the limit. The track beam steers at the two strongest gw_1202
satellites (`gnss_fleet_chord.yaml`), so pages exist for at most two of the ~10 seeded ones;
at 01:18Z 2 of 11 covered PRNs carried live pages. One satellite's pages cannot stand in for
another's: DIGUI-126 and DIGUI-57, decoded at the same SOW, differ in 74 (page 2) and 102
(page 3) of 316 payload bits. A wipe on every tracked satellite needs a track beam and a decode
per satellite.

## Literature consulted

Copies in `itu_refs/` (see `SIGNAL.md` for the signal-level references).

- ITU-R M.1787-6 Annex 17: XW-B2b, 125 / 1500 bit/s. CSNC 2024 Yan et al.: the MCSK(6,2,1)
  example. The full 630-page proceedings (`csnc2024_full.txt`) have no SATNET message-format
  chapter; a LEO ephemeris appears only as "16 BDS parameters + 6" (ref. [12] of the
  single-satellite-positioning chapter).
- Patents: CN119788292A/B (authentication by CSK polarity + chip replacement); CN115685267A
  (Qixin receiver: high-rate 1 s subframes LDPC + CRC, low-rate convolutional + CRC);
  CN110244328B (MCSK multiplexing, 125 bit/s low / 2.25 kbit/s high examples). Cited paper:
  "一种低轨星座导航增强信号的电文快速同步及解码方法".
- Pocket SDR `sdr_ldpc.c` for the BeiDou B1C / B2a GF(64) LDPC H tables.
- A first hypothesis, 486 symbols = 3 x BeiDou LDPC(162,81) (PPP-B2b), was replaced by the
  (200,100) x 2 + (96,48) layout.

## Pointers

Under `/home/kvand/gnss/builds/gw1202/` unless noted.

- Low-rate: `slow_bits.py` (Viterbi, CRC-24Q), `slow_bits2.py` (signs), `slow_code_test.py`
  (generator search), `slow_decode.py` (subframe phase, CRC scan, pages), `parse_slow.py` (fields).
- High-rate: `csk_extract.py` (streams and messages; the cf05 loop runs it after `gw_reduce.py`),
  `csk_fold.py`, `csk_ldpc_probe.py`, `gw_ldpc.py`, `msg_study.py`, `msg_dataset.py`,
  `msgs_all.npz`, `csk_streams/` (`csk_codes.json`, `msgs/`, `bit_change_profile.npz`).
- Negative searches: `almanac_probe.py`, `orbit_field_scan.py`, `theta_field_probe.py`.
- Off-boresight readout: `~/gnss/bb1030/csk_bb_test.py`. References: `itu_refs/README.md`.

Sources: chord-guowang-payload-20261104.md, chord-guowang-1202-is-mcsk.md,
chord-guowang-itu-satnet-leo-filing.md, chord-guowang-encoding-20261031.md,
chord-guowang-tracking-design-20261102.md, chord-guowang-csk-lane-plan-20261104.md (archive
key), builds/gw1202/parse_slow.py (subframe arithmetic).
