# GuoWang 1202.025 MHz: the message content

Status: 2026-11-05 (the high-rate message decoded as a per-GNSS-satellite ephemeris from the
overnight track-beam stream; the 11-04 sections below were decoded during the 11-04 site outage
from stored gw_reduce products, no raw baseband). The signal is in `SIGNAL.md`. Products: cf05 `/var/tmp/kvand-gw1202/reduce/wNNN/<sat>.npz`
(per-frame `car_ph`, `data_bits`, `code_err`, `theta`, `n` = UTC frame number); scripts under
`/home/kvand/gnss/builds/gw1202/`.

Two channels share every 2 ms frame: the sign of the ranging half (low rate) and the cyclic shift
of the CSK half (high rate). ITU-R M.1787-6 files them as 125 / 1500 bit/s; measured 125 and
1488 (1464 data + 24 CRC).

## 2026-11-10 review: corrections

A pass over every open item against the five-day archive (285 822 CRC-valid high-rate messages,
134 senders; 54 355 low-rate subframes; probes in `builds/gw1202/review_20261110/` and
`review_20261110b/`). Where this section and the text below disagree, this section holds.

- **The low-rate "5-minute record" is the sender's own ephemeris, not a MAC or an encrypted
  block.** Page 3 payload bits 166.. are Omega0 in the BeiDou convention, RAAN(t) - GMST(start of
  the BDT week): against the TLE the median is -0.0002 deg and the MAD 0.0011 deg (10 316
  subframes, 118 senders). The 11-09 scans compared with RAAN - GMST(now), where the MAD is 74 deg.
  Page 3 bits 130.. are M0 and page 4 bits 92.. are omega; M0 + omega equals the TLE's argument
  of latitude at the BDT 5-minute boundary to 0.07 s (3118 of 3333 cycles; 169 carry the previous
  boundary's record). So toe is the 5-minute boundary.
- **The sender's ephemeris, decoded (`review_20261110b/`; user algorithm `gw_eph_user.py`; 1029
  issue records, 103 senders).** A BeiDou-style 16-parameter ephemeris plus four third-order
  harmonic terms. The receiver adds nothing. With t the time from toe:

      A = Re + h + dA                    Re = 6 378 137 m; h = 1150 km (50 deg shell), 1175 km (polar)
      n = sqrt(mu / A^3) + dn            M = M0 + n t; Kepler gives E, nu; phi = nu + omega
      u = phi + Cus sin 2phi + Cuc cos 2phi
      r = A (1 - e cos E) + Crs sin 2phi + Crc cos 2phi + Crs3 sin 3phi + Crc3 cos 3phi
      i = i_ref + di + Cis sin 2phi + Cis3 sin 3phi + Cic3 cos 3phi      i_ref = 50 deg, 86.5 deg
      Omega = Omega0 + Omega-dot t       (Omega0 = RAAN - GMST at the start of the BDT week)

  | Word | Page, first bit / width | LSB |
  |---|---|---|
  | dA | 2, 83/27 signed | 2^-10 m |
  | di | 2, 110/31 signed | 2^-35 semicircle |
  | Omega-dot | 2, 163/26 signed | 2^-45 semicircle/s |
  | Cis | 2, 189/21 signed | 2^-34 semicircle |
  | dn | 3, 80/27 signed | 2^-45 semicircle/s |
  | M0, Omega0 | 3, 130/36 and 166/36 | 2^-36 circle |
  | e | 3, 202/30 | 2^-35 |
  | omega | 4, 92/36 | 2^-36 circle |
  | Crs, Crc, Crs3, Crc3 | 4, 128, 154, 180, 206 / 26 signed | 2^-11 m |
  | Cus, Cuc | 1, 140 and 165 / 25 signed | 2^-34 semicircle |
  | Cis3, Cic3 | 1, 190 and 211 / 21 signed | 2^-34 semicircle |

  Zero in every record: page 2 141/22 (by position the inclination rate; it reads +-1 in 6
  records), page 2 210/21 (the place of Cic) and page 3 107/23 (a rate after dn). The shell is
  the page-1 key word's bit 2 (1 = 50 deg).
- Checks (338 pairs of consecutive issues; 312 in the 50 deg shell, 26 polar). Issue k
  evaluated 200 to 600 s after its toe agrees with issue k+1 to about 1 cm in radius, 1 cm along
  track and 0.4 cm across track (3-D median 1.2 cm, 95 % below 5 cm). The fit arc is
  therefore toe - 100 s to toe + 600 s; outside it the two differ by 0.7 m at toe, 10 m at
  toe - 300 s and 2.3 m at toe + 900 s. Every scale was fitted freely first: dn 2^-45.000, dA
  2^-10.000, Cus and Cuc 0.99998 and 0.99999 of 2^-34, the inclination terms 1.0001, 0.9999,
  0.9999, the radial harmonics 0.999 of 2^-11 m. The harmonic argument is nu + omega (with E
  or M the mismatch is 4 to 8 times larger). Fitted from the along-track rate with mu =
  3.986004418e14, the reference radius is Re + h to 1.2 cm (50 deg shell) and 1.3 cm (polar),
  +-0.4 cm.
- Against SGP4 at mid-arc: radius -4 m (scatter 77 m), along track -45 m (485 m), across track
  +13 m (87 m) in the 50 deg shell; +52 (119), -616 (741), -141 (70) in the polar shell. These
  are the TLEs' errors. The mean Crc is 866 m and 1482 m (J2: 858 and 1453) and the mean Cus
  5.65e-5 and 9.33e-5 rad (J2 in the mean plane: 5.70e-5 and 9.63e-5).
- Against CHORD's own code timing (`review_20261110b/cmp5.py`). A beam-dump theta is the sender's
  clock offset plus the error of the range it was reduced with, so swapping the TLE range for the
  broadcast one grades both. Over 23 beam-dump detections inside an issue's arc (23 satellites,
  11-05 to 11-10), theta plus the page-1 clock bias agrees across satellites to 34 ns (MAD) with
  the broadcast range and 173 ns with the TLE range; its median, -10.42 us, is our receiver
  constant. Within a window the broadcast removes the TLE's curvature (DIGUI-36: 29 ns to 2.5 ns
  rms about a line) and leaves our chunk noise of 2 to 5 ns. The TLE range differs from the
  broadcast by 90 m (median) and up to 530 m at these windows. The track beam's own timing
  follows the broadcast range with slope 1.03, but at about 0.3 us per block it is too coarse to
  grade it.
- One issue carried past its arc: about 2 m at t_oe + 15 min and 16 m at + 20 min. It is a
  now-cast; the 6-hour almanac (TLE grade, about 0.3 km along track) remains the forecast.
- The elements are re-fitted for every issue over that short arc, so dA (+-0.6 km), e (+-2e-4),
  omega (sd 7 deg) and the harmonics wander from issue to issue while the position holds. CHORD
  sees the 50 deg shell near phi = 90 deg and the polar shell near 49 and 131 deg, where the
  2phi and 3phi terms look alike. A model without the third harmonics therefore returns
  plausible but wrong scales for the rest (1.111 mm for dA, 0.295 mm for Crs and Crc, apparent
  J2 terms "added by the receiver"). Earlier revisions of this section carried those.
- Page 1 bits 100-117 are an issue tag in 3 s units (boundary SOW / 3 + 1); bits 91-99 read 34 in
  both weeks. Page 1 bits 138-139 are two flags. The regions called constant below (page 1
  91-139, page 3 0-79, page 4 0-91) change per issue or per pass and belong to the same data
  set.
- The page-2 time stamp is 3606-3607 s behind the subframe start, not 3600 s, and is sometimes
  10 or 60 s stale: a data time tag. The page-2 count N is not the number of GNSS satellites the
  sender lists (Pearson -0.34).
- **GNSS records use BeiDou's Earth rotation rate (7.292115e-5 rad/s) for every system.** The
  ~10 m cos u cross-track term on GPS and Galileo came from evaluating them with the GPS value.
  With BeiDou's value: Galileo 0.37 m, GPS 0.79 m, BeiDou 0.54 m median at the epoch.
- Record bits 346-349 are the sign extension of af1 (16 bits at 346-361). Bits 364-369 are an
  accuracy index in the SSR class/value form (median 80 mm), refreshed every 2 h at about 35 min
  past odd UTC hours. Bits 370-437 are a count (4) and four [4-bit signal][12-bit code bias x
  0.017 m] entries, the PPP-B2b form: BeiDou signals 2, 5, 0, 12; GPS 0, 7, 8, 12; Galileo 2, 4,
  8, 11. BeiDou b[0] - b[12] against the broadcast TGD1: r 0.999, slope 1.002. Galileo b[2] - b[4]
  against BGD(E1,E5a): r -0.997, slope -0.785 (theory -0.793). Bits 438-489 are a second count
  (4) and four 12-bit entries, 0x300 for GPS and Galileo and 0 for BeiDou: open, with bits
  362-363 (01).
- Message bits 1412-1431 and 1441-1460 are the fixed pattern 11110 x 4, one group per signal.
- Almanac @204/22 is the rate of the e sin(omega) word, 2^-35 per second: Pearson 0.966 with its
  day-to-day change over 552 satellite-day pairs in the 50 deg shell. In the polar shell it is
  noise.

## The low-rate BPSK channel

- Symbols: one sign per frame pair, pairs starting on even UTC frame numbers: 4 ms, 250 sym/s.
  Read from `car_ph` frame to frame with the 10 ms carrier pattern modelled per class n mod 5 (a
  smoothed absolute reference gives 10 % sign errors); `slow_bits2.py` pair agreement 0.9998.
- Code: rate-1/2, K = 7 convolutional, G1 = 171, G2 = 133 (octal, GPS convention, G1 first),
  found by an exhaustive parity-check search (`slow_code_test.py`: 97-98 % of checks, the only
  pair). Hard-decision Viterbi in `slow_bits.py`. Odd-weight generators leave the bit polarity
  ambiguous per gap segment: both are tried per subframe. Register convention that reproduces
  the transmitted symbols (11-09, `pol_check.py`, 99 % agreement with the measured signs at
  offset 0): the newest bit enters at the LSB, i.e. taps 0o117 / 0o155 in `nav_predict_eval.py`'s
  encode() form (its own 0o171 / 0o133 is the mirror; harmless there, it encodes both sides).
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
  satellite at once, and two modular ramps (page 1 @59/10 and @77/14). On four days of data
  (`slow_all.npz`, 41 668 subframes, 126 senders) the 5-minute record (~300 bits per sender over
  the four pages) is random-looking: every bit's mean is 0.5, consecutive values differ in half
  their bits, and no window follows the sender's orbit at the subframe time or at either 5-min
  boundary, nor copies its almanac entry: a per-satellite signature/MAC-like block (page 1 bits
  118-138 beside it is a shared, rarely changing 20-bit word). 11-09 (`rec_shared.py`): two
  senders decoded in the same BDT 5-minute period agree on 0.53 of the record's bits (random; the
  bits the fleet shares are only the counters, the time stamp and the constants), so the record
  holds no fleet-wide block: a TESLA-style disclosed key would be common to every sender. It is a
  per-satellite MAC/signature, or an encrypted per-satellite block (the patent says the low-rate
  message carries the sender's own ephemeris and clock, which appear nowhere in clear). Neither
  can be checked or predicted without the operator's key material. The two ramps are the sender's
  CLOCK (`ramp_dig4-6.py`): page-1 payload bits 42-68 are one signed 27-bit clock bias in
  2^-36 s (14.55 ps per LSB, +-0.98 ms; the "own theta" field of 11-05 is its top 10 bits, the
  @59/10 ramp its bottom 10, and the value is continuous across the ramp's wraps: 0 jumps in
  5832 consecutive cycles). Its drift over four days against the theta drift the track-beam
  loop measures per satellite: slope -0.954, Spearman -0.86 (85 satellites); theta_lib + bias
  = -10.0 us (MAD 1.5 us over 98 satellites), the -10 us being our receiver and geometry
  constant. Bits 69-90 are the signed clock DRIFT: the per-satellite median against the
  measured bias drift gives Pearson 0.90 with 3.45e-13 s/s per LSB at bit 81 = 2^-25 s/day
  (29.8 ns/day); the low bits (the old "@77/14 ramp") are its estimation noise (~5 LSB at bit
  81 within a pass). Neither ramp is an inter-satellite range (best of 170 satellites R^2 0.64
  = chance, no repeating partner) nor a range to any fixed ground point (grid best 0.46). No IEEE floats anywhere; no correlation with the sender's geometry.
- **11-09, the page-2 head (`ramp_dig7-14.py`, `slow_all.npz`):** bits 0-13 are a header
  (00001000000000 on 91 %, bit 2 set in bursts on some senders; the H satellites send
  01010011010100). Bits 37-52 are a 16-bit time of week in 10-s units = floor((BDT SOW - 3600 s)
  / 10) to within our 5-s time tag, wrapping at 60480 (the "6-bit counter at 47-52" was its low
  bits), and bits 53-65 are the BDT week (1087 -> 1088 when that counter wraps, an hour after the
  BDT rollover): a time stamp one hour behind BDT. Bits 32-36 and 66-88 are constants. Bits 14-31
  are an 18-bit STATE FIELD with two layouts by satellite group. Group B (97 senders): bits 14-18
  a 5-bit value N (8-19, stepping +-1 every 2-5 min against the ramp's direction in 94 % of
  steps) and bits 19-31 a signed 13-bit x (|x| median 167, 95 % below 500) that ramps linearly
  within a pass (|slope| median 0.27 LSB/s, max 1.3, either sign, through zero, carries intact).
  Group A (27 senders: DIGUI-96-98, 100-104, 119-126, 155-163, 40, 58): bits 14-24 = the BDT week
  and bits 25-31 the same kind of ramp mod 128 (the old "ramp C" at 24-31 straddled the two).
  The field was all zero on every catalogued satellite from between 11-05 14:45Z and 22:00Z
  until 11-07 07:30-08:00Z, and active before and after (CRC-valid subframes; the clock bias
  kept ramping meanwhile, so the zeros are the satellites'). Rejected for x: range or range rate
  to any catalogued satellite (zero crossings vs range extrema no better than chance, no fixed
  power-of-two scale, no plane relation), range to CHORD or any ground point, the sender's clock
  bias/drift, latitude, sun beta angle, orbit-noon phase, eclipse; it is not orbit-periodic
  (passes an orbit apart differ by hundreds). A satellite-internal state (attitude-control-like)
  is the best guess. The page-1 shared word (bits 118-137) toggles between 0 and 0xfffff per
  sender, independently of that idle window.

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
  shell's 62 (IDs 22-83, including the two silent planes). Entry fields after a 26-bit table header
  (bits 0-14 the BDT day, 15-25 a per-shell constant, 1009 / 495), from the 11-09 field scans
  (`alm_words.py`, `alm_words2.py`, 168 satellites against the cached TLEs): @26/12 unsigned = the
  semi-major axis as an offset from a per-shell reference (~7529 km for the 50 deg shell, ~7547 km
  polar; 1 LSB ~ 6 cm; Pearson 0.93 / 0.98 vs the TLE's a); @40/27 x 2^-32 = e sin(omega), the
  eccentricity vector's y-component (polar: the TLE's mean value, offset 1.6e-5, Pearson 0.999;
  50 deg: the TLE value + 7.63e-4, which is the J3 frozen eccentricity -J3/(2 J2) (Re/a) sin i =
  7.59e-4, so the 50 deg entries keep the long-period J3 term and their omega word sits at 90.4 +-
  3 deg for the whole shell; the polar "@45/22 = e" of 11-05 was the low 22 bits of this word);
  @67/25 = inclination offset at 2^-19 semicircles from a per-shell reference; @92/30 x 360/2^30 =
  node minus GMST at the reference epoch (residual 0.000 deg); @122/30 = omega and @152/30 = M0
  (x 360/2^30; their sum, the argument of latitude, matches the TLE to 0.003 deg, the two trading
  off between re-fits); @182/22 signed = the J2 short-period argument-of-latitude coefficient
  (J2 Re^2 / 8 p^2) (7 cos^2 i - 1), the sin 2u term SGP4 applies: both shells on one scale,
  Pearson 1.0000, intercept -87 LSB, residual 0.13 %; 1 LSB = 2.73e-9 rad = 2.06 cm along track
  (2^-30 semicircles would be 7 % smaller); re-fitted daily, 0.1-0.3 % scatter per satellite;
  @204/22 signed OPEN: 0 +- 1 on every 50 deg plane except plane 5 (DIGUI-49..57, e = 1e-3, omega
  ~ 160 deg: -10..-13), 0..13 in the polar shell with no e, omega or plane relation; @226/17
  signed = a fitted along-track (drag-type) parameter, UNIT NOT PINNED (`alm_words3-4.py`): in
  the 50 deg shell it follows the TLE's own fitted drag (B*, ndot: Spearman -0.6..-0.8 on every
  issue 11-04..11-08) but not the physical decay (a 5-day fit of the TLE semi-major axis gives
  -0.25 m/day fleet-wide, +-0.15 between satellites, Spearman < 0.5 with the word); in the polar
  shell it follows nothing but the operator's own day-to-day a-word change (Spearman -0.7,
  anti-correlated: the estimator trades a against the along-track term). Sign: positive = orbit
  rising (-ndot). Slope bracket vs the TLE's ndot: 1 LSB in 3.6e-10..2e-9 rev/day^2 (2^-31..
  2^-29), or 1-7 mm/day in da/dt. A lower shell, where drag dominates, would calibrate it; @243/25 signed = Omega-dot in 2^-44 semicircles/s: -3.59
  deg/day for the 50 deg shell, -0.34 polar, the J2 nodal rate to the TLE's precision (11-09, via
  the slow channel's copy of the table). The reference epoch is 00:00:00 BDT of the day; the table is
  re-issued about every 6 h (00:20, 06:25, 12:20 UTC) at the same epoch. It covers 170
  satellites: the nine extra per plane in planes 0, 1, 2, 11 (the inferred H sets among them) are
  not in it. Not a GNSS almanac, not code biases (both tested).
- Propagating the almanac (`builds/gw1202/table_prop2.py`): it is an SGP4-style mean-element set
  (a plain Kepler conversion is 190 km off, exactly as it is for the TLE's own mean elements).
  Build a pseudo-TLE: epoch = toa (header bits 0-14 = BDT day number, 00:00 BDT); inclination =
  shell reference + @67/25 (signed) x 2^-29 semicircles, reference 50.017 deg for the 50 deg
  shell (index bit 8 set) and 86.50 for polar; RAAN = @92/30 x 360/2^30 + GMST(toa); omega =
  @122/30 and M0 = @152/30 (x 360/2^30); e = @40/27 x 2^-32 for the polar shell (for the 50 deg
  shell that word carries the J3 frozen offset 7.6e-4 that SGP4 adds itself: subtract it, or use
  e = 0, which costs < 1.5 km); mean motion = the shell's nominal
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
  per 10 s epoch); the variant satellites'
  format; the almanac's @204/22 and the units of @182 and @226; the slow channel's 5-minute
  record (per sender, no fleet-wide part).
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
- CSK POLARITY (11-09, `pol_abs2-4.py`, cf05 syms archive 11-08/09, 1465 files, 5.2 M frames,
  ~120 satellites): the loop reads the CSK sign relative to the same frame's ranging half, so the
  old "~54 % positive" was the ranging half's data sign imprinted on the raw readout. Multiplied
  by the ranging half's own sign, re-encoded from the CRC-valid slow pages, the CSK polarity is
  CONSTANT: 0.917 of all frames, 0.991 of frames with peak/second-peak > 2, 99 % of seconds, one
  flip per 1000 decided seconds. The residual 1 % sits in the header slots 0-42 after the sync
  word (small-k symbols, half of them with a k different from the slot's usual symbol), at the
  same 0.9 % rate on 57 satellites, with no UTC-second structure, normal ranging amplitude and
  carrier-phase step: readout noise. So the CSK code period carries a fixed polarity: the
  patent's polarity channel is unused, like its chip replacement.
- Patents: CN119788292A/B (China SatNet, filed 2025-03-04, granted 2025-07-11; read 11-09): an
  authentication message once per period T (10 s in its example), carried by the CSK code-period
  POLARITY, holds an ECDSA signature (r, s) over the 256-bit hash of the low-rate message's time
  tag, ephemeris and clock plus part of the high-rate message, and the positions of "encryption
  code" chips that replace the last 200 chips of 30 CSK periods in the last second of each period
  (the hash's first 32 bits seed the generator; we see no replaced chips). The public key comes
  from a CA, the factory, a network, or the authentication frames themselves. No TESLA, no key
  chain, no hash named, no bit lengths, and no authentication block in the low-rate message, which
  it says carries the satellite's time tag, ephemeris, clock, status, almanac and ionosphere. The
  CSNC 2024 BDSBAS chapter (Chen et al.) is the family template: an SM3 key chain, SM3-HMAC MACs
  truncated to 30 bits (BigMAC) or 16 bits (LittleMAC), 115- or 128-bit keys disclosed one 6-s
  epoch late, an SM2 signature on the root key. CN115685267A
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
