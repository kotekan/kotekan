# GuoWang 1202.025 MHz: the message content

Status: 2026-11-04 (decoded during the 11-04 site outage from stored gw_reduce products, no raw
baseband). The signal is in `SIGNAL.md`. Products: cf05 `/var/tmp/kvand-gw1202/reduce/wNNN/<sat>.npz`
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
- Interpretation: live GNSS augmentation corrections (PPP-B2b-like) at ~10 s epochs, the
  "augmentation" product of the filing. Decoding them as corrections is a separate project.

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
- Readability off boresight (`~/gnss/bb1030/csk_bb_test.py`, 10-30 baseband): with per-channel
  spatial prewhitening, DIGUI-94 (43 deg off) and DIGUI-112 (70 deg off) read 419/419 frames
  with 0 symbol errors, ~35 dB per symbol, while DIGUI-26 railed the quantizer; without
  prewhitening the readout fails. DIGUI-125 at 0.6 deg elevation: not detected.

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
