# BeiDou-3S M2-S (C58) codes recovered from the sky

## Source

- **Capture:** a 75 s burst of raw bladeRF frames at gx10, 2026-11-28 06:45:09Z, with M2-S at 79 deg elevation.
- **Fill:** about half the 16 ms frames were captured; the burst has gaps.
- **Code clock:** the B3I PRN 58 track supplies it.
- **Recovery:** each band is folded at 1 ms, and the chips are the principal eigenvector of the folded periods. This is free of the data sign and carrier phase per period.
- **Generators:** taken from Berlekamp-Massey on the chips.
- **Independent check:** 1 s bursts at 05:00, 06:00, 07:00 and 08:00Z correlate with the generated codes at the phase the B3I track predicts.

## Files

All files are int8, +1/-1, one value per chip or per overlay chip.

| File | Contents |
|---|---|
| `b2a_sky.i8` | B2a (1176.45 MHz), 10230 chips. Data on even chips, pilot on odd, from the B3I code epoch. |
| `b2b_sky.i8` | B2b (1207.14 MHz), the same layout. |
| `b3i_sky.i8` | B3I PRN 58, 10230 chips. Equals `generate_b3i_code(58, false)`: G1 is not truncated. |
| `b2a_pilot_secondary_sky.i8` | 100-chip B2a pilot overlay, one chip per ms. Equals the ICD B2a pilot secondary of PRN 58, cyclically shifted. |
| `b2b_pilot_overlay_sky.i8` | 1800-symbol B2b pilot overlay, one value per 10 ms symbol (18 s). 0 marks the 46 positions the burst did not cover. |

The sign of each component is arbitrary, as is the cyclic phase of each overlay file. The overlay phases count from the burst's first full code period.

## Structure

- **Interleaving:** each band chip-interleaves its data and pilot components on one carrier phase.
- **Component codes:** each component is a 5115-chip code from two 13-stage LFSRs. The generator and start states are in `lib/stages/gnss/beidou3sCode.cpp`.

**Overlays:**
- B2a data: the 5-chip secondary 00010, 5 ms symbols.
- B2a pilot: the ICD pilot secondary of PRN 58.
- B2b data: the 10-chip secondary 0001110110, 10 ms symbols.
- B2b pilot: the same 10-chip secondary times the 1800-symbol overlay above.
