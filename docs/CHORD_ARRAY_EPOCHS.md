# Which physical feed an element was — and what may be co-added

**The problem.** `element 9` is not a thing. It is a slot, and what sits in that slot changes:
the site team recables a dish, the dishes get re-pointed along the N–S axis every few months,
the live element set grows as dishes are populated. Every one of those makes a sum across the
boundary a sum of two different instruments — and it shows up not as an error but as a slightly
wrong answer nobody can trace back.

**The hook.** [`python/scripts/gnss/gnss_arraymap.py`](../python/scripts/gnss/gnss_arraymap.py)
resolves *time → array configuration*, and refuses when the answer is not a single one. The
table it reads today is [`config/chord_array_epochs.json`](../config/chord_array_epochs.json);
the intended source is the array database, which is why everything goes through one `Backend`
class with one method.

```sh
P=/home/kvand/gnss/venv/bin/python
$P python/scripts/gnss/gnss_arraymap.py epochs                 # the intervals
$P python/scripts/gnss/gnss_arraymap.py show   --at 20260908   # the whole array that day
$P python/scripts/gnss/gnss_arraymap.py element 9 --at 20260908
$P python/scripts/gnss/gnss_arraymap.py diff   20260908 20260916
$P python/scripts/gnss/gnss_arraymap.py masters fixtures/beamcube/cube_2026090*_nside64.npz
$P python/scripts/gnss/gnss_arraymap.py check                  # validate the table
```

## 1. An epoch is an identity interval, not a health interval

A new epoch starts when **what an element IS** changes: a recabling, a re-pointing, a change to
`array.live_element_ranges`. An element going dark does **not** start one — health is a property
of the data, and making it an epoch boundary would fragment the table into uselessness while
telling you nothing you cannot measure from the cube itself.

## 2. The resolver never extrapolates

A time outside every recorded interval raises `NoEpoch`. That is the whole value of the thing.
Two cases matter:

* **Before the first epoch.** `valid_from` is the date of the first *written* mapping, not a
  claim that nothing changed earlier. Data older than that raises, and the fix is to record the
  interval, not to widen the first one.
* **Inside a gap.** The two epochs today are separated by a deliberate one-day gap: the
  recabling happened during 2026-09-15 and nobody wrote down the hour, so any data from that day
  resolves to nothing. GNSS was off, so nothing is lost — but had it been on, refusing is still
  the correct answer.

`Straddle` is the other refusal: the interval is covered, but by more than one configuration.
Both subclass `Refused`, and **callers should catch the base** — which one fires depends on
where the unrecorded gaps happen to fall, so catching only one is a bug waiting for a boundary
to move.

## 3. Two numberings, and they are different axes

| axis | width | formula | used by |
|---|---|---|---|
| correlator element | 128 | `pol*64 + dish` | the fleet, the position survey, this table |
| beam-cube element | 32 | `pol*16 + dish` | `gnss_beam_cube.py`, `gnss_beam_static.py`, the viewer |

Cube index `i` is correlator element `cube_order[i]`, and `cube_order` is **per epoch** because
the live set changes. So cube 25 is correlator 73, not 25. `element()` takes `index=` or `cube=`
and refuses both or neither; there is no arithmetic anywhere else, on purpose.

## 4. Feed planes are labelled A/B, and that is not X/Y

The plane assignment comes from a *relative* measurement — a beam-contrast clustering that says
ten dishes put one plane in slot 0 and three put the other. It does not say which plane is X.
**Never promote A/B to X/Y without an absolute reference.** Six dishes were never classified at
all (three dark LNAs, three with one pol degraded) and one is ambiguous; those carry
`plane: null`, which is a real state — `Element.known` is the test, not `plane is not None`.

## 5. Stamping artifacts

`gnss_beam_cube.py build` now resolves the epoch **before doing any work**, refuses a day it
cannot place or that straddles a boundary, cross-checks the pointing the archive claims against
the pointing the epoch records, and writes into every master:

```json
"array_epoch": "2026-08-29-baseline",
"array_epoch_key": "2026-08-29-baseline:471481b1"
```

The key is the name **plus a digest of what the table says**. If the table is later corrected in
place — a plane pinned, a position re-measured — artifacts built against the old content stop
matching rather than silently claiming agreement.

Masters built before this existed carry no stamp. `assert_compatible` places those by their
`day` and **says so in a note**; it does not treat unstamped as compatible-by-default.

## 6. Co-add guards: what is wired and what is not

| where | status |
|---|---|
| `gnss_beam_cube.py build` | ✅ refuses unplaceable/straddling days, stamps the master |
| `gnss_beam_cube.py export` | ✅ carries the stamp into the per-day manifest |
| `gnss_arraymap.py masters` | ✅ standalone preflight for any set of masters |
| `gnss_beam_static.py` | ⚠️ **not wired** — patch below |

`gnss_beam_static.py` already refuses to co-add masters of differing `pointing` or `units`
(`prepare()`), which is the same shape of guard; it just does not know about epochs yet. It is
being edited in another session, so the two-line change is written here rather than applied:

```python
# at the imports
import gnss_arraymap                                    # noqa: E402

# in prepare(), beside the existing units/pointing check
epoch, notes = gnss_arraymap.assert_compatible(masters)  # raises Refused on a mixed sum
for n in notes:
    print("  note: %s" % n)
```

`load_master` already keeps `path` and `day`, and passes the whole meta through, so nothing else
needs to move. With that in, a `map` summing across a recabling fails loudly instead of
producing a plausible picture of two arrays at once.

## 7. Pointing

`pointings` carries the boresight each named pointing implies. Today there is one,
`p0_dec40p73` → az 180.0, el 81.41 (dec +40.73, dishes 8.59° south of zenith).

⚠️ `BORE_AZ, BORE_EL` in `gnss_beam_cube.py` are still module-level constants for that one
pointing, and `gnss_beam_static.py` imports them from there. **Until they are read from the
epoch, a master built at a new pointing would be mapped around the wrong boresight** — the
epoch guard stops the two being *summed*, but not a single new-pointing master being drawn
wrong. That is the next change to make, and it wants doing before the next re-point, not after.
`telescope.dish_coelev_deg` reads −27.3 and does **not** give the boresight; do not reach for it.

## 8. When something changes on site

1. Add an interval to `config/chord_array_epochs.json`. Close the previous one. If the hour of
   the change is unknown, **leave a gap** rather than guessing a boundary.
2. Set `verified: false` and say in `provenance` who reported it. It stays false until measured
   on sky — `assert_compatible` surfaces that as a note on every co-add.
3. `gnss_arraymap.py check` — it validates overlaps, the `pol*64+dish` invariant, that no dish
   has one plane in both slots, and that the open epoch's `cube_order` matches
   `array.live_element_ranges` in the shipped node config.
4. `python -m unittest python.scripts.gnss.test_arraymap`.
5. Rebuild nothing. Old masters stay valid for their own epoch; that is the point of the stamp.

### Verifying the 2026-09-15 swap

The current open epoch is **unverified**: the array was handed over before any post-swap beam
data existed. To close it out, build one post-swap day and re-run the contrast split — A06, A07
and B07 should have joined the majority cluster. Whether the site team also touched A05, B05 and
B08 (flagged UNTESTED when the three were reported) is unknown; they carry `plane: null` either
way, so the table is not wrong if they did, only less complete.
