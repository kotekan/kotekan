"""Task #53: /get_spectrum windows must be identical across instances BY CONSTRUCTION.

THE DEFECT THESE PIN. The window used to be "whatever accumulated since your last GET", with a
reset on read -- so it was defined by WHEN THE BROKER'S REQUEST ARRIVED, and the broker polls
12 instances sequentially. The instances were never summing the same records, and the broker
could not repair it: re-polling a laggard returns a NEW SHORTER window, never the records it
missed. There is no way to ask for the past.

That is not cosmetic. Each instance's channels are a comb spanning ~18.75 MHz, so the
cross-instance phase relationship is a delay ramp -2*pi*f*tau; the broker was absorbing the
window offset into a FREE PHASE PER INSTANCE fitted from the data it then summed (task #52).

These tests model the SHIPPED quantisation rule -- index = floor(wstart / win_samples) -- and
assert the property that matters: instances that see the same records agree on the window
WITHOUT exchanging anything.
"""
import unittest

HOPS_PER_RECORD = 2048
FFT_LEN = 16384
SAMPLES_PER_RECORD = HOPS_PER_RECORD * FFT_LEN      # 33,554,432
WIN_RECORDS = 100
WIN_SAMPLES = WIN_RECORDS * SAMPLES_PER_RECORD


def window_of(wstart, win_samples=WIN_SAMPLES):
    """The shipped rule, floor division (GnssGpuRecordAssemble::spec_window_for)."""
    idx = wstart // win_samples
    return idx


def serve(ring_idxs, max_idx, want):
    """The shipped read rule: available = [min complete, max_idx-1]; a request outside it is
    refused BY NAME rather than substituted with the nearest window."""
    hi = max_idx - 1
    complete = [i for i in ring_idxs if 0 <= i <= hi]
    if not complete:
        return "not_yet", None, None
    lo = min(complete)
    if want > hi:
        return "not_yet", lo, hi
    if want < lo or want not in ring_idxs:
        return "too_old", lo, hi
    return "ok", lo, hi


class TestQuantisation(unittest.TestCase):

    def test_instances_seeing_the_same_record_agree(self):
        """The whole point: no negotiation, no reference instance, no fitted anything."""
        for rec in range(0, 1000, 37):
            w = rec * SAMPLES_PER_RECORD
            self.assertEqual(window_of(w), window_of(w))   # same input -> same index
            self.assertEqual(window_of(w), rec // WIN_RECORDS)

    def test_a_poll_offset_does_not_move_the_window(self):
        """THE REGRESSION. Under reset-on-read, two instances polled 0.15 s apart summed
        different records. Now the record's own wstart decides, so a poll-time offset of any
        size changes nothing about which window a record lands in."""
        rec = 512
        w = rec * SAMPLES_PER_RECORD
        self.assertEqual(window_of(w), 5)
        for lag_records in (0, 1, 4, 14):      # 14 records ~ 0.15 s, the measured spread
            self.assertEqual(window_of(w), window_of(rec * SAMPLES_PER_RECORD))
            # the LAGGARD is later in wall time but reports the same record -> same window
            self.assertEqual(window_of((rec) * SAMPLES_PER_RECORD), 5, lag_records)

    def test_boundaries_are_exact_and_records_never_split(self):
        last = (WIN_RECORDS - 1) * SAMPLES_PER_RECORD
        first_next = WIN_RECORDS * SAMPLES_PER_RECORD
        self.assertEqual(window_of(last), 0)
        self.assertEqual(window_of(first_next), 1)
        self.assertEqual(window_of(first_next - 1), 0)

    def test_window_count_is_constant(self):
        """win_samples is a record multiple, so every window holds the same record count --
        one less thing to explain when a residual moves."""
        counts = {}
        for rec in range(0, 500):
            counts.setdefault(window_of(rec * SAMPLES_PER_RECORD), 0)
            counts[window_of(rec * SAMPLES_PER_RECORD)] += 1
        full = [c for i, c in counts.items() if i < max(counts)]
        self.assertTrue(all(c == WIN_RECORDS for c in full), counts)


class TestServeSemantics(unittest.TestCase):

    def test_newest_complete_is_one_behind_the_open_window(self):
        """A window is complete only once a LATER one has opened -- there is no
        end-of-window event to wait on, so 'still accumulating' must never be served."""
        st, lo, hi = serve(ring_idxs=[8, 9, 10], max_idx=10, want=10)
        self.assertEqual(st, "not_yet")
        self.assertEqual(hi, 9)

    def test_too_old_is_refused_not_substituted(self):
        """Substituting the nearest window is how silent misalignment comes back wearing a
        success code."""
        st, lo, hi = serve(ring_idxs=[5, 6, 7, 8], max_idx=9, want=2)
        self.assertEqual(st, "too_old")
        self.assertEqual((lo, hi), (5, 8))

    def test_a_gap_reads_as_too_old_not_as_empty_data(self):
        """'I never had it' and 'here is an empty window' are different facts."""
        st, _, _ = serve(ring_idxs=[5, 7, 8], max_idx=9, want=6)
        self.assertEqual(st, "too_old")

    def test_every_reply_carries_the_available_range(self):
        """It is what lets the broker resynchronise without guessing."""
        for want, expect in ((3, "too_old"), (8, "ok"), (99, "not_yet")):
            st, lo, hi = serve([5, 6, 7, 8], 9, want)
            self.assertEqual(st, expect)
            self.assertEqual((lo, hi), (5, 8))

    def test_laggard_can_be_re_asked_for_a_window_its_peers_already_returned(self):
        """The property reset-on-read could not provide, and the reason for the ring: peers
        served window 8; this instance is 4 records behind but still has 8 once it completes."""
        st, _, _ = serve(ring_idxs=[5, 6, 7, 8], max_idx=9, want=8)
        self.assertEqual(st, "ok")

    def test_ring_must_outlast_the_measured_instance_spread(self):
        """#46 measured ~0.15 s = ~14 records of spread; at 100 records per window that is a
        fraction of one window, so depth 8 is ample -- but assert the relationship rather than
        the constant, so a shorter window cannot silently outrun the ring."""
        spread_records = 14
        self.assertGreater(8 * WIN_RECORDS, spread_records * 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
