"""Streaming Galileo E6-B C/NAV decoder (the E6 chain's HAS pages, run by nav_decode.py).

Ingests the e6b_symbols stage's per-PRN nav_obs windows (GnssNavSymbols, one symbol per 1 ms
record), decodes every whole page in each, and gathers the HAS pages of all satellites by message
type and ID: a message of MS pages goes out as Reed-Solomon pages spread over the satellites, and
any MS distinct ones give it back (galileo_cnav.recover). Each window is decoded on its own, as
for B2b: a 3 s window holds two whole 1 s pages, and a page seen again in the next window is
counted once, by its start.

A recovered message stays held under its ID; a later page of that ID is checked against it, and
one that disagrees starts the next message on that ID. Pages older than MID_TIMEOUT_S are
dropped (the ICD's message completion time-out). Type 1 messages are parsed with the masks the
earlier ones carried (galileo_cnav.parse_mt1).

There is no ephemeris here: ephemeris() is None. health() counts pages; newest holds the newest
type 1 message of each kind (orbit, clock, bias).

nav_obs: {utc_ref, rec_dt, br, phase, bits=[[idx, +-1], ...], soft=[10 x value, ...]}, br = 1,
rec_dt = 0.001. soft, when present, weights the Viterbi; without it the decode is hard-decision.
"""

import time

import numpy as np

import galileo_cnav as C

PAGE = C.PAGE_SYMS
MID_TIMEOUT_S = 150.0
SEEN_KEEP_S = 600.0
SETS_KEEP = 3


class _PrnState:
    def __init__(self):
        self.last_obs = None
        self.seen = {}  # page start (s, rounded) -> host time decoded
        self.n_pages = 0  # CRC-valid pages
        self.n_dummy = 0
        self.n_bad = 0  # synced pages whose CRC failed
        self.n_msgs = 0  # recovered messages this satellite sent a page of
        self.page = None  # (bits, host time decoded) of the newest page


class HasPredictor:
    def __init__(self):
        self._p = {}
        # (type, mid) -> {ms, pages {pid: octets}, prns {pid: prn}, t, octets}
        self._msgs = {}
        self._masks = {}  # mask ID -> systems (galileo_cnav.parse_mt1)
        self.n_messages = 0
        # "orbit" / "clock" / "bias" / "mask" -> (host time, parsed message)
        self.newest = {}
        # (kind, mask ID, IOD set ID) -> (host time, parsed message), the newest SETS_KEEP of a
        # kind: a clock message pairs with the orbit message of its own set, which is not always
        # the newest one.
        self.sets = {}

    def ingest(self, prn, obs):
        st = self._p.setdefault(prn, _PrnState())
        try:
            utc_ref = float(obs["utc_ref"])
            sym_s = int(obs["br"]) * float(obs["rec_dt"])
            pairs = obs["bits"]
            soft = obs.get("soft")
        except (KeyError, TypeError, ValueError, AttributeError):
            return
        key = (utc_ref, len(pairs))
        if key == st.last_obs or sym_s <= 0 or len(pairs) < PAGE:
            return
        st.last_obs = key
        idx = np.array([int(p[0]) for p in pairs])
        sg = np.zeros(idx.max() + 1)
        sg[idx] = [int(p[1]) for p in pairs]
        s = sg
        if soft is not None and len(soft) == len(pairs):
            v = np.zeros_like(sg)
            v[idx] = np.asarray(soft, dtype=float) / 10.0
            s = np.where(
                v * sg > 0, v, 0.01 * sg
            )  # a value rounded to 0 keeps its sign
        now = time.time()
        sync = 1.0 - 2.0 * np.array(C.SYNC, dtype=float)
        c = np.correlate(sg[: len(sg) - PAGE + C.N_SYNC], sync, "valid")
        for i in np.nonzero(np.abs(c) == C.N_SYNC)[0]:
            t = round(utc_ref + i * sym_s)
            if t in st.seen or not sg[i : i + PAGE].all():
                continue
            bits = C.decode_page(np.sign(c[i]) * s[i : i + PAGE])
            if bits is None:
                continue
            if not C.check_page(bits):
                st.n_bad += 1
                continue
            st.seen[t] = now
            st.n_pages += 1
            st.page = (bits, now)
            h = C.header(bits)
            if h is None:
                st.n_dummy += 1
            else:
                self._add(prn, h, C.body_octets(bits), now)
        if len(st.seen) > SEEN_KEEP_S:
            st.seen = {k: v for k, v in st.seen.items() if now - v < SEEN_KEEP_S}

    def _add(self, prn, h, octets, now):
        k = (h["type"], h["mid"])
        m = self._msgs.get(k)
        held = m["pages"].get(h["pid"]) if m else None
        if (
            m is None
            or m["ms"] != h["ms"]
            or now - m["t"] > MID_TIMEOUT_S
            or (held is not None and not np.array_equal(held, octets))
            or (m["octets"] is not None and not self._agrees(m, h["pid"], octets))
        ):
            m = self._msgs[k] = {
                "ms": h["ms"],
                "pages": {},
                "prns": {},
                "t": now,
                "octets": None,
            }
        m["pages"][h["pid"]] = octets
        m["prns"][h["pid"]] = prn
        m["t"] = now
        if m["octets"] is not None or len(m["pages"]) < m["ms"]:
            return
        msg, _ = C.recover(m["pages"], m["ms"])
        if msg is None:
            return
        m["octets"] = msg
        self.n_messages += 1
        for p in set(m["prns"].values()):
            if p in self._p:
                self._p[p].n_msgs += 1
        if h["type"] != 1:
            return
        try:
            d = C.parse_mt1(C.message_bits(msg), self._masks)
        except ValueError:
            return
        d["mid"], d["ms"] = h["mid"], h["ms"]
        for kind, flags in (
            ("mask", ("mask",)),
            ("orbit", ("orbit",)),
            ("clock", ("clock", "clock_subset")),
            ("bias", ("code_bias", "phase_bias")),
        ):
            if any(f in d["flags"] for f in flags) and "sats" in d:
                self.newest[kind] = (now, d)
                self.sets[(kind, d["mask_id"], d["iod_set"])] = (now, d)
                held = sorted((v[0], k) for k, v in self.sets.items() if k[0] == kind)
                for _, k in held[:-SETS_KEEP]:
                    del self.sets[k]

    @staticmethod
    def _agrees(m, pid, octets):
        """Whether a page fits the message already recovered under its ID."""
        return np.array_equal(C._combine(C.G[pid - 1, : m["ms"]], m["octets"]), octets)

    def ephemeris(self, prn):
        return None

    def corrections(self):
        """{kind: [message, ...]} for orbit, clock and bias, newest first, each parse_mt1's form
        with t_rx, the host time it was recovered: gnss_has's input."""
        out = {}
        for (kind, _, _), (t, d) in sorted(self.sets.items(), key=lambda kv: -kv[1][0]):
            if kind != "mask":
                out.setdefault(kind, []).append(dict(d, t_rx=round(t, 2)))
        return out

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {
            "pages": st.n_pages,
            "dummy": st.n_dummy,
            "crc_fail": st.n_bad,
            "messages": st.n_msgs,
        }
