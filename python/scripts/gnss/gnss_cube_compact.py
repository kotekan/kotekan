#!/usr/bin/env python3
"""Offline compaction of the raw beam-cube archive -- the trim ladder (recording plan P3/P4).

    gnss_cube_compact.py compact  --raw DIR --out ROOT [--utc0 S] [--include-open] [--batch N]
        raw rawFileWrite bundles  ->  L0 HDF5, one file per (pointing, sender, UTC day):
        <ROOT>/l0/<pointing>/<sender>/<YYYYMMDD>.h5. Only LIVE slots are kept (a slot with
        n_rec == 0 is silence, not a zero -- see gnssRecord.hpp), the pad is dropped, and the
        window's UTC is written next to its F-engine sample index. Idempotent: a manifest at
        <ROOT>/l0/manifest.json records every raw file already folded in.
    gnss_cube_compact.py rung     --n 12|60 FILE.h5 [...] --out ROOT
        L0  ->  <ROOT>/rung<n>/...: exact SUMS over n consecutive windows (absolute blocks,
        idx // n, so every sender's rungs line up in time). Per (block, slot) with ONE prn.
    gnss_cube_compact.py ls       FILE.h5 [...]
        completeness: windows present vs the day, holes attributed sender-drop / downstream,
        live rows, bytes.
    gnss_cube_compact.py verify   --raw FILE.raw --l0 ROOT [--n 50] [--utc0 S]
        round trip: N random live (window, slot) cells of a raw file re-read from the L0 tree
        and compared BIT-FOR-BIT (float32 in, float32 out; nothing here is a summary).

⚠️ RUN THIS ON cf06 (`/home/kvand/gnss/venv/bin/python`, absolute paths), never on a node.
   The raw tree is on /mnt/cs00; cx43 is a production node and a 9 MB/s read there is a tax on
   the tracker it does not need to pay.

⚠️ THE TIME AXIS. A v3 frame carries utc0 (UTC of F-engine sample 0) and this tool believes
   it. A v2 frame (everything archived 2026-09-05 before the v3 cycle) carries only F-engine
   samples, and the ONE number that turns those into a date -- frame0_utc from the node config,
   GPS-week-rollover corrected -- lived outside the record. For v2 you must pass --utc0, and
   the value is GATED against the raw file's mtime: rawFileWrite closes a file within seconds
   of its last frame, so |utc(last window) - mtime| > --mtime-tol seconds means the epoch is
   wrong (or the F-engine restarted since) and the file is REFUSED, not dated wrong.
   time0 for 2026-09-05 (frame0_nano 1169225859000002870 + rollover): 1788541059.000002870.

⚠️ NOTHING HERE IS A dB, AND NOTHING IS A MEAN. L0 copies the frame's linear SUMS with their
   term counts (`w`), so cells still add; a rung is a sum of sums. The one derived quantity, the
   rung's `cohref` = SUM_windows coh[e] * conj(coh[ref_elem]), exists because a coherent sum
   across windows is meaningless without a common phase reference (phi0 is per window), while
   the cross-element product is blind to that common phase and is exactly the per-element
   relative response the phased-array calibration wants. `ref_elem` is recorded in the attrs;
   `cohpow` = SUM |coh|^2 is the per-window coherence power for the coherence self-check.

@author Keith Vanderlinde
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_cube_read as raw  # noqa: E402

TOOL_VERSION = 1
L0_ROWS = ("idx", "slot", "prn", "n_rec", "n_reanchor", "phi0", "w", "energy", "coh", "incoh")
DEFAULT_POINTINGS = "/home/kvand/gnss/kotekan/config/pointings.yaml"


def h5py_mod():
    try:
        import h5py
    except ImportError:
        raise SystemExit("h5py missing -- run with /home/kvand/gnss/venv/bin/python (venv-ft "
                         "has no h5py)")
    return h5py


def utc_day(t):
    return datetime.datetime.fromtimestamp(t, datetime.UTC).strftime("%Y%m%d")


def iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.UTC).isoformat(timespec="milliseconds")


def sender_name(chain):
    """'cx19//gnss0_b2b_n2assemble' -> 'cx19_gnss0_b2b_n2assemble' (a path component)."""
    host, _, stage = chain.partition("/")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", host + "_" + stage.strip("/"))


# ---------------------------------------------------------------------------------------------
# pointings: declared epochs, never inferred from the data
# ---------------------------------------------------------------------------------------------
def load_pointings(path):
    """Tiny YAML subset reader: a list of {id, from, to, ...} mappings. No pyyaml in venv."""
    if not os.path.exists(path):
        return []
    epochs, cur = [], None
    for line in open(path):
        s = line.split("#", 1)[0].rstrip()
        if not s.strip():
            continue
        m = re.match(r"^\s*-\s+(\w+):\s*(.*)$", s)
        if m:
            cur = {m.group(1): m.group(2).strip().strip('"')}
            epochs.append(cur)
            continue
        m = re.match(r"^\s+(\w+):\s*(.*)$", s)
        if m and cur is not None:
            cur[m.group(1)] = m.group(2).strip().strip('"')
    out = []
    for e in epochs:
        if "id" not in e or "from" not in e:
            continue
        t0 = datetime.datetime.fromisoformat(e["from"].replace("Z", "+00:00")).timestamp()
        t1 = (datetime.datetime.fromisoformat(e["to"].replace("Z", "+00:00")).timestamp()
              if e.get("to") not in (None, "", "null", "~") else None)
        out.append((t0, t1, e["id"]))
    return sorted(out)


def pointing_at(epochs, t):
    for t0, t1, pid in epochs:
        if t >= t0 and (t1 is None or t < t1):
            return pid
    # ⚠️ Undeclared is a fact worth keeping visible in the path, not a silent default: a
    # beam map built from an "unknown" epoch is a map of an unknown pointing.
    return "unknown"


# ---------------------------------------------------------------------------------------------
# L0 writer
# ---------------------------------------------------------------------------------------------
class L0File:
    """One (pointing, sender, day) HDF5 file with append-only extensible datasets."""

    def __init__(self, h5py, path, h, utc0, utc0_source, pointing, day):
        self.h5py = h5py
        self.path = path
        new = not os.path.exists(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = h5py.File(path, "a")
        nb, ne = h["n_bin"], h["n_elem"]
        if new:
            a = self.f.attrs
            a["format"] = "gnss_cube_l0"
            a["tool_version"] = TOOL_VERSION
            a["chain"] = h["chain"]
            a["sender"] = sender_name(h["chain"])
            a["gpu"] = h["gpu"]
            a["n_bin"] = nb
            a["n_elem"] = ne
            a["bin_width"] = h["bin_width"]
            a["win_samples"] = h["win_samples"]
            a["sample_rate"] = h["sample_rate"]
            a["utc0"] = utc0
            a["utc0_source"] = utc0_source
            a["pointing"] = pointing
            a["day"] = day
            a["created"] = iso(time.time())
            a["cube_versions"] = json.dumps([h["version"]])
            a["source_files"] = json.dumps([])
            a["units"] = ("w: (record,channel) term count; energy: SUM replica energy; incoh: "
                          "SUM |A|^2 (beam); coh: SUM A*rot referenced to phi0 (arc); all linear "
                          "SUMS over the window's records, float32 as shipped")
            w = self.f.create_group("win")
            self._mk(w, "idx", np.int64)
            self._mk(w, "utc", np.float64)
            self._mk(w, "wstart0", np.int64)
            self._mk(w, "wstart1", np.int64)
            self._mk(w, "dropped", np.int64)
            self._mk(w, "n_live", np.int32)
            self._mk(w, "freq_id_lo", np.int32, (nb,))
            self._mk(w, "freq_id_hi", np.int32, (nb,))
            r = self.f.create_group("rows")
            self._mk(r, "idx", np.int64)
            self._mk(r, "slot", np.int16)
            self._mk(r, "prn", np.int16)
            self._mk(r, "n_rec", np.int32)
            self._mk(r, "n_reanchor", np.int32)
            self._mk(r, "phi0", np.float64)
            self._mk(r, "w", np.float32, (nb,))
            self._mk(r, "energy", np.float32, (nb,))
            self._mk(r, "coh", np.complex64, (nb, ne))
            self._mk(r, "incoh", np.float32, (nb, ne))
        else:
            a = self.f.attrs
            for k, v in (("n_bin", nb), ("n_elem", ne), ("win_samples", h["win_samples"]),
                         ("sample_rate", h["sample_rate"])):
                if a[k] != v:
                    raise SystemExit(f"{path}: existing file has {k}={a[k]}, frame says {v} -- "
                                     f"the sender's geometry changed mid-day; refusing to mix")
            if abs(float(a["utc0"]) - utc0) > 1e-6:
                raise SystemExit(f"{path}: existing utc0 {a['utc0']!r} != {utc0!r}: two epochs "
                                 f"for one day means an F-engine restart; refusing to mix")
        self.win = {k: [] for k in ("idx", "utc", "wstart0", "wstart1", "dropped", "n_live",
                                    "freq_id_lo", "freq_id_hi")}
        self.rows = {k: [] for k in L0_ROWS}
        self.versions = set(json.loads(self.f.attrs["cube_versions"]))
        self.sources = list(json.loads(self.f.attrs["source_files"]))

    def _mk(self, g, name, dtype, inner=()):
        chunk = 4096 if not inner else max(1, min(1024, (1 << 20) // max(1, int(np.prod(inner)) * np.dtype(dtype).itemsize)))
        g.create_dataset(name, shape=(0,) + inner, maxshape=(None,) + inner, dtype=dtype,
                         chunks=(chunk,) + inner)

    def add(self, h, a, utc):
        live = np.nonzero(a["n_rec"] > 0)[0]
        self.win["idx"].append(h["idx"])
        self.win["utc"].append(utc)
        self.win["wstart0"].append(h["wstart0"])
        self.win["wstart1"].append(h["wstart1"])
        self.win["dropped"].append(h["dropped"])
        self.win["n_live"].append(len(live))
        self.win["freq_id_lo"].append(np.array(a["freq_id_lo"], dtype=np.int32))
        self.win["freq_id_hi"].append(np.array(a["freq_id_hi"], dtype=np.int32))
        self.versions.add(h["version"])
        for p in live:
            self.rows["idx"].append(h["idx"])
            self.rows["slot"].append(p)
            self.rows["prn"].append(a["prn"][p])
            self.rows["n_rec"].append(a["n_rec"][p])
            self.rows["n_reanchor"].append(a["n_reanchor"][p])
            self.rows["phi0"].append(a["phi0"][p])
            self.rows["w"].append(a["w"][p])
            self.rows["energy"].append(a["energy"][p])
            # complex64 from the two float32 planes: re/im survive exactly (same 24-bit mantissa)
            self.rows["coh"].append(a["coh_re"][p].astype(np.float32)
                                    + 1j * a["coh_im"][p].astype(np.float32))
            self.rows["incoh"].append(a["incoh"][p])

    def flush(self, source_files):
        n = len(self.win["idx"])
        if n:
            for k, v in self.win.items():
                self._append(self.f["win"][k], np.asarray(v))
            for k, v in self.rows.items():
                self._append(self.f["rows"][k], np.asarray(v))
        for s in source_files:
            if s not in self.sources:
                self.sources.append(s)
        self.f.attrs["cube_versions"] = json.dumps(sorted(self.versions))
        self.f.attrs["source_files"] = json.dumps(self.sources)
        self.f.attrs["updated"] = iso(time.time())
        for k in self.win:
            self.win[k] = []
        for k in self.rows:
            self.rows[k] = []
        self.f.flush()
        return n

    @staticmethod
    def _append(ds, arr):
        if len(arr) == 0:
            return
        n0 = ds.shape[0]
        ds.resize(n0 + len(arr), axis=0)
        ds[n0:] = arr

    def close(self):
        self.f.close()


def raw_files(rawdir, include_open):
    files = sorted(glob.glob(os.path.join(rawdir, "*.raw")))
    if not files:
        raise SystemExit(f"no .raw under {rawdir}")
    if not include_open:
        # The newest file is the one rawFileWrite is still appending to; everything before it is
        # closed (900 frames or the writer moved on). A file that is not the newest but is short
        # is a writer that was restarted -- complete as far as it will ever be.
        files = files[:-1]
    return files


def frame_utc(h, utc0):
    return utc0 + h["wstart0"] / h["sample_rate"]


def load_manifest(root):
    p = os.path.join(root, "l0", "manifest.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {}


def save_manifest(root, m):
    p = os.path.join(root, "l0", "manifest.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(m, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, p)


def cmd_compact(args):
    h5py = h5py_mod()
    epochs = load_pointings(args.pointings)
    if not epochs:
        print(f"⚠️ no pointing epochs in {args.pointings}; files go under l0/unknown/", file=sys.stderr)
    manifest = load_manifest(args.out)
    files = raw_files(args.raw, args.include_open)
    todo = []
    for p in files:
        st = os.stat(p)
        key = os.path.basename(p)
        prev = manifest.get(key)
        if prev and prev["size"] == st.st_size and not args.redo:
            continue
        if prev and prev["size"] != st.st_size:
            # Grew since we folded it: rows already in L0 cannot be un-appended, and appending
            # the whole file again would duplicate. Only the newest file is ever open, so this
            # means --include-open was used on it earlier; it needs a --redo of that day.
            print(f"⚠️ {key}: size {prev['size']} -> {st.st_size} since it was compacted "
                  f"(--include-open earlier?). SKIPPED; rebuild the day with --redo.",
                  file=sys.stderr)
            continue
        todo.append(p)
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to do")
        return
    print(f"{len(todo)} raw file(s) to fold into {args.out}/l0")

    open_files = {}   # (pointing, sender, day) -> L0File
    stats = dict(frames=0, rows=0, files=0, refused=0)
    t_start = time.time()
    batch_sources = []

    def flush_all():
        n = 0
        for lf in open_files.values():
            n += lf.flush(batch_sources)
        batch_sources.clear()
        return n

    for i, path in enumerate(todo):
        key = os.path.basename(path)
        st = os.stat(path)
        frames = list(raw.iter_frames(path))
        if not frames:
            manifest[key] = dict(size=st.st_size, mtime=st.st_mtime, frames=0, note="empty")
            continue
        # -- epoch: from the frame (v3) or the flag (v2), gated against the file's mtime --------
        per_frame_utc0 = [h["utc0"] for h, _ in frames if h.get("utc0")]
        if per_frame_utc0:
            utc0, src = float(per_frame_utc0[-1]), "frame"
            if args.utc0 is not None and abs(args.utc0 - utc0) > 1e-3:
                print(f"⚠️ {key}: frames carry utc0 {utc0!r}, --utc0 {args.utc0!r} ignored",
                      file=sys.stderr)
        elif args.utc0 is not None:
            utc0, src = float(args.utc0), "flag"
        else:
            print(f"✗ {key}: v2 frames carry no utc0 and no --utc0 given -- REFUSED (an archive "
                  f"dated by guess is worse than one not dated)", file=sys.stderr)
            stats["refused"] += 1
            continue
        last_utc = max(frame_utc(h, utc0) for h, _ in frames)
        gap = st.st_mtime - last_utc
        if not args.no_mtime_gate and not (-args.mtime_tol <= gap <= args.mtime_tol):
            print(f"✗ {key}: last window dated {iso(last_utc)} but the file was last written "
                  f"{iso(st.st_mtime)} ({gap:+.1f} s): the epoch ({src} utc0={utc0!r}) does not "
                  f"describe this file. REFUSED.", file=sys.stderr)
            stats["refused"] += 1
            continue
        for h, a in frames:
            utc = frame_utc(h, utc0)
            day = utc_day(utc)
            pid = pointing_at(epochs, utc)
            snd = sender_name(h["chain"])
            k = (pid, snd, day)
            lf = open_files.get(k)
            if lf is None:
                lf = L0File(h5py, os.path.join(args.out, "l0", pid, snd, day + ".h5"), h, utc0,
                            src, pid, day)
                open_files[k] = lf
            lf.add(h, a, utc)
            stats["frames"] += 1
        batch_sources.append(key)
        manifest[key] = dict(size=st.st_size, mtime=st.st_mtime, frames=len(frames),
                             utc0=utc0, utc0_source=src, first_utc=frame_utc(frames[0][0], utc0),
                             last_utc=last_utc, done=iso(time.time()))
        stats["files"] += 1
        if (i + 1) % args.batch == 0 or i + 1 == len(todo):
            stats["rows"] += sum(len(lf.rows["idx"]) for lf in open_files.values())
            flush_all()
            save_manifest(args.out, manifest)
            el = time.time() - t_start
            print(f"  {i + 1}/{len(todo)} raw files, {stats['frames']} frames, "
                  f"{len(open_files)} L0 files open, {el:.0f} s "
                  f"({stats['frames'] / max(el, 1e-9):.0f} frames/s)", flush=True)
            # Close files for days that can no longer receive rows (keeps handles bounded).
            days = sorted({k[2] for k in open_files})
            if len(days) > 1:
                for k in [k for k in open_files if k[2] != days[-1]]:
                    open_files.pop(k).close()
    for lf in open_files.values():
        lf.close()
    el = time.time() - t_start
    print(f"done: {stats['files']} raw files, {stats['frames']} frames, {stats['rows']} live rows "
          f"in {el:.0f} s; {stats['refused']} refused")


# ---------------------------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------------------------
def expand_h5(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.h5"), recursive=True))
        else:
            out += sorted(glob.glob(p))
    if not out:
        raise SystemExit("no .h5 matched")
    return out


def cmd_ls(args):
    """Completeness per L0 file, with every hole ATTRIBUTED.

    Three faults look alike in one file's window index and are told apart here:
      sender dropped   the frame's cumulative `dropped` rose across the gap: the assembler's
                       output buffer was full (tracker outran the archive) -- the node's fault;
      host-wide        every sender on that host has the same gap: the NODE was down or
                       restarting (a node_up leaves ~75 windows of nothing, fleet cycles do it
                       to every host at once) -- not a loss in the recording chain at all;
      downstream       only this sender, counter flat: bufferSend drop_frames, the network, or
                       the archiver -- the recording chain's fault, and the one worth chasing.
    """
    h5py = h5py_mod()
    files = expand_h5(args.paths)
    info = []
    for p in files:
        with h5py.File(p, "r") as f:
            idx = f["win/idx"][:]
            if len(idx) == 0:
                info.append((p, None))
                continue
            order = np.argsort(idx)
            d = dict(idx=idx[order], dropped=f["win/dropped"][:][order], utc=f["win/utc"][:][order],
                     nrows=f["rows/idx"].shape[0], host=str(f.attrs["sender"]).split("_", 1)[0])
            dd = np.diff(d["idx"])
            d["gaps"] = {(int(d["idx"][j]), int(d["idx"][j + 1])): int(d["dropped"][j + 1] - d["dropped"][j])
                         for j in np.nonzero(dd > 1)[0]}
            info.append((p, d))
    # host-wide = every other sender file of that host has a gap at the same place. The edges never
    # match exactly: senders close their last window a second apart, and the GPU-1 stages come up
    # ~5 windows after GPU-0 at a restart -- so two gaps agree when they OVERLAP by >= 80% of the
    # longer one (09-06 cx19 restart: gnss0 77 windows, gnss1 82-83, offset 1-2).
    # Only files whose window range COVERS the gap get a vote -- ls over several days lists
    # files that never saw that hour, and a file with no gap because it has no such windows is
    # not evidence that the sender was fine.
    by_host = {}
    for p, d in info:
        if d:
            by_host.setdefault(d["host"], []).append((int(d["idx"][0]), int(d["idx"][-1]), list(d["gaps"])))

    def is_hostwide(host, a, b):
        others = [g for i0, i1, g in by_host.get(host, []) if i0 <= a and i1 >= b]
        if len(others) < 2:
            return False
        def same(a2, b2):
            ov = min(b, b2) - max(a, a2)
            return ov >= 0.8 * max(b - a, b2 - a2)
        return all(any(same(a2, b2) for a2, b2 in g) for g in others)

    print(f"{'file':<58} {'wins':>6} {'span':>6} {'holes':>5} {'sdrop':>5} {'rows':>8} "
          f"{'live/w':>6} {'MB':>7}  first .. last (UTC)")
    tot = dict(holes=0, sdrop=0, hostwide=0, downstream=0)
    for p, d in info:
        name = os.path.relpath(p, args.rel) if args.rel else p
        if d is None:
            print(f"{name:<58} empty")
            continue
        idx, dropped, utc = d["idx"], d["dropped"], d["utc"]
        span = int(idx[-1] - idx[0] + 1)
        uniq = len(np.unique(idx))
        holes = span - uniq
        sdrop = int(dropped[-1] - dropped[0])
        mb = os.path.getsize(p) / 1e6
        print(f"{name:<58} {uniq:>6} {span:>6} {holes:>5} {sdrop:>5} {d['nrows']:>8} "
              f"{d['nrows'] / max(uniq, 1):>6.2f} {mb:>7.1f}  {iso(utc[0])[11:23]} .. {iso(utc[-1])[11:23]}")
        if uniq != len(idx):
            print(f"   ⚠️ {len(idx) - uniq} DUPLICATE window(s) -- a raw file folded twice?")
        tot["holes"] += holes
        tot["sdrop"] += sdrop
        pos = {int(v): k for k, v in enumerate(idx)}
        for (a, b), sd in sorted(d["gaps"].items()):
            n = b - a - 1
            if sd:
                kind = f"sender dropped {sd}"
            elif is_hostwide(d["host"], a, b):
                kind, tot["hostwide"] = "host-wide (node down/restart)", tot["hostwide"] + n
            else:
                kind, tot["downstream"] = "DOWNSTREAM (transport/archiver)", tot["downstream"] + n
            if args.gaps or kind.startswith("DOWNSTREAM"):
                print(f"   gap {n:>5} win after idx {a} ({iso(utc[pos[a]])[11:19]})  {kind}")
    print(f"\nholes {tot['holes']}: host-wide {tot['hostwide']}, downstream {tot['downstream']}, "
          f"sender-side drops {tot['sdrop']} (never added: different faults)")


# ---------------------------------------------------------------------------------------------
# rung
# ---------------------------------------------------------------------------------------------
def cmd_rung(args):
    h5py = h5py_mod()
    n = args.n
    for p in expand_h5(args.files):
        with h5py.File(p, "r") as f:
            a = dict(f.attrs)
            if a.get("format") != "gnss_cube_l0":
                raise SystemExit(f"{p}: not an L0 file (format={a.get('format')!r})")
            idx = f["rows/idx"][:]
            if len(idx) == 0:
                print(f"{p}: no rows")
                continue
            slot = f["rows/slot"][:].astype(np.int64)
            prn = f["rows/prn"][:].astype(np.int64)
            n_rec = f["rows/n_rec"][:]
            n_re = f["rows/n_reanchor"][:]
            w = f["rows/w"][:]
            en = f["rows/energy"][:]
            coh = f["rows/coh"][:]
            incoh = f["rows/incoh"][:]
            widx = f["win/idx"][:]
            wutc = f["win/utc"][:]
            ne = int(a["n_elem"])
            # -- reference element: the strongest MEAN incoh over the file, unless given ---------
            if args.ref_elem is not None:
                ref = args.ref_elem
            else:
                ref = int(np.argmax(incoh.sum(axis=(0, 1))))
            # -- group (block, slot, prn); a prn change inside a block makes a second row --------
            blk = idx // n
            key = np.stack([blk, slot, prn], axis=1)
            uk, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.ravel()
            m = len(uk)
            out = dict(
                blk=uk[:, 0], slot=uk[:, 1].astype(np.int16), prn=uk[:, 2].astype(np.int16),
                idx0=np.full(m, np.iinfo(np.int64).max, np.int64), idx1=np.full(m, -1, np.int64),
                n_win=np.zeros(m, np.int32), n_rec=np.zeros(m, np.int64), n_reanchor=np.zeros(m, np.int64),
                w=np.zeros((m,) + w.shape[1:], np.float64), energy=np.zeros((m,) + en.shape[1:], np.float64),
                incoh=np.zeros((m,) + incoh.shape[1:], np.float64),
                cohpow=np.zeros((m,) + incoh.shape[1:], np.float64),
                cohref=np.zeros((m,) + coh.shape[1:], np.complex128))
            np.minimum.at(out["idx0"], inv, idx)
            np.maximum.at(out["idx1"], inv, idx)
            np.add.at(out["n_win"], inv, 1)
            np.add.at(out["n_rec"], inv, n_rec)
            np.add.at(out["n_reanchor"], inv, n_re)
            np.add.at(out["w"], inv, w.astype(np.float64))
            np.add.at(out["energy"], inv, en.astype(np.float64))
            np.add.at(out["incoh"], inv, incoh.astype(np.float64))
            np.add.at(out["cohpow"], inv, (coh.real.astype(np.float64) ** 2 + coh.imag.astype(np.float64) ** 2))
            c128 = coh.astype(np.complex128)
            np.add.at(out["cohref"], inv, c128 * np.conj(c128[:, :, ref:ref + 1]))
            # block time: from the window table (utc of the block's first/last window present)
            worder = np.argsort(widx)
            widx_s, wutc_s = widx[worder], wutc[worder]
            pos0 = np.searchsorted(widx_s, out["idx0"])
            pos1 = np.searchsorted(widx_s, out["idx1"])
            utc0 = wutc_s[np.clip(pos0, 0, len(wutc_s) - 1)]
            utc1 = wutc_s[np.clip(pos1, 0, len(wutc_s) - 1)]
            # windows present per block, for completeness
            wblk = widx // n
            ub, cnt = np.unique(wblk, return_counts=True)
            rel = os.path.relpath(p, os.path.join(args.l0_root, "l0")) if args.l0_root else os.path.basename(p)
            dst = os.path.join(args.out, f"rung{n}", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                os.remove(dst)
            with h5py.File(dst, "w") as g:
                for k, v in a.items():
                    g.attrs[k] = v
                g.attrs["format"] = f"gnss_cube_rung"
                g.attrs["rung_n"] = n
                g.attrs["ref_elem"] = ref
                g.attrs["source_l0"] = p
                # The covering channels, from the L0 window table (constant over a file unless
                # the F-engine was reconfigured mid-day): a rung has no window table of its own.
                g.attrs["freq_id_lo"] = f["win/freq_id_lo"][0]
                g.attrs["freq_id_hi"] = f["win/freq_id_hi"][0]
                g.attrs["created"] = iso(time.time())
                g.attrs["units"] = ("SUMS over the block's windows of the L0 sums (float64); cohpow = "
                                    "SUM |coh|^2 per window; cohref = SUM coh[e]*conj(coh[ref_elem]) "
                                    "-- the per-element relative response, blind to the per-window "
                                    "phase reference; n_win = windows with this (slot, prn) present")
                b = g.create_group("blk")
                b.create_dataset("blk", data=ub)
                b.create_dataset("n_win_present", data=cnt.astype(np.int32))
                r = g.create_group("rows")
                r.create_dataset("utc0", data=utc0)
                r.create_dataset("utc1", data=utc1)
                for k, v in out.items():
                    if k in ("w", "energy", "incoh", "cohpow"):
                        v = v.astype(np.float32)
                    if k == "cohref":
                        v = v.astype(np.complex64)
                    r.create_dataset(k, data=v)
            split = int(np.sum(np.unique(uk[:, :2], axis=0, return_counts=True)[1] > 1))
            print(f"{dst}: {m} rows from {len(idx)} L0 rows, {len(ub)} blocks, ref_elem {ref}, "
                  f"{split} (block,slot) pairs split by a prn swap, "
                  f"{os.path.getsize(dst) / 1e6:.1f} MB (L0 {os.path.getsize(p) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------
def cmd_verify(args):
    h5py = h5py_mod()
    epochs = load_pointings(args.pointings)
    rng = np.random.default_rng(args.seed)
    frames = list(raw.iter_frames(args.raw))
    if not frames:
        raise SystemExit("no frames")
    per_frame_utc0 = [h["utc0"] for h, _ in frames if h.get("utc0")]
    utc0 = float(per_frame_utc0[-1]) if per_frame_utc0 else args.utc0
    if utc0 is None:
        raise SystemExit("v2 frames: pass --utc0")
    cells = [(fi, p) for fi, (h, a) in enumerate(frames) for p in np.nonzero(a["n_rec"] > 0)[0]]
    pick = rng.choice(len(cells), size=min(args.n, len(cells)), replace=False)
    bad = 0
    cache = {}
    for c in pick:
        fi, p = cells[c]
        h, a = frames[fi]
        utc = frame_utc(h, utc0)
        path = os.path.join(args.l0, "l0", pointing_at(epochs, utc), sender_name(h["chain"]),
                            utc_day(utc) + ".h5")
        if path not in cache:
            if not os.path.exists(path):
                print(f"✗ missing {path}")
                bad += 1
                cache[path] = None
                continue
            f = h5py.File(path, "r")
            ridx = f["rows/idx"][:]
            rslot = f["rows/slot"][:]
            cache[path] = (f, ridx, rslot)
        if cache[path] is None:
            bad += 1
            continue
        f, ridx, rslot = cache[path]
        hit = np.nonzero((ridx == h["idx"]) & (rslot == p))[0]
        if len(hit) != 1:
            print(f"✗ idx {h['idx']} slot {p}: {len(hit)} row(s) in {os.path.basename(path)}")
            bad += 1
            continue
        r = int(hit[0])
        checks = [
            ("prn", int(f["rows/prn"][r]) == int(a["prn"][p])),
            ("n_rec", int(f["rows/n_rec"][r]) == int(a["n_rec"][p])),
            ("n_reanchor", int(f["rows/n_reanchor"][r]) == int(a["n_reanchor"][p])),
            ("phi0", float(f["rows/phi0"][r]) == float(a["phi0"][p])),
            ("w", np.array_equal(f["rows/w"][r], a["w"][p])),
            ("energy", np.array_equal(f["rows/energy"][r], a["energy"][p])),
            ("incoh", np.array_equal(f["rows/incoh"][r], a["incoh"][p])),
            ("coh_re", np.array_equal(f["rows/coh"][r].real, a["coh_re"][p])),
            ("coh_im", np.array_equal(f["rows/coh"][r].imag, a["coh_im"][p])),
        ]
        wi = np.nonzero(f["win/idx"][:] == h["idx"])[0]
        checks.append(("win", len(wi) == 1 and abs(float(f["win/utc"][wi[0]]) - utc) < 1e-6
                       and int(f["win/wstart0"][wi[0]]) == h["wstart0"]))
        fails = [k for k, ok in checks if not ok]
        if fails:
            print(f"✗ idx {h['idx']} slot {p} prn {a['prn'][p]}: {fails}")
            bad += 1
    for v in cache.values():
        if v:
            v[0].close()
    print(f"{'PASS' if not bad else 'FAIL'}: {len(pick) - bad}/{len(pick)} cells bit-identical "
          f"({len(cache)} L0 file(s), {len(frames)} frames in {os.path.basename(args.raw)})")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compact", help="raw -> L0 HDF5 per (pointing, sender, day)")
    p.add_argument("--raw", default="/mnt/cs00/data/kvand/gnss_cube/raw")
    p.add_argument("--out", default="/mnt/cs00/data/kvand/gnss_cube")
    p.add_argument("--utc0", type=float, help="UTC of F-engine sample 0 (REQUIRED for v2 frames)")
    p.add_argument("--mtime-tol", type=float, default=60.0,
                   help="max |utc(last window) - file mtime| before the epoch is refused "
                        "(measured 09-05: mtime - last window = +1.2 .. +2.6 s)")
    p.add_argument("--no-mtime-gate", action="store_true")
    p.add_argument("--include-open", action="store_true", help="also fold the newest (open) file")
    p.add_argument("--batch", type=int, default=12, help="raw files per HDF5 flush")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--redo", action="store_true", help="ignore the manifest (DUPLICATES rows in "
                   "existing L0 files -- delete the day's L0 first)")
    p.add_argument("--pointings", default=DEFAULT_POINTINGS)
    p.set_defaults(fn=cmd_compact)
    p = sub.add_parser("ls", help="completeness of L0 files")
    p.add_argument("paths", nargs="+")
    p.add_argument("--gaps", action="store_true", help="list every hole with its attribution")
    p.add_argument("--rel", help="print paths relative to this dir")
    p.set_defaults(fn=cmd_ls)
    p = sub.add_parser("rung", help="L0 -> exact n-window sums")
    p.add_argument("files", nargs="+")
    p.add_argument("--n", type=int, required=True, choices=(12, 60))
    p.add_argument("--out", default="/mnt/cs00/data/kvand/gnss_cube")
    p.add_argument("--l0-root", default="/mnt/cs00/data/kvand/gnss_cube",
                   help="root whose l0/ prefix is replaced by rung<n>/ in the output path")
    p.add_argument("--ref-elem", type=int, help="element for cohref (default: strongest mean incoh)")
    p.set_defaults(fn=cmd_rung)
    p = sub.add_parser("verify", help="raw <-> L0 bit-for-bit round trip")
    p.add_argument("--raw", required=True)
    p.add_argument("--l0", default="/mnt/cs00/data/kvand/gnss_cube")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--utc0", type=float)
    p.add_argument("--pointings", default=DEFAULT_POINTINGS)
    p.set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
