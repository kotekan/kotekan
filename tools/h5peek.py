#!/usr/bin/env python3
"""Quick HDF5 inspector.

Usage:
  h5peek.py file.h5 [--path /group/subgroup] [--per-dim 4] [--max-attrs 80]

Shows:
- tree of groups/datasets
- attributes (metadata)
- a compact preview of dataset data (corner slice)
"""
import argparse
import sys
import numpy as np

try:
    import h5py
    from h5py import h5d
except ImportError as e:
    print("This tool requires 'h5py'. Install with: pip install h5py", file=sys.stderr)
    raise


def shorten(s, maxlen=80):
    s = str(s)
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


def fmt_attrs(obj, maxlen=80, indent="  "):
    out = []
    if hasattr(obj, "attrs") and len(obj.attrs) > 0:
        out.append(f"{indent}# attributes:")
        for k, v in obj.attrs.items():
            try:
                val = v.tolist() if hasattr(v, "tolist") else v
            except Exception:
                val = v
            out.append(f"{indent}- {k} = {shorten(val, maxlen)}")
    return "\n".join(out)


def dtype_info(d):
    if d.names:  # compound dtype
        fields = ", ".join([f"{name}:{d[name].str}" for name in d.names])
        return f"compound({fields})"
    meta = getattr(d, "metadata", {}) or {}
    if "vlen" in meta:
        return f"vlen({meta['vlen']})"
    return str(d)


def filter_info(dset):
    """Return a short string describing any filters on the dataset."""
    try:
        plist = dset.id.get_create_plist()
        n = plist.get_nfilters()
        if n == 0:
            return "filters=none"

        entries = []
        for i in range(n):
            info = plist.get_filter(i)
            # h5py returns a tuple like (id, flags, values, name[, config])
            filter_id = info[0]
            flags = info[1] if len(info) > 1 else 0
            params = info[2] if len(info) > 2 else ()
            name = info[3] if len(info) > 3 else ""
            config = info[4] if len(info) > 4 else None

            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            params_list = list(params) if params is not None else []
            extra = f", config={config}" if config is not None else ""
            entries.append(
                f"{name or filter_id} (id={filter_id}, flags={flags}, params={params_list}{extra})"
            )
        return "filters=" + "; ".join(entries)
    except Exception:
        return ""


def allocation_info(dset):
    """Return human-readable fill/alloc time and fillvalue info."""
    try:
        plist = dset.id.get_create_plist()
        alloc = plist.get_alloc_time()
        fill_time = plist.get_fill_time()
        alloc_name = {
            h5d.ALLOC_TIME_EARLY: "EARLY",
            h5d.ALLOC_TIME_INCR: "INCR",
            h5d.ALLOC_TIME_LATE: "LATE",
            h5d.ALLOC_TIME_DEFAULT: "DEFAULT",
        }.get(alloc, str(alloc))
        fill_name = {
            h5d.FILL_TIME_IFSET: "IFSET",
            h5d.FILL_TIME_ALLOC: "ALLOC",
            h5d.FILL_TIME_NEVER: "NEVER",
        }.get(fill_time, str(fill_time))
        fillval = dset.fillvalue
        extra = [f"alloc_time={alloc_name}", f"fill_time={fill_name}"]
        if fillval is not None:
            extra.append(f"fillvalue={fillval}")
        return ", ".join(extra)
    except Exception:
        return ""


def storage_props(dset):
    props = []
    try:
        if dset.compression:
            props.append(f"compression={dset.compression}")
        if dset.compression_opts is not None:
            props.append(f"opts={dset.compression_opts}")
        if dset.chunks:
            props.append(f"chunks={dset.chunks}")
        if dset.scaleoffset is not None:
            props.append(f"scaleoffset={dset.scaleoffset}")
        if dset.shuffle:
            props.append("shuffle=True")
        if dset.fletcher32:
            props.append("fletcher32=True")
        filt = filter_info(dset)
        if filt:
            props.append(filt)
        alloc = allocation_info(dset)
        if alloc:
            props.append(alloc)
    except Exception:
        pass
    return (", ".join(props)) if props else ""


def corner_slice(shape, per_dim):
    """Return a tuple of slices selecting a small 'corner' of data."""
    if len(shape) == 0:
        return ()
    return tuple(slice(0, min(n, per_dim)) for n in shape)


def preview_data(dset, per_dim=4, max_chars=240):
    try:
        sl = corner_slice(dset.shape, per_dim)
        arr = dset[sl]
        # For byte strings, convert to str for readability (limit length)
        if np.issubdtype(arr.dtype, np.bytes_):
            arr = arr.astype("U")  # decode as UTF-8 if possible
        text = np.array2string(
            arr, threshold=per_dim ** 2, edgeitems=per_dim, max_line_width=120
        )
        return shorten(text, max_chars)
    except Exception as e:
        return f"<preview failed: {e}>"


def walk(name, obj, args, depth=0):
    ind = "  " * depth
    if isinstance(obj, h5py.Group) and not isinstance(obj, h5py.Dataset):
        print(f"{ind}[GROUP] {name or '/'}")
        attrs_text = fmt_attrs(obj, maxlen=args.max_attrs, indent=ind + "  ")
        if attrs_text:
            print(attrs_text)
        # Iterate children sorted for stable output
        keys = list(obj.keys())
        for k in sorted(keys):
            walk(f"{name}/{k}" if name else "/" + k, obj[k], args, depth + 1)
    else:  # dataset
        shape = obj.shape
        dt = dtype_info(obj.dtype)
        props = storage_props(obj)
        prop_str = f"  {{{props}}}" if props else ""
        print(f"{ind}[DATASET] {name}  shape={shape}  dtype={dt}{prop_str}")
        attrs_text = fmt_attrs(obj, maxlen=args.max_attrs, indent=ind + "  ")
        if attrs_text:
            print(attrs_text)
        print(
            f"{ind}  preview: {preview_data(obj, per_dim=args.per_dim, max_chars=args.max_preview)}"
        )


def main():
    p = argparse.ArgumentParser(description="Quick HDF5 inspector")
    p.add_argument("file", help="Path to .h5/.hdf5 file")
    p.add_argument(
        "--path", default="/", help="Group or dataset path to start from (default: /)"
    )
    p.add_argument(
        "--per-dim",
        type=int,
        default=4,
        help="Elements per dimension for preview corner",
    )
    p.add_argument(
        "--max-attrs", type=int, default=80, help="Max chars per attribute value"
    )
    p.add_argument(
        "--max-preview", type=int, default=240, help="Max chars for data preview text"
    )
    args = p.parse_args()

    try:
        with h5py.File(args.file, "r") as f:
            start = f[args.path] if args.path != "/" else f
            if start is f:
                walk("", start, args, depth=0)
            else:
                # Print the subtree root first
                name = args.path.rstrip("/")
                walk(name if name else "/", start, args, depth=0)
    except KeyError:
        print(f"Path not found: {args.path}", file=sys.stderr)
        sys.exit(2)
    except OSError as e:
        print(f"Error opening file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
