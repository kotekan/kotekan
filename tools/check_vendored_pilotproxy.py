#!/usr/bin/env python3
"""Check the PilotProxy vendor pin and file hashes.

--offline compares local files with VENDOR.json (the default).
--fetch also compares them with the pinned upstream checkout.
Returns 0 on success and 1 on a mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT / "external" / "pilotproxy"
MANIFEST = VENDOR_DIR / "VENDOR.json"


def digest(path: pathlib.Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def check_offline(manifest: dict) -> list[str]:
    problems = []
    for name, want in manifest["files"].items():
        path = VENDOR_DIR / name
        if not path.is_file():
            problems.append(f"{name}: missing from {VENDOR_DIR}")
            continue
        got = digest(path)
        if got != want:
            problems.append(
                f"{name}: digest mismatch\n    manifest {want}\n    on disk  {got}"
            )
    listed = set(manifest["files"])
    # Local build files and vendor records are allowed alongside the upstream copies.
    allowed_extra = {"README.md", "VENDOR.json", "CMakeLists.txt", "LICENSE"}
    for path in sorted(VENDOR_DIR.iterdir()):
        if (
            path.is_file()
            and path.name not in listed
            and path.name not in allowed_extra
        ):
            problems.append(
                f"{path.name}: present in vendor dir but not in the manifest"
            )
    return problems


def check_fetch(manifest: dict) -> list[str]:
    if shutil.which("git") is None:
        return ["--fetch requested but git is not available"]
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp) / "upstream"
        try:
            subprocess.run(["git", "init", "--quiet", str(work)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(work),
                    "fetch",
                    "--quiet",
                    "--depth",
                    "1",
                    manifest["upstream_repo"],
                    manifest["upstream_commit"],
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(work), "checkout", "--quiet", "FETCH_HEAD"],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            return [
                f"could not fetch {manifest['upstream_repo']} "
                f"at {manifest['upstream_commit']}: {exc}"
            ]
        subdir = work / manifest["upstream_subdir"]
        for name in manifest["files"]:
            upstream_file = subdir / name
            if not upstream_file.is_file():
                problems.append(f"{name}: not present upstream at the pinned commit")
                continue
            if digest(upstream_file) != digest(VENDOR_DIR / name):
                problems.append(
                    f"{name}: vendored copy differs from upstream at the pinned commit"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--offline",
        dest="fetch",
        action="store_false",
        help="verify only the local vendor manifest (the default)",
    )
    mode.add_argument(
        "--fetch",
        action="store_true",
        help="also diff against upstream at the pinned commit (needs network)",
    )
    parser.set_defaults(fetch=False)
    args = parser.parse_args()

    if not MANIFEST.is_file():
        print(f"FAIL: no vendor manifest at {MANIFEST}", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text())

    problems = check_offline(manifest)
    if args.fetch and not problems:
        problems += check_fetch(manifest)

    if problems:
        print(
            "FAIL: vendored PilotProxy core does not match its manifest",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nVendored files are byte-identical copies of upstream and must not be edited\n"
            "in place. Fix upstream (%s), then re-vendor and refresh\n"
            "external/pilotproxy/VENDOR.json." % manifest["upstream_repo"],
            file=sys.stderr,
        )
        return 1

    scope = "manifest + upstream" if args.fetch else "manifest"
    print(
        f"OK: {len(manifest['files'])} vendored files match the {scope} "
        f"({manifest['upstream_repo']} @ {manifest['upstream_commit'][:12]})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
