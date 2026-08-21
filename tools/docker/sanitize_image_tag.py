#!/usr/bin/env python3
"""Convert a branch or ref name into a valid, collision-resistant Docker tag."""

import argparse
import hashlib
import re
from typing import Optional, Sequence


MAX_TAG_LENGTH = 128
HASH_LENGTH = 12
TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
INVALID_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]+")
INVALID_PREFIX = re.compile(r"^[.-]+")


def sanitize_image_tag(source: str) -> str:
    """Return a valid Docker tag derived deterministically from *source*.

    A short hash is included whenever sanitizing or truncating the source could
    otherwise make distinct Git refs resolve to the same image tag.
    """

    if not source:
        raise ValueError("image tag source must not be empty")

    sanitized = INVALID_CHARACTERS.sub("-", source)
    sanitized = INVALID_PREFIX.sub("", sanitized) or "ref"

    if sanitized != source or len(sanitized) > MAX_TAG_LENGTH:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:HASH_LENGTH]
        prefix_length = MAX_TAG_LENGTH - HASH_LENGTH - 1
        sanitized = f"{sanitized[:prefix_length]}-{digest}"

    if not TAG_PATTERN.fullmatch(sanitized):
        raise ValueError(f"could not derive a valid Docker tag from {source!r}")

    return sanitized


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a branch or ref name into a valid Docker image tag."
    )
    parser.add_argument("source", help="Branch, ref, or manually supplied tag name")
    args = parser.parse_args(argv)

    try:
        tag = sanitize_image_tag(args.source)
    except ValueError as error:
        parser.error(str(error))

    print(tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
