#!/usr/bin/env python3

import hashlib
import re
import unittest
from contextlib import redirect_stdout
from io import StringIO

from sanitize_image_tag import MAX_TAG_LENGTH, main, sanitize_image_tag


DOCKER_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


class SanitizeImageTagTest(unittest.TestCase):
    def test_valid_tags_are_unchanged(self):
        for source in (
            "develop",
            "pilotproxy-dtv-detector",
            "release_2026.08",
            "a" * MAX_TAG_LENGTH,
        ):
            with self.subTest(source=source):
                self.assertEqual(sanitize_image_tag(source), source)

    def test_invalid_characters_add_a_hash(self):
        source = "codex/feature+gpu@ci"
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        self.assertEqual(sanitize_image_tag(source), f"codex-feature-gpu-ci-{digest}")

    def test_sanitized_and_already_valid_names_do_not_collide(self):
        self.assertNotEqual(
            sanitize_image_tag("feature/foo"), sanitize_image_tag("feature-foo")
        )

    def test_invalid_prefix_and_empty_result_are_repaired(self):
        for source in (".leading-dot", "-leading-hyphen", "+++"):
            with self.subTest(source=source):
                tag = sanitize_image_tag(source)
                self.assertRegex(tag, DOCKER_TAG_PATTERN)
                self.assertNotEqual(tag, source)

    def test_long_tag_is_truncated_with_a_hash(self):
        source = "a" * (MAX_TAG_LENGTH + 1)
        tag = sanitize_image_tag(source)
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        self.assertEqual(len(tag), MAX_TAG_LENGTH)
        self.assertTrue(tag.endswith(f"-{digest}"))
        self.assertRegex(tag, DOCKER_TAG_PATTERN)

    def test_unicode_is_replaced_with_ascii(self):
        tag = sanitize_image_tag("feature/føø-🚀")
        self.assertTrue(tag.isascii())
        self.assertRegex(tag, DOCKER_TAG_PATTERN)

    def test_empty_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            sanitize_image_tag("")

    def test_cli_accepts_a_leading_hyphen_after_option_separator(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--", "-leading-hyphen"]), 0)
        self.assertEqual(
            output.getvalue().strip(), sanitize_image_tag("-leading-hyphen")
        )


if __name__ == "__main__":
    unittest.main()
