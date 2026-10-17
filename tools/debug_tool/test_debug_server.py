"""Run with: python -m unittest discover -s tools/debug_tool -p 'test_*.py'."""

import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import debug_server


class Links(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.hrefs = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.append(dict(attrs)["href"])


class DebugServerSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dumps = self.root / "dumps"
        self.dumps.mkdir()
        (self.root / "secret.txt").write_text("private")
        self.directory = patch.object(debug_server, "DUMP_DIR", str(self.dumps))
        self.directory.start()
        self.addCleanup(self.directory.stop)
        self.client = debug_server.app.test_client()

    def test_existing_viewer_urls_work(self):
        for path in (
            "/",
            "/templates/pipeline_tree.html",
            "/templates/dump_viewer.html",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)
        self.assertEqual(self.client.get("/debug_server.py").status_code, 404)
        self.assertEqual(self.client.get("/templates/files.html").status_code, 404)

    def test_dump_file_and_encoded_name_round_trip(self):
        nested = self.dumps / "a folder"
        nested.mkdir()
        name = 'crash_stats <script> & "quoted".json'
        (nested / name).write_text('{"buffers": {}, "trackers": {}}')
        response = self.client.get("/dump_dir/a%20folder")
        links = Links(response.get_data(as_text=True)).hrefs
        self.assertEqual(len(links), 1)
        self.assertNotIn("<script>", response.get_data(as_text=True))
        file_response = self.client.get(links[0])
        self.addCleanup(file_response.close)
        self.assertEqual(file_response.status_code, 200)
        self.assertIn(b'"buffers"', file_response.data)
        self.assertTrue(
            file_response.headers["Content-Disposition"].startswith("attachment")
        )
        self.assertEqual(file_response.headers["X-Content-Type-Options"], "nosniff")

    def test_internal_symlink_remains_readable(self):
        (self.dumps / "target.json").write_text('{"buffers": {}}')
        (self.dumps / "alias.json").symlink_to(self.dumps / "target.json")
        response = self.client.get("/dump_dir/alias.json")
        self.addCleanup(response.close)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'{"buffers": {}}')

    def test_parent_and_absolute_paths_are_rejected(self):
        for path in (
            "/dump_dir/../secret.txt",
            "/dump_dir/%2e%2e/secret.txt",
            "/dump_dir/%2e%2e%2fsecret.txt",
            "/dump_dir/%2fetc/passwd",
            "/dump_dir/a/../../secret.txt",
        ):
            with self.subTest(path=path):
                response = self.client.get(path, follow_redirects=True)
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(b"private", response.data)

    def test_symlink_escape_is_rejected_and_not_listed(self):
        (self.dumps / "escape").symlink_to(self.root, target_is_directory=True)
        (self.dumps / "secret-link").symlink_to(self.root / "secret.txt")
        for path in ("escape/secret.txt", "secret-link"):
            self.assertEqual(self.client.get("/dump_dir/" + path).status_code, 404)
        listing = self.client.get("/dump_dir").get_data(as_text=True)
        self.assertNotIn("escape", listing)
        self.assertNotIn("secret-link", listing)

    def test_uploaded_html_is_never_rendered(self):
        name = "attack.html"
        (self.dumps / name).write_text("<script>alert(1)</script>")
        response = self.client.get("/dump_dir/" + name)
        self.addCleanup(response.close)
        self.assertEqual(response.mimetype, "application/octet-stream")
        self.assertTrue(
            response.headers["Content-Disposition"].startswith("attachment")
        )


if __name__ == "__main__":
    unittest.main()
