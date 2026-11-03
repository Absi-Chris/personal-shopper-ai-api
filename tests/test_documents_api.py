import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main


class DocumentApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.documents_dir = Path(self.temp_dir.name)
        self.original_documents_dir = main.DOCUMENTS_DIR
        main.DOCUMENTS_DIR = self.documents_dir
        self.client = TestClient(main.app)

    def tearDown(self):
        main.DOCUMENTS_DIR = self.original_documents_dir
        self.temp_dir.cleanup()

    def test_list_documents(self):
        stored_as = f"{'a' * 32}_policy.txt"
        (self.documents_dir / stored_as).write_text("policy text", encoding="utf-8")

        response = self.client.get("/documents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"documents": [{"stored_as": stored_as, "filename": "policy.txt", "size_bytes": 11}]},
        )

    def test_delete_document_removes_file_and_vectors(self):
        stored_as = f"{'b' * 32}_manual.md"
        stored_path = self.documents_dir / stored_as
        stored_path.write_text("manual text", encoding="utf-8")

        with patch.object(main, "delete_document_vectors") as delete_vectors:
            response = self.client.delete(f"/documents/{stored_as}")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(stored_path.exists())
        delete_vectors.assert_called_once_with(stored_as)

    def test_delete_missing_document_returns_404(self):
        response = self.client.delete(f"/documents/{'c' * 32}_missing.pdf")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()