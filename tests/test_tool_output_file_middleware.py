"""Unit-тесты middleware сохранения больших tool outputs в pickle."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import ToolMessage

from deep_agent_test.tool_output_file_middleware import ToolOutputFileMiddleware


class ToolOutputFileMiddlewareTests(unittest.TestCase):
    def test_large_tabular_result_saved_as_pkl(self) -> None:
        rows = [{"id": index, "value": f"row-{index}"} for index in range(12)]
        with tempfile.TemporaryDirectory() as temp_dir:
            middleware = ToolOutputFileMiddleware(
                output_dir=Path(temp_dir),
                min_rows_to_save=10,
                min_content_chars_to_save=60000,
                preview_rows=2,
            )
            original = ToolMessage(
                content="ignored",
                tool_call_id="call-1",
                name="read_table",
                artifact={"rows": rows},
            )
            processed = middleware._process_tool_message(result=original, tool_name="read_table")
            self.assertNotEqual(processed.content, original.content)
            self.assertIn(".pkl", processed.content)
            self.assertIn("pickle", processed.content.lower())
            self.assertIn("Preview", processed.content)

            saved_path = Path(processed.artifact["saved_file"])
            self.assertTrue(saved_path.exists())
            self.assertEqual(saved_path.suffix, ".pkl")
            with saved_path.open("rb") as file:
                loaded_rows = pickle.load(file)
            self.assertEqual(len(loaded_rows), 12)

    def test_small_result_saved_to_pkl_but_keeps_inline_content(self) -> None:
        rows = [{"id": 1}]
        with tempfile.TemporaryDirectory() as temp_dir:
            middleware = ToolOutputFileMiddleware(
                output_dir=Path(temp_dir),
                min_rows_to_save=10,
                min_content_chars_to_save=60000,
            )
            original = ToolMessage(
                content="small inline payload",
                tool_call_id="call-1",
                name="read_table",
                artifact={"rows": rows},
            )
            processed = middleware._process_tool_message(result=original, tool_name="read_table")
            self.assertIn("small inline payload", str(processed.content))
            self.assertIn("переиспользования", str(processed.content).lower())
            self.assertIn(".pkl", str(processed.content))

            saved_path = Path(processed.artifact["saved_file"])
            self.assertTrue(saved_path.exists())
            with saved_path.open("rb") as file:
                self.assertEqual(pickle.load(file), rows)


if __name__ == "__main__":
    unittest.main()
