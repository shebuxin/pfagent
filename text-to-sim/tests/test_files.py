"""Unit tests for src.files pure helpers and tmpdir-testable functions.

Before these tests, src/files.py (615 lines of code-execution plumbing,
directory scanning, and generated-image tracking) had zero coverage.
These tests pin the pieces that are pure or only touch the filesystem,
deliberately skipping ``execute_python_code`` (which is tightly coupled
to streamlit.session_state and spawns subprocesses) and
``display_file_section`` (pure Streamlit UI).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.files import (  # noqa: E402
    _build_case_preview_records,
    _normalize_case_preview_value,
    _summarize_case_preview_table,
    copy_new_images_to_output,
    delete_file,
    detect_file_operations,
    extract_generated_image_paths,
    get_file_size,
    get_files_in_directory,
    infer_andes_case_context,
    list_image_files,
    modify_code_file_paths,
    resolve_generated_image_path,
    resolve_python_executable,
    summarize_andes_case_file,
)


class DetectFileOperationsTests(unittest.TestCase):
    def test_detects_direct_open_call(self):
        code = 'data = open("case.xlsx").read()'
        self.assertEqual(detect_file_operations(code), ["case.xlsx"])

    def test_detects_with_open_call(self):
        code = 'with open("network.json") as f:\n    content = f.read()'
        self.assertEqual(detect_file_operations(code), ["network.json"])

    def test_deduplicates_multiple_references(self):
        code = 'open("a.xlsx"); open("a.xlsx"); open("b.json")'
        result = sorted(detect_file_operations(code))
        self.assertEqual(result, ["a.xlsx", "b.json"])

    def test_empty_code_returns_empty_list(self):
        self.assertEqual(detect_file_operations(""), [])

    def test_single_quotes_work(self):
        code = "open('ieee14.xlsx')"
        self.assertEqual(detect_file_operations(code), ["ieee14.xlsx"])

    def test_ignores_non_literal_paths(self):
        # Only string literals are matched; variables fall through.
        code = 'path = "case.xlsx"; open(path)'
        self.assertEqual(detect_file_operations(code), [])


class ModifyCodeFilePathsTests(unittest.TestCase):
    def test_rewrites_open_call(self):
        code = 'open("case.xlsx")'
        rewritten = modify_code_file_paths(code, {"case.xlsx": "/tmp/run/case.xlsx"})
        self.assertIn('open("/tmp/run/case.xlsx"', rewritten)
        self.assertNotIn('open("case.xlsx")', rewritten)

    def test_rewrites_with_open_call(self):
        code = 'with open("case.xlsx") as f: pass'
        rewritten = modify_code_file_paths(code, {"case.xlsx": "/tmp/x.xlsx"})
        self.assertIn('with open("/tmp/x.xlsx"', rewritten)

    def test_leaves_unmapped_files_alone(self):
        code = 'open("kept.xlsx"); open("remap.xlsx")'
        rewritten = modify_code_file_paths(code, {"remap.xlsx": "/new/remap.xlsx"})
        self.assertIn('open("kept.xlsx")', rewritten)
        self.assertIn('open("/new/remap.xlsx"', rewritten)

    def test_empty_mappings_returns_unchanged(self):
        code = 'open("a.xlsx")'
        self.assertEqual(modify_code_file_paths(code, {}), code)


class ResolvePythonExecutableTests(unittest.TestCase):
    def test_returns_existing_executable(self):
        # The default resolution should land on a path that exists and
        # is executable (sys.executable is always a valid candidate).
        resolved = resolve_python_executable()
        self.assertTrue(os.path.exists(resolved) or resolved == "python3")

    def test_conda_env_path_bin_dir_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "envs", "fake", "bin")
            os.makedirs(bin_dir)
            python_path = os.path.join(bin_dir, "python")
            # Create a fake executable
            Path(python_path).write_text("#!/bin/sh\necho fake\n")
            os.chmod(python_path, 0o755)
            with patch.dict(os.environ, {"CONDA_ENV_PATH": bin_dir}, clear=False):
                resolved = resolve_python_executable()
            self.assertEqual(resolved, python_path)

    def test_conda_env_path_env_prefix_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_prefix = os.path.join(tmp, "envs", "fake")
            bin_dir = os.path.join(env_prefix, "bin")
            os.makedirs(bin_dir)
            python_path = os.path.join(bin_dir, "python")
            Path(python_path).write_text("#!/bin/sh\necho fake\n")
            os.chmod(python_path, 0o755)
            with patch.dict(os.environ, {"CONDA_ENV_PATH": env_prefix}, clear=False):
                resolved = resolve_python_executable()
            self.assertEqual(resolved, python_path)


class NormalizeCasePreviewValueTests(unittest.TestCase):
    def test_passthrough_primitives(self):
        self.assertEqual(_normalize_case_preview_value("x"), "x")
        self.assertEqual(_normalize_case_preview_value(42), 42)
        self.assertEqual(_normalize_case_preview_value(1.5), 1.5)
        self.assertEqual(_normalize_case_preview_value(True), True)

    def test_none_passthrough(self):
        self.assertIsNone(_normalize_case_preview_value(None))

    def test_nan_becomes_none(self):
        import pandas as pd
        self.assertIsNone(_normalize_case_preview_value(float("nan")))
        self.assertIsNone(_normalize_case_preview_value(pd.NA))

    def test_tuple_becomes_string(self):
        # The domain of this helper is Excel/JSON cell values (scalars).
        # A scalar that isn't a primitive (e.g. a tuple from a JSON list
        # that somehow sneaks through) falls through to str().
        self.assertEqual(_normalize_case_preview_value((1, 2)), "(1, 2)")


class BuildCasePreviewRecordsTests(unittest.TestCase):
    def test_trims_to_max_rows(self):
        records = [{"a": i} for i in range(5)]
        result = _build_case_preview_records(records, max_rows=2)
        self.assertEqual(result, [{"a": 0}, {"a": 1}])

    def test_normalizes_each_value(self):
        records = [{"x": "keep", "y": (1, 2)}]
        result = _build_case_preview_records(records, max_rows=5)
        self.assertEqual(result, [{"x": "keep", "y": "(1, 2)"}])

    def test_empty_input_returns_empty(self):
        self.assertEqual(_build_case_preview_records([], max_rows=3), [])


class SummarizeCasePreviewTableTests(unittest.TestCase):
    def test_empty_records_yields_empty_list(self):
        self.assertEqual(_summarize_case_preview_table("Bus", []), [])

    def test_prefers_known_columns(self):
        records = [
            {"idx": "Bus_1", "name": "a", "bus": 1, "extra": "z"},
            {"idx": "Bus_2", "name": "b", "bus": 2, "extra": "y"},
        ]
        lines = _summarize_case_preview_table("Bus", records)
        self.assertEqual(len(lines), 2)
        self.assertIn("Sheet Bus columns:", lines[0])
        self.assertIn("sample rows:", lines[1])
        # Sample rows should mention at least the known columns in JSON.
        self.assertIn('"idx"', lines[1])
        self.assertIn('"bus"', lines[1])

    def test_truncates_column_list_at_8(self):
        records = [{f"c{i}": i for i in range(12)}]
        lines = _summarize_case_preview_table("Bus", records)
        self.assertIn("...", lines[0])

    def test_falls_back_to_first_five_columns_when_none_known(self):
        records = [{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}]
        lines = _summarize_case_preview_table("Custom", records)
        self.assertIn("Sheet Custom columns:", lines[0])


class ResolveGeneratedImagePathTests(unittest.TestCase):
    def test_absolute_existing_image_path_returns_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "foo.png"
            img.write_bytes(b"fake")
            resolved = resolve_generated_image_path(str(img), tmp)
            self.assertEqual(resolved, str(img))

    def test_relative_path_resolves_under_session_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "plot_1.png"
            img.write_bytes(b"fake")
            resolved = resolve_generated_image_path("plot_1.png", tmp)
            self.assertEqual(resolved, str(img))

    def test_output_subfolder_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output"
            out.mkdir()
            img = out / "chart.png"
            img.write_bytes(b"fake")
            # Strip path, should find it under output/.
            resolved = resolve_generated_image_path("missing/chart.png", tmp)
            self.assertEqual(resolved, str(img))

    def test_empty_input_returns_none(self):
        self.assertIsNone(resolve_generated_image_path("", "/tmp"))
        self.assertIsNone(resolve_generated_image_path(None, "/tmp"))

    def test_strips_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "plot.png"
            img.write_bytes(b"fake")
            resolved = resolve_generated_image_path(f'"{img}"', tmp)
            self.assertEqual(resolved, str(img))

    def test_non_image_extension_not_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            txt = Path(tmp) / "data.txt"
            txt.write_text("hi")
            self.assertIsNone(resolve_generated_image_path(str(txt), tmp))


class ExtractGeneratedImagePathsTests(unittest.TestCase):
    def test_parses_saved_plots_section(self):
        # The "Saved plot(s):" header marks the start of a bulleted list
        # of newly-generated images. Any "- path" line under it is an
        # image to resolve.
        session_id = "test_extract_1"
        session_root = Path("code_executions") / session_id / "data"
        session_root.mkdir(parents=True, exist_ok=True)
        try:
            img = session_root / "plot_5.png"
            img.write_bytes(b"fake")
            output = "Saved plot(s):\n- plot_5.png\n"
            paths = extract_generated_image_paths(output, session_id)
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].endswith("plot_5.png"))
        finally:
            img.unlink(missing_ok=True)
            # clean up empty dirs we created
            session_root.rmdir()
            session_root.parent.rmdir()

    def test_no_headers_yields_empty(self):
        self.assertEqual(extract_generated_image_paths("random text", "s_id"), [])

    def test_empty_output_yields_empty(self):
        self.assertEqual(extract_generated_image_paths("", "s_id"), [])


class DirectoryScanningTests(unittest.TestCase):
    def test_get_files_in_directory_excludes_exec_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "keep.txt").write_text("a")
            Path(tmp, "exec_code_123.py").write_text("b")
            Path(tmp, "exec_code_XYZ.py").write_text("c")
            Path(tmp, "case.xlsx").write_text("d")
            result = get_files_in_directory(tmp)
            self.assertEqual(result, ["case.xlsx", "keep.txt"])

    def test_get_files_creates_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nested", "created")
            result = get_files_in_directory(missing)
            self.assertEqual(result, [])
            self.assertTrue(os.path.isdir(missing))

    def test_get_files_sorts_alphabetically(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ["zebra.txt", "apple.txt", "mango.txt"]:
                Path(tmp, name).write_text("x")
            self.assertEqual(
                get_files_in_directory(tmp),
                ["apple.txt", "mango.txt", "zebra.txt"],
            )

    def test_list_image_files_ignores_output_and_pycache(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "pic.png").write_bytes(b"1")
            out = Path(tmp, "output"); out.mkdir()
            Path(out, "ignored.png").write_bytes(b"2")
            pyc = Path(tmp, "__pycache__"); pyc.mkdir()
            Path(pyc, "ignored.png").write_bytes(b"3")
            sub = Path(tmp, "sub"); sub.mkdir()
            Path(sub, "chart.jpg").write_bytes(b"4")
            images = list_image_files(tmp)
            self.assertIn("pic.png", images)
            self.assertIn(os.path.join("sub", "chart.jpg"), images)
            self.assertEqual(len(images), 2)

    def test_list_image_files_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.png").write_bytes(b"1")
            Path(tmp, "b.JPG").write_bytes(b"2")  # case-insensitive
            Path(tmp, "c.txt").write_text("no")
            images = list_image_files(tmp)
            self.assertEqual(images, {"a.png", "b.JPG"})


class CopyNewImagesTests(unittest.TestCase):
    def test_copies_only_new_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "old.png").write_bytes(b"old")
            (base / "new.png").write_bytes(b"new")
            out = base / "output"; out.mkdir()
            before = {"old.png"}
            after = {"old.png", "new.png"}
            copied = copy_new_images_to_output(before, after, str(base), str(out))
            self.assertEqual(copied, ["new.png"])
            self.assertTrue((out / "new.png").exists())
            self.assertFalse((out / "old.png").exists())

    def test_counter_suffix_on_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "plot.png").write_bytes(b"new")
            out = base / "output"; out.mkdir()
            (out / "plot.png").write_bytes(b"preexisting")
            copied = copy_new_images_to_output(set(), {"plot.png"}, str(base), str(out))
            self.assertEqual(copied, ["plot_1.png"])
            self.assertTrue((out / "plot_1.png").exists())
            # Preexisting file untouched
            self.assertEqual((out / "plot.png").read_bytes(), b"preexisting")

    def test_skips_missing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "output"); out.mkdir()
            copied = copy_new_images_to_output(set(), {"ghost.png"}, tmp, str(out))
            self.assertEqual(copied, [])


class DeleteFileTests(unittest.TestCase):
    def test_deletes_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp, "to_delete.txt")
            f.write_text("bye")
            self.assertTrue(delete_file(str(f)))
            self.assertFalse(f.exists())

    def test_returns_false_for_missing_file(self):
        self.assertFalse(delete_file("/nonexistent/path/somefile.txt"))


class GetFileSizeTests(unittest.TestCase):
    def test_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp, "x.bin")
            f.write_bytes(b"a" * 500)
            self.assertEqual(get_file_size(str(f)), "500.0 B")

    def test_kilobytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp, "x.bin")
            f.write_bytes(b"a" * 2048)
            self.assertEqual(get_file_size(str(f)), "2.0 KB")

    def test_missing_file_returns_unknown(self):
        self.assertEqual(get_file_size("/nonexistent/file"), "Unknown")


class InferAndesCaseContextTests(unittest.TestCase):
    def _tmp_runtime(self, files=()):
        tmp = tempfile.mkdtemp()
        for name in files:
            Path(tmp, name).write_text("x")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        return tmp

    def test_no_load_call_returns_none(self):
        self.assertIsNone(infer_andes_case_context("x = 1", "/tmp"))

    def test_get_case_returns_builtin(self):
        runtime = self._tmp_runtime()
        code = 'import andes\nssa = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True)'
        ctx = infer_andes_case_context(code, runtime)
        self.assertEqual(ctx, {"source": "builtin", "value": "ieee14/ieee14_full.xlsx"})

    def test_last_get_case_wins(self):
        runtime = self._tmp_runtime()
        code = (
            'first = andes.get_case("a.xlsx")\n'
            'second = andes.get_case("b.xlsx")\n'
            'ssa = andes.load(second)'
        )
        ctx = infer_andes_case_context(code, runtime)
        self.assertEqual(ctx["value"], "b.xlsx")

    def test_uploaded_case_via_join(self):
        # NB: the join-regex ([^)]*) stops at the first ")", so its
        # left-hand argument has to be a bare identifier like
        # script_dir -- a nested call like os.getcwd() would break it.
        # This matches the template emitted by the fallback generators.
        runtime = self._tmp_runtime(files=["network.xlsx"])
        code = (
            'import os, andes\n'
            'script_dir = os.getcwd()\n'
            'case = os.path.join(script_dir, "network.xlsx")\n'
            'ssa = andes.load(case, setup=True)'
        )
        ctx = infer_andes_case_context(code, runtime)
        self.assertEqual(ctx, {"source": "uploaded", "value": "network.xlsx"})

    def test_local_case_not_in_runtime_dir(self):
        runtime = self._tmp_runtime()
        code = (
            'case = "/abs/path/stuff.xlsx"\n'
            'ssa = andes.load(case, setup=True)'
        )
        ctx = infer_andes_case_context(code, runtime)
        self.assertEqual(ctx, {"source": "local", "value": "/abs/path/stuff.xlsx"})

    def test_literal_load_uploaded(self):
        runtime = self._tmp_runtime(files=["kundur.xlsx"])
        code = 'ssa = andes.load("kundur.xlsx", setup=True)'
        ctx = infer_andes_case_context(code, runtime)
        self.assertEqual(ctx, {"source": "uploaded", "value": "kundur.xlsx"})


class SummarizeAndesCaseFileTests(unittest.TestCase):
    def test_json_case_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp, "case.json")
            case.write_text(json.dumps({
                "Bus": [
                    {"idx": "Bus_1", "name": "n1", "v0": 1.0},
                    {"idx": "Bus_2", "name": "n2", "v0": 1.0},
                ],
                "PQ": [{"idx": "PQ_1", "bus": 1, "p0": 0.3}],
            }))
            summary = summarize_andes_case_file(str(case))
            self.assertIn("Selected ANDES case preview for case.json", summary)
            self.assertIn("Sheet Bus columns:", summary)
            self.assertIn("Sheet PQ columns:", summary)

    def test_json_without_known_sections_yields_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp, "case.json")
            case.write_text(json.dumps({"meta": {}, "stuff": []}))
            self.assertEqual(summarize_andes_case_file(str(case)), "")

    def test_bad_file_returns_empty(self):
        self.assertEqual(summarize_andes_case_file("/nonexistent.json"), "")

    def test_malformed_json_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp, "bad.json")
            f.write_text("{not-json")
            self.assertEqual(summarize_andes_case_file(str(f)), "")


if __name__ == "__main__":
    unittest.main()
