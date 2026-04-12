import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


sys.modules.setdefault("PyPDF2", types.SimpleNamespace(PdfReader=object))

TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src import andes_manual


FAKE_MANUAL_PAGES = (
    (1, "Introduction and overview of ANDES workflows."),
    (2, 'Built-in cases can be loaded with andes.load(andes.get_case("ieee14/ieee14.raw"), setup=True, no_output=True, log=False).'),
    (3, "After loading a built-in case, call ssa.PFlow.run() and inspect ssa.Bus.v.v for bus voltages."),
    (4, "Optional model edits can be made before setup when the workflow requires it."),
    (5, 'For uploaded cases, use script_dir = os.getcwd() and case = os.path.join(script_dir, "uploaded_ieee39.xlsx").'),
    (6, "Then run ssa = andes.load(case, setup=True, no_output=True, log=False) followed by ssa.PFlow.run()."),
    (7, "Voltage profile plotting can be done with matplotlib after results are available."),
)


class AndesManualRetrievalTests(unittest.TestCase):
    def test_builtin_case_query_retrieves_manual_window(self):
        query = "Use an ANDES built-in IEEE 14 case, run power flow, and print slack bus voltage."
        with patch.object(andes_manual, "load_andes_manual_pages", return_value=FAKE_MANUAL_PAGES):
            windows = andes_manual.retrieve_relevant_andes_manual_windows(
                query,
                window_pages=3,
                max_windows=1,
                max_chars_per_window=5000,
            )

        self.assertEqual(len(windows), 1)
        self.assertIn('ieee14/ieee14.raw', windows[0]["content"])
        self.assertIn("ssa.PFlow.run()", windows[0]["content"])

    def test_uploaded_case_query_prefers_exact_filename(self):
        query = "Use my uploaded file uploaded_ieee39.xlsx, run power flow, and plot the voltage profile."
        with patch.object(andes_manual, "load_andes_manual_pages", return_value=FAKE_MANUAL_PAGES):
            windows = andes_manual.retrieve_relevant_andes_manual_windows(
                query,
                window_pages=3,
                max_windows=1,
                max_chars_per_window=5000,
            )

        self.assertEqual(len(windows), 1)
        self.assertIn("uploaded_ieee39.xlsx", windows[0]["content"])
        self.assertIn("andes.load(case", windows[0]["content"])

    def test_runtime_context_noise_is_ignored_for_manual_search(self):
        query = (
            "Generate runnable Python code only. Use an ANDES built-in IEEE 14 case and print slack bus voltage.\n\n"
            "Runtime file context:\n"
            "- Working directory for execution: ./code_executions/example/data\n"
            "- Uploaded files available during execution:\n"
            "- uploaded_ieee39.xlsx\n"
            "- Use these filenames directly in generated Python code when needed.\n"
        )
        with patch.object(andes_manual, "load_andes_manual_pages", return_value=FAKE_MANUAL_PAGES):
            windows = andes_manual.retrieve_relevant_andes_manual_windows(
                query,
                window_pages=3,
                max_windows=1,
                max_chars_per_window=5000,
            )

        self.assertEqual(len(windows), 1)
        self.assertIn('ieee14/ieee14.raw', windows[0]["content"])
        self.assertNotIn("uploaded_ieee39.xlsx", windows[0]["content"])


if __name__ == "__main__":
    unittest.main()
