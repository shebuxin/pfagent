import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.files import build_uploaded_case_prompt_context


class CasePreviewContextTests(unittest.TestCase):
    def _write_sample_case(self, target: Path) -> None:
        with pd.ExcelWriter(target) as writer:
            pd.DataFrame(
                [
                    {"uid": 0, "idx": 1, "name": "BUS1", "v0": 1.03},
                    {"uid": 1, "idx": 31, "name": "BUS31", "v0": 1.04},
                ]
            ).to_excel(writer, sheet_name="Bus", index=False)
            pd.DataFrame(
                [
                    {"uid": 0, "idx": 2, "bus": 31, "p0": 5.72, "q0": 4.29, "v0": 1.04},
                    {"uid": 1, "idx": 3, "bus": 32, "p0": 6.5, "q0": 2.1, "v0": 0.99},
                ]
            ).to_excel(writer, sheet_name="PV", index=False)
            pd.DataFrame(
                [
                    {"uid": 0, "idx": "PQ_1", "bus": 3, "p0": 6.0, "q0": 2.5},
                ]
            ).to_excel(writer, sheet_name="PQ", index=False)
            pd.DataFrame(
                [
                    {"uid": 0, "idx": "Line_1", "bus1": 1, "bus2": 31, "tap": 1.0},
                ]
            ).to_excel(writer, sheet_name="Line", index=False)

    def test_build_uploaded_case_prompt_context_summarizes_selected_excel_case(self):
        with tempfile.TemporaryDirectory(prefix="pfagent-case-preview-") as tmp:
            case_path = Path(tmp) / "review_case.xlsx"
            self._write_sample_case(case_path)

            context = build_uploaded_case_prompt_context(
                tmp,
                user_input="Modify the PV at bus 31 in review_case.xlsx and rerun power flow.",
                active_case=None,
            )

            self.assertIn("Selected ANDES case preview for review_case.xlsx", context)
            self.assertIn("Sheet PV sample rows", context)
            self.assertIn('"idx": 2', context)
            self.assertIn('"bus": 31', context)

    def test_build_uploaded_case_prompt_context_falls_back_to_active_case(self):
        with tempfile.TemporaryDirectory(prefix="pfagent-case-preview-") as tmp:
            case_path = Path(tmp) / "followup_case.xlsx"
            self._write_sample_case(case_path)

            context = build_uploaded_case_prompt_context(
                tmp,
                user_input="Keep modifying the same case and adjust the PV device.",
                active_case={"source": "uploaded", "value": "followup_case.xlsx"},
            )

            self.assertIn("Selected ANDES case preview for followup_case.xlsx", context)


if __name__ == "__main__":
    unittest.main()
