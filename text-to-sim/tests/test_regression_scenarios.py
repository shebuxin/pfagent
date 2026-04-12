import sys
import unittest
from pathlib import Path


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from scripts.andes_regression_check import build_scenarios, get_runtime_upload_sources


class AndesRegressionScenarioTests(unittest.TestCase):
    def test_strict_regression_suite_contains_20_unique_scenarios(self):
        scenarios = build_scenarios()
        scenario_names = [scenario["name"] for scenario in scenarios]

        self.assertEqual(20, len(scenarios))
        self.assertEqual(len(scenario_names), len(set(scenario_names)))

    def test_uploaded_scenarios_reference_known_runtime_files(self):
        runtime_sources = get_runtime_upload_sources()

        for scenario in build_scenarios():
            for uploaded_file in scenario.get("uploaded_files", []):
                self.assertIn(uploaded_file, runtime_sources, msg=f"Missing runtime source for {uploaded_file}")


if __name__ == "__main__":
    unittest.main()
