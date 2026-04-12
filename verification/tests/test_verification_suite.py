import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verification.oracle import compute_oracle_for_scenario
from verification.suite import (
    FULL_SUITE_SCENARIO_COUNT,
    OPEN_GENERALIZATION_SCENARIO_COUNT,
    build_open_generalization_suite,
    build_verification_suite,
)


class VerificationSuiteTests(unittest.TestCase):
    def test_suite_contains_full_unique_scenarios(self):
        scenarios = build_verification_suite()
        scenario_ids = [scenario["scenario_id"] for scenario in scenarios]

        self.assertEqual(FULL_SUITE_SCENARIO_COUNT, len(scenarios))
        self.assertEqual(FULL_SUITE_SCENARIO_COUNT, len(set(scenario_ids)))

    def test_every_scenario_has_three_turns_and_two_follow_up_modifications(self):
        scenarios = build_verification_suite()

        for scenario in scenarios:
            self.assertEqual(3, len(scenario["turns"]))
            self.assertEqual([], scenario["turns"][0]["current_operations"])
            self.assertTrue(scenario["turns"][1]["current_operations"])
            self.assertTrue(scenario["turns"][2]["current_operations"])
            for turn in scenario["turns"]:
                self.assertTrue(turn["result_keys"])
                self.assertIn("RESULT_JSON=", turn["prompt"])

    def test_oracle_keys_match_turn_contract_for_builtin_and_uploaded_samples(self):
        scenarios = build_verification_suite()
        sample_builtin = next(item for item in scenarios if item["case_source"] == "builtin")
        sample_uploaded = next(item for item in scenarios if item["case_source"] == "uploaded")

        for scenario in [sample_builtin, sample_uploaded]:
            oracle_results = compute_oracle_for_scenario(scenario)
            self.assertEqual(3, len(oracle_results))
            for turn, oracle in zip(scenario["turns"], oracle_results):
                self.assertEqual(set(turn["result_keys"]), set(oracle.keys()))
                if turn["plot_filename"]:
                    self.assertEqual(turn["plot_filename"], oracle["plot_file"])

    def test_suite_includes_targeted_case_edit_and_n1_blueprints(self):
        scenarios = build_verification_suite()
        blueprint_names = {scenario["blueprint"] for scenario in scenarios}

        self.assertIn("targeted_pq_edit_then_n1_screening", blueprint_names)
        self.assertIn("failure_aware_targeted_pq_then_n1_screening", blueprint_names)
        self.assertIn("targeted_pv_edit_then_line_outage", blueprint_names)
        self.assertIn("targeted_pq_scale_then_line_outage_threshold", blueprint_names)
        self.assertIn("generalized_targeted_pq_then_branch_trip", blueprint_names)
        self.assertIn("generalized_targeted_pv_then_branch_trip", blueprint_names)
        self.assertIn("open_story_pq_branch_trip", blueprint_names)
        self.assertIn("open_story_pv_branch_trip", blueprint_names)
        self.assertIn("open_story_targeted_n1_screening", blueprint_names)
        self.assertIn("open_story_failure_aware_n1_screening", blueprint_names)

        n1_scenario = next(
            item for item in scenarios if item["blueprint"] == "targeted_pq_edit_then_n1_screening"
        )
        pv_line_scenario = next(
            item for item in scenarios if item["blueprint"] == "targeted_pv_edit_then_line_outage"
        )
        pq_outage_threshold_scenario = next(
            item for item in scenarios if item["blueprint"] == "targeted_pq_scale_then_line_outage_threshold"
        )
        failure_aware_n1_scenario = next(
            item for item in scenarios if item["blueprint"] == "failure_aware_targeted_pq_then_n1_screening"
        )
        generalized_pq_trip_scenario = next(
            item for item in scenarios if item["blueprint"] == "generalized_targeted_pq_then_branch_trip"
        )
        generalized_pv_trip_scenario = next(
            item for item in scenarios if item["blueprint"] == "generalized_targeted_pv_then_branch_trip"
        )

        for scenario in [
            n1_scenario,
            failure_aware_n1_scenario,
            pv_line_scenario,
            pq_outage_threshold_scenario,
            generalized_pq_trip_scenario,
            generalized_pv_trip_scenario,
        ]:
            oracle_results = compute_oracle_for_scenario(scenario)
            for turn, oracle in zip(scenario["turns"], oracle_results):
                self.assertEqual(set(turn["result_keys"]), set(oracle.keys()))

    def test_open_generalization_suite_is_small_and_oracle_consistent(self):
        scenarios = build_open_generalization_suite()

        self.assertEqual(OPEN_GENERALIZATION_SCENARIO_COUNT, len(scenarios))
        self.assertEqual(OPEN_GENERALIZATION_SCENARIO_COUNT, len({item["scenario_id"] for item in scenarios}))

        blueprint_names = {scenario["blueprint"] for scenario in scenarios}
        self.assertIn("open_story_pq_branch_trip", blueprint_names)
        self.assertIn("open_story_pv_branch_trip", blueprint_names)
        self.assertIn("open_story_targeted_n1_screening", blueprint_names)
        self.assertIn("open_story_failure_aware_n1_screening", blueprint_names)

        for scenario in scenarios:
            oracle_results = compute_oracle_for_scenario(scenario)
            self.assertEqual(3, len(oracle_results))
            for turn, oracle in zip(scenario["turns"], oracle_results):
                self.assertEqual(set(turn["result_keys"]), set(oracle.keys()))


if __name__ == "__main__":
    unittest.main()
