import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.agent_evolution import (
    build_adaptive_guidance_section,
    build_evolution_profile_from_failures,
    build_evolution_profile_from_results,
    get_profile_marker_overrides,
    get_profile_pattern_overrides,
    merge_evolution_profiles,
    save_evolution_profile,
)
from src.andes_manual import RAG_ANDES_MANUAL_POLICY
from src.prompt_builder import AndesPromptBuilderConfig, AndesSystemPromptBuilder


class AgentEvolutionTests(unittest.TestCase):
    def test_build_evolution_profile_activates_expected_packs(self):
        synthetic_results = {
            "models": {
                "fine_tuned_rag": [
                    {
                        "scenario_id": "open_scenario_001",
                        "turns": [
                            {
                                "turn_id": 1,
                                "prompt": "Make that same demand record 4% heavier and rerun power flow.",
                                "turn_passed": False,
                                "execution_output": "ValueError: invalid literal for int() with base 10: np.str_('PQ_2')",
                                "issues": [],
                            },
                            {
                                "turn_id": 2,
                                "prompt": "Raise that regulator target to 1.02 and rerun.",
                                "turn_passed": False,
                                "execution_output": "",
                                "issues": ["pv_bus: expected 3, got 2"],
                            },
                            {
                                "turn_id": 3,
                                "prompt": "Use this outage set for a stressed case N-1 screen and put the transmission corridor between buses 2 and 3 out of service.",
                                "turn_passed": False,
                                "execution_output": "AttributeError: 'PFlow' object has no attribute 'set'",
                                "issues": [],
                            },
                        ],
                    }
                ]
            }
        }

        with tempfile.TemporaryDirectory(prefix="pfagent-agent-evolution-") as tmp:
            results_path = Path(tmp) / "verification_results.json"
            results_path.write_text(json.dumps(synthetic_results), encoding="utf-8")

            profile = build_evolution_profile_from_results(
                [results_path],
                profile_version="synthetic-v1",
            )

        self.assertEqual("synthetic-v1", profile["profile_version"])
        self.assertIn("string_device_idx_guardrail", profile["active_mutation_packs"])
        self.assertIn("pq_percentage_scaling", profile["active_mutation_packs"])
        self.assertIn("pv_regulator_aliases", profile["active_mutation_packs"])
        self.assertIn("corridor_outage_aliases", profile["active_mutation_packs"])
        self.assertIn("n1_outage_set_aliases", profile["active_mutation_packs"])
        self.assertIn("line_outage_api_guardrail", profile["active_mutation_packs"])
        self.assertTrue(profile["prompt_guidance"])
        signature_counts = {item["signature_id"]: item["count"] for item in profile["root_cause_summary"]}
        self.assertEqual(1, signature_counts["device_idx_cast_to_int"])
        self.assertEqual(1, signature_counts["open_ended_pq_percentage_language"])
        self.assertEqual(1, signature_counts["open_ended_pv_regulator_language"])
        self.assertEqual(1, signature_counts["corridor_outage_language"])
        self.assertEqual(1, signature_counts["n1_outage_set_language"])

    def test_profile_overrides_extend_defaults_and_prompt_builder_reads_guidance(self):
        profile = {
            "profile_version": "test-profile",
            "prompt_guidance": ["Use the adaptive rule."],
            "pattern_overrides": {"target_pq_bus": [r"demand object on bus (\d+)"]},
            "marker_overrides": {"structured_activation_markers": ["narrative stress case"]},
        }

        with tempfile.TemporaryDirectory(prefix="pfagent-agent-profile-") as tmp:
            profile_path = save_evolution_profile(profile, Path(tmp) / "agent_profile.json")
            patterns = get_profile_pattern_overrides(
                "target_pq_bus",
                [r"existing pq load at bus (\d+)"],
                profile_path=profile_path,
            )
            markers = get_profile_marker_overrides(
                "structured_activation_markers",
                ["power flow"],
                profile_path=profile_path,
            )

            self.assertIn(r"existing pq load at bus (\d+)", patterns)
            self.assertIn(r"demand object on bus (\d+)", patterns)
            self.assertIn("power flow", markers)
            self.assertIn("narrative stress case", markers)
            self.assertIn("Use the adaptive rule.", build_adaptive_guidance_section(profile_path))

            builder = AndesSystemPromptBuilder(
                AndesPromptBuilderConfig(
                    include_context_placeholder=True,
                    include_tools_info=True,
                    enforce_code_only_fence_rule=True,
                    include_andes_guardrails=True,
                    ban_test_wrappers=True,
                )
            )
            with patch("src.prompt_builder.build_adaptive_guidance_section", return_value="Adaptive Evolution Rules:\n- Use the adaptive rule."):
                prompt = builder.build_prompt(
                    andes_manual_policy=RAG_ANDES_MANUAL_POLICY,
                    tools_info="TOOLS_SECTION",
                    custom_instructions="Custom line.",
                    few_shot_section="FEW_SHOT_SECTION",
                )

            self.assertIn("Adaptive Evolution Rules:", prompt)
            self.assertIn("Use the adaptive rule.", prompt)

    def test_build_profile_from_failures_and_merge_profiles(self):
        failure_records = [
            {
                "scenario_id": "user_session_alpha",
                "turn_id": 1,
                "prompt": "Make that same demand record 4% heavier.",
                "execution_output": "ValueError: invalid literal for int() with base 10: 'PQ_2'",
                "issues": ["wrong device idx / inspect the case", "response not runnable"],
                "turn_passed": False,
            }
        ]
        delta = build_evolution_profile_from_failures(
            failure_records,
            profile_version="runtime-feedback-v1",
            source_runs=["session_alpha.json"],
        )
        self.assertIn("string_device_idx_guardrail", delta["active_mutation_packs"])
        self.assertIn("pq_percentage_scaling", delta["active_mutation_packs"])
        self.assertIn("runnable_code_contract", delta["active_mutation_packs"])

        merged = merge_evolution_profiles(
            {
                "profile_version": "base",
                "source_runs": ["old_run.json"],
                "active_mutation_packs": ["targeted_device_resolution"],
                "prompt_guidance": ["Base guidance."],
                "pattern_overrides": {},
                "marker_overrides": {},
                "root_cause_summary": [
                    {
                        "signature_id": "positional_idx_used_as_device_idx",
                        "label": "Bus number or array index was used as device idx",
                        "count": 2,
                        "example_turns": ["old/turn_01"],
                        "activated_packs": ["targeted_device_resolution"],
                    }
                ],
            },
            delta,
            profile_version="merged-v1",
        )
        self.assertEqual("merged-v1", merged["profile_version"])
        self.assertIn("old_run.json", merged["source_runs"])
        self.assertIn("session_alpha.json", merged["source_runs"])
        signature_counts = {item["signature_id"]: item["count"] for item in merged["root_cause_summary"]}
        self.assertEqual(3, signature_counts["positional_idx_used_as_device_idx"])
        self.assertEqual(1, signature_counts["device_idx_cast_to_int"])
        self.assertEqual(1, signature_counts["response_not_runnable"])


if __name__ == "__main__":
    unittest.main()
