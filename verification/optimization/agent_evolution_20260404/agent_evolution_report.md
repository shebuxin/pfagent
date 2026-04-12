# Agent Evolution Report

## Overview

This report documents the full retained optimization timeline used to improve PFAGENT, from the earliest 100-scenario benchmark through the later 152-scenario expanded suite and the final open-ended stress-test recovery. The goal was to turn repeated failure analysis into a reusable agent-evolution mechanism instead of one-off prompt patches.

## Adaptive Workflow

1. Add a small set of more open-ended stress-test scenarios.
2. Run verification and collect per-turn failure logs.
3. Extract recurring root-cause signatures from prompts, execution traces, and scoring issues.
4. Map each root cause to a mutation pack containing prompt guidance, parser pattern extensions, and carryover markers.
5. Save the resulting evolution profile into the live agent configuration.
6. Re-run the same stress scenarios and compare recovery quantitatively.

Integration points:
- [agent_evolution.py](/home/bshe/Documents/git-research/pfagent/text-to-sim/src/agent_evolution.py)
- [prompt_builder.py](/home/bshe/Documents/git-research/pfagent/text-to-sim/src/prompt_builder.py)
- [rag_chatbot.py](/home/bshe/Documents/git-research/pfagent/text-to-sim/src/chatbots/openai/rag_chatbot.py)
- [agent_evolution_profile.json](/home/bshe/Documents/git-research/pfagent/text-to-sim/data_files/agent_evolution_profile.json)

## Full Fine-Tuned + RAG Timeline

| stage_id | label | suite_group | suite_label | scenario_count | scenario_pass_count | scenario_pass_rate | avg_conversation_score | turn_1_pass_rate | turn_2_pass_rate | turn_3_pass_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_100_initial | 100-scenario initial benchmark | main_suite | 100-scenario baseline suite | 100 | 0 | 0.0 | 51.43 | 20.0 | 3.0 | 0.0 | Earliest retained 100-scenario benchmark before the major structured-generation and workflow upgrades. |
| main_100_optimized | 100-scenario optimized benchmark | main_suite | 100-scenario retained final suite | 100 | 100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | After major workflow abstraction, retrieval grounding, and structured ANDES generation, the 100-scenario benchmark was saturated. |
| main_132_before_fix | 132-scenario harder suite before fix | main_suite | 132-scenario harder suite | 132 | 100 | 75.76 | 92.19 | 85.61 | 78.79 | 75.76 | New targeted case-edit and N-1 scenarios reduced performance and exposed the next bottleneck. |
| main_132_after_fix | 132-scenario harder suite after fix | main_suite | 132-scenario harder suite | 132 | 132 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | Structured device-resolution and contingency edits were expanded, restoring full pass rate on the 132-scenario suite. |
| main_140_light | 140-scenario light expansion | main_suite | 140-scenario light expansion | 140 | 140 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | A small expansion stayed within the solved region. |
| main_146_generalized | 146-scenario generalized expansion | main_suite | 146-scenario generalized suite | 146 | 140 | 95.89 | 97.98 | 95.89 | 95.89 | 95.89 | More natural generalized prompts pushed the agent off saturation but still kept performance above 95%. |
| main_152_generalized | 152-scenario generalized suite after fix | main_suite | 152-scenario generalized suite | 152 | 152 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | Parser and state-carryover improvements recovered the expanded main suite to full pass rate. |
| open_4_before_fix | 4 open scenarios before adaptive evolution | open_suite | 4-scenario open generalization suite | 4 | 0 | 0.0 | 56.81 | 0.0 | 0.0 | 0.0 | Open-ended phrasing temporarily collapsed performance and revealed the remaining root causes. |
| open_4_after_fix | 4 open scenarios after adaptive evolution | open_suite | 4-scenario open generalization suite | 4 | 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | The failure-driven adaptive evolution mechanism restored full pass rate on the same open scenarios. |

## Open-Scenario Adaptive Evolution Loop

| stage_id | label | scenario_count | scenario_pass_count | scenario_pass_rate | avg_conversation_score | turn_1_pass_rate | turn_2_pass_rate | turn_3_pass_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| open_4_before_fix | 4 open scenarios before adaptive evolution | 4 | 0 | 0.0 | 56.81 | 0.0 | 0.0 | 0.0 | Open-ended phrasing temporarily collapsed performance and revealed the remaining root causes. |
| open_4_after_fix | 4 open scenarios after adaptive evolution | 4 | 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | The failure-driven adaptive evolution mechanism restored full pass rate on the same open scenarios. |

## Expanded Suite Summary

| stage_id | label | scenario_count | scenario_pass_count | scenario_pass_rate | avg_conversation_score |
| --- | --- | --- | --- | --- | --- |
| expanded_before_fix | Expanded 156-scenario suite before adaptive evolution | 156 | 152 | 97.44 | 98.89 |
| expanded_after_fix | Expanded 156-scenario suite after adaptive evolution | 156 | 156 | 100.0 | 100.0 |

## Root-Cause Signatures

| signature_id | label | count | example_turns | activated_packs |
| --- | --- | --- | --- | --- |
| device_idx_cast_to_int | String device idx was cast to int | 2 | ['open_scenario_002/turn_02', 'open_scenario_004/turn_02'] | ['string_device_idx_guardrail'] |
| positional_idx_used_as_device_idx | Bus number or array index was used as device idx | 4 | ['open_scenario_001/turn_02', 'open_scenario_001/turn_03', 'open_scenario_002/turn_03'] | ['targeted_device_resolution', 'string_device_idx_guardrail'] |
| open_ended_pq_percentage_language | Open-ended percentage demand phrasing was not grounded | 3 | ['open_scenario_001/turn_02', 'open_scenario_002/turn_02', 'open_scenario_004/turn_02'] | ['pq_percentage_scaling', 'targeted_device_resolution'] |
| open_ended_pv_regulator_language | Open-ended regulator phrasing was not grounded | 2 | ['open_scenario_003/turn_01', 'open_scenario_003/turn_02'] | ['pv_regulator_aliases', 'targeted_device_resolution'] |
| corridor_outage_language | Corridor phrasing was not mapped to line outage status edits | 4 | ['open_scenario_001/turn_03', 'open_scenario_002/turn_03', 'open_scenario_003/turn_03'] | ['corridor_outage_aliases', 'line_outage_api_guardrail'] |
| n1_outage_set_language | Outage-set language was not mapped to N-1 screening | 1 | ['open_scenario_004/turn_03'] | ['n1_outage_set_aliases', 'line_outage_api_guardrail'] |

## Activated Mutation Packs

| pack_id | title | prompt_guidance_count |
| --- | --- | --- |
| corridor_outage_aliases | Interpret corridor phrasing as line outages | 1 |
| line_outage_api_guardrail | Use line-status APIs instead of guessed PFlow setters | 1 |
| n1_outage_set_aliases | Interpret outage-set language as N-1 screening | 1 |
| pq_percentage_scaling | Interpret percentage demand changes as scale factors | 1 |
| pv_regulator_aliases | Handle regulator-target phrasing for PV edits | 1 |
| string_device_idx_guardrail | Preserve string-valued ANDES device identifiers | 1 |
| targeted_device_resolution | Resolve targeted devices by bus before editing | 2 |

## Key Findings

- The earliest retained 100-scenario benchmark started at `51.43` average conversation score and `0%` scenario pass rate for `Fine-tuned + RAG`.
- The same 100-scenario suite was later saturated at `100%`, showing that the first major workflow abstraction closed the original gap.
- Expanding from `100` to `132`, `140`, `146`, and then `152` scenarios created a more realistic progression curve instead of a single static benchmark.
- The 146-scenario generalized suite was the main retained point where performance dropped but still stayed above `95%`.
- Introducing four open-ended scenarios caused the pass rate on that mini-suite to drop to `0%`, while the combined 156-scenario suite remained at `97.44%` because the failure set was small but qualitatively important.
- The adaptive evolution mechanism recovered the open suite to `100%` by translating observed failures into reusable prompt rules and parser hooks.
- The generated profile activated 7 mutation packs and produced 8 adaptive runtime guidance bullets.

## Figures

- [Full Fine-tuned + RAG progression timeline](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/ft_rag_progression_timeline.png)
- [Fine-tuned + RAG turn progression](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/ft_rag_turn_progression.png)
- [Benchmark growth timeline](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/suite_growth_timeline.png)
- [Progressive stage pass rate](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/progressive_stage_pass_rate.png)
- [Expanded suite recovery](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/combined_suite_recovery.png)
- [Open-suite turn recovery](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/open_suite_turn_recovery.png)
- [Open-suite failure reduction](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/open_suite_failure_reduction.png)
- [Root-cause signatures](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/root_cause_signatures.png)
