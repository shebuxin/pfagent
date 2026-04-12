from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_TO_SIM_ROOT = REPO_ROOT / "text-to-sim"
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.agent_evolution import (
    DEFAULT_PROFILE_PATH,
    MUTATION_LIBRARY,
    build_evolution_profile_from_results,
    save_evolution_profile,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "verification" / "optimization" / "agent_evolution_20260404"
MODEL_KEY = "fine_tuned_rag"
MODEL_LABEL = "Fine-tuned + RAG"
OPEN_LOOP_STAGES = [
    {
        "stage_id": "structured_baseline",
        "label": "Stage 1: Structured Benchmark Before Open Stress Test",
        "suite_label": "152-scenario structured suite",
        "results_path": REPO_ROOT
        / "verification"
        / "artifacts_recheck_20260404_152_generalized"
        / "fine_tuned_rag"
        / "20260404_100400"
        / "verification_results.json",
        "notes": "The current structured benchmark was fully solved, but it still used stronger RESULT_JSON-style task contracts.",
    },
    {
        "stage_id": "open_generalization_gap",
        "label": "Stage 2: Open Scenario Stress Test Before Adaptive Evolution",
        "suite_label": "4-scenario open generalization suite",
        "results_path": REPO_ROOT
        / "verification"
        / "artifacts_open_generalization_20260404"
        / "fine_tuned_rag"
        / "20260404_102736"
        / "verification_results.json",
        "notes": "Four more open-ended scenarios were introduced to expose natural-language grounding gaps in case modification and contingency analysis.",
    },
    {
        "stage_id": "open_generalization_recovered",
        "label": "Stage 3: Open Scenario Stress Test After Adaptive Evolution",
        "suite_label": "4-scenario open generalization suite",
        "results_path": REPO_ROOT
        / "verification"
        / "artifacts_open_generalization_20260404"
        / "fine_tuned_rag_after_fix"
        / "20260404_103919"
        / "verification_results.json",
        "notes": "The same four open scenarios were re-run after integrating the adaptive evolution mechanism into the agent workflow.",
    },
]

FULL_PROGRESS_STAGES = [
    {
        "stage_id": "main_100_initial",
        "label": "100-scenario initial benchmark",
        "suite_group": "main_suite",
        "suite_label": "100-scenario baseline suite",
        "notes": "Earliest retained 100-scenario benchmark before the major structured-generation and workflow upgrades.",
        "model_summary_source": {
            "kind": "git",
            "rev": "d3e11d266",
            "path": "verification/artifacts_parallel/combined_20260328/reports/model_summary.json",
        },
        "turn_summary_source": {
            "kind": "git",
            "rev": "d3e11d266",
            "path": "verification/artifacts_parallel/combined_20260328/reports/tables/turn_summary.csv",
        },
    },
    {
        "stage_id": "main_100_optimized",
        "label": "100-scenario optimized benchmark",
        "suite_group": "main_suite",
        "suite_label": "100-scenario retained final suite",
        "notes": "After major workflow abstraction, retrieval grounding, and structured ANDES generation, the 100-scenario benchmark was saturated.",
        "model_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "final" / "reports" / "model_summary.json",
        },
        "turn_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "final" / "reports" / "tables" / "turn_summary.csv",
        },
    },
    {
        "stage_id": "main_132_before_fix",
        "label": "132-scenario harder suite before fix",
        "suite_group": "main_suite",
        "suite_label": "132-scenario harder suite",
        "notes": "New targeted case-edit and N-1 scenarios reduced performance and exposed the next bottleneck.",
        "model_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_recheck_20260403_132_parallel" / "fine_tuned_rag" / "20260403_184605" / "reports" / "model_summary.json",
        },
        "turn_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_recheck_20260403_132_parallel" / "fine_tuned_rag" / "20260403_184605" / "reports" / "tables" / "turn_summary.csv",
        },
    },
    {
        "stage_id": "main_132_after_fix",
        "label": "132-scenario harder suite after fix",
        "suite_group": "main_suite",
        "suite_label": "132-scenario harder suite",
        "notes": "Structured device-resolution and contingency edits were expanded, restoring full pass rate on the 132-scenario suite.",
        "model_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_recheck_20260403_132_optimized" / "fine_tuned_rag" / "20260403_203014" / "reports" / "model_summary.json",
        },
        "turn_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_recheck_20260403_132_optimized" / "fine_tuned_rag" / "20260403_203014" / "reports" / "tables" / "turn_summary.csv",
        },
    },
    {
        "stage_id": "main_140_light",
        "label": "140-scenario light expansion",
        "suite_group": "main_suite",
        "suite_label": "140-scenario light expansion",
        "notes": "A small expansion stayed within the solved region.",
        "model_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_recheck_20260403_140_light" / "fine_tuned_rag" / "20260403_210841" / "reports" / "model_summary.json",
        },
        "turn_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_recheck_20260403_140_light" / "fine_tuned_rag" / "20260403_210841" / "reports" / "tables" / "turn_summary.csv",
        },
    },
    {
        "stage_id": "main_146_generalized",
        "label": "146-scenario generalized expansion",
        "suite_group": "main_suite",
        "suite_label": "146-scenario generalized suite",
        "notes": "More natural generalized prompts pushed the agent off saturation but still kept performance above 95%.",
        "model_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_recheck_20260403_146_generalized" / "fine_tuned_rag" / "20260404_093100" / "reports" / "model_summary.json",
        },
        "turn_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_recheck_20260403_146_generalized" / "fine_tuned_rag" / "20260404_093100" / "reports" / "tables" / "turn_summary.csv",
        },
    },
    {
        "stage_id": "main_152_generalized",
        "label": "152-scenario generalized suite after fix",
        "suite_group": "main_suite",
        "suite_label": "152-scenario generalized suite",
        "notes": "Parser and state-carryover improvements recovered the expanded main suite to full pass rate.",
        "model_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_recheck_20260404_152_generalized" / "fine_tuned_rag" / "20260404_100400" / "reports" / "model_summary.json",
        },
        "turn_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_recheck_20260404_152_generalized" / "fine_tuned_rag" / "20260404_100400" / "reports" / "tables" / "turn_summary.csv",
        },
    },
    {
        "stage_id": "open_4_before_fix",
        "label": "4 open scenarios before adaptive evolution",
        "suite_group": "open_suite",
        "suite_label": "4-scenario open generalization suite",
        "notes": "Open-ended phrasing temporarily collapsed performance and revealed the remaining root causes.",
        "model_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_open_generalization_20260404" / "fine_tuned_rag" / "20260404_102736" / "reports" / "model_summary.json",
        },
        "turn_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_open_generalization_20260404" / "fine_tuned_rag" / "20260404_102736" / "reports" / "tables" / "turn_summary.csv",
        },
    },
    {
        "stage_id": "open_4_after_fix",
        "label": "4 open scenarios after adaptive evolution",
        "suite_group": "open_suite",
        "suite_label": "4-scenario open generalization suite",
        "notes": "The failure-driven adaptive evolution mechanism restored full pass rate on the same open scenarios.",
        "model_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_open_generalization_20260404" / "fine_tuned_rag_after_fix" / "20260404_103919" / "reports" / "model_summary.json",
        },
        "turn_summary_source": {
            "kind": "file",
            "path": REPO_ROOT / "verification" / "artifacts_open_generalization_20260404" / "fine_tuned_rag_after_fix" / "20260404_103919" / "reports" / "tables" / "turn_summary.csv",
        },
    },
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def _read_source_text(source: Dict[str, Any]) -> str:
    kind = source["kind"]
    if kind == "file":
        return Path(source["path"]).read_text(encoding="utf-8")
    if kind == "git":
        result = subprocess.run(
            ["git", "show", f"{source['rev']}:{source['path']}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    raise ValueError(f"Unsupported source kind: {kind}")


def _load_json_source(source: Dict[str, Any]) -> Any:
    return json.loads(_read_source_text(source))


def _load_csv_source(source: Dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(StringIO(_read_source_text(source)))


def _load_results(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario in results["models"][MODEL_KEY]:
        turn_scores = [turn["score_total"] for turn in scenario["turns"]]
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "blueprint": scenario["blueprint"],
                "case_family": scenario["case_family"],
                "case_source": scenario["case_source"],
                "scenario_passed": all(turn["turn_passed"] for turn in scenario["turns"]),
                "conversation_score": round(sum(turn_scores) / len(turn_scores), 4),
            }
        )
    return rows


def _turn_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario in results["models"][MODEL_KEY]:
        for turn in scenario["turns"]:
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "blueprint": scenario["blueprint"],
                    "turn_id": turn["turn_id"],
                    "turn_passed": turn["turn_passed"],
                    "score_total": turn["score_total"],
                    "failure_categories": list(turn.get("failure_categories", [])),
                }
            )
    return rows


def _summarize_progress_stage(stage: Dict[str, Any]) -> Dict[str, Any]:
    model_summary = pd.DataFrame(_load_json_source(stage["model_summary_source"]))
    turn_summary = _load_csv_source(stage["turn_summary_source"])

    model_row = model_summary[model_summary["model_key"] == MODEL_KEY].iloc[0]
    model_label_column = "model_label" if "model_label" in turn_summary.columns else None
    if model_label_column:
        turn_summary = turn_summary[turn_summary["model_label"] == MODEL_LABEL]

    turn_rates = {}
    turn_scores = {}
    for turn_id in [1, 2, 3]:
        row = turn_summary[turn_summary["turn_id"] == turn_id].iloc[0]
        turn_rates[f"turn_{turn_id}_pass_rate"] = float(row["turn_pass_rate"])
        turn_scores[f"turn_{turn_id}_score"] = float(row["avg_turn_score"])

    scenario_count = int(model_row["scenarios"])
    scenario_pass_rate = float(model_row["scenario_pass_rate"])
    scenario_pass_count = int(round(scenario_count * scenario_pass_rate / 100.0))

    return {
        "stage_id": stage["stage_id"],
        "label": stage["label"],
        "suite_group": stage["suite_group"],
        "suite_label": stage["suite_label"],
        "scenario_count": scenario_count,
        "scenario_pass_count": scenario_pass_count,
        "scenario_pass_rate": scenario_pass_rate,
        "avg_conversation_score": float(model_row["avg_conversation_score"]),
        "notes": stage["notes"],
        "model_summary_source": stage["model_summary_source"],
        "turn_summary_source": stage["turn_summary_source"],
        **turn_rates,
        **turn_scores,
    }


def _summarize_stage(stage: Dict[str, Any]) -> Dict[str, Any]:
    results = _load_results(stage["results_path"])
    scenario_rows = _scenario_rows(results)
    turn_rows = _turn_rows(results)
    scenario_df = pd.DataFrame(scenario_rows)
    turn_df = pd.DataFrame(turn_rows)

    failure_counter: Counter[str] = Counter()
    for categories in turn_df["failure_categories"]:
        for category in categories:
            if category:
                failure_counter[category] += 1

    turn_summary = []
    for turn_id in sorted(turn_df["turn_id"].unique()):
        subset = turn_df[turn_df["turn_id"] == turn_id]
        turn_summary.append(
            {
                "turn_id": int(turn_id),
                "pass_count": int(subset["turn_passed"].sum()),
                "turn_count": int(len(subset)),
                "turn_pass_rate": round(100.0 * float(subset["turn_passed"].mean()), 2),
                "avg_turn_score": round(float(subset["score_total"].mean()), 2),
            }
        )

    return {
        "stage_id": stage["stage_id"],
        "label": stage["label"],
        "suite_label": stage["suite_label"],
        "results_path": str(stage["results_path"]),
        "notes": stage["notes"],
        "scenario_count": int(len(scenario_df)),
        "scenario_pass_count": int(scenario_df["scenario_passed"].sum()),
        "scenario_pass_rate": round(100.0 * float(scenario_df["scenario_passed"].mean()), 2),
        "avg_conversation_score": round(float(scenario_df["conversation_score"].mean()), 2),
        "turn_summary": turn_summary,
        "failure_counts": dict(sorted(failure_counter.items())),
        "scenario_rows": scenario_rows,
    }


def _combine_stages(stage_id: str, label: str, stages: List[Dict[str, Any]]) -> Dict[str, Any]:
    scenario_rows: List[Dict[str, Any]] = []
    for stage in stages:
        scenario_rows.extend(stage["scenario_rows"])

    scenario_df = pd.DataFrame(scenario_rows)

    results_maps = []
    for stage in stages:
        results = _load_results(Path(stage["results_path"]))
        turn_map = {}
        for scenario in results["models"][MODEL_KEY]:
            for turn in scenario["turns"]:
                turn_map[(scenario["scenario_id"], turn["turn_id"])] = {
                    "turn_passed": turn["turn_passed"],
                    "score_total": turn["score_total"],
                }
        results_maps.append(turn_map)

    combined_turn_rows: List[Dict[str, Any]] = []
    for stage, turn_map in zip(stages, results_maps):
        for (scenario_id, turn_id), payload in turn_map.items():
            combined_turn_rows.append(
                {
                    "scenario_id": scenario_id,
                    "turn_id": turn_id,
                    "turn_passed": payload["turn_passed"],
                    "score_total": payload["score_total"],
                    "stage_id": stage["stage_id"],
                }
            )

    turn_df = pd.DataFrame(combined_turn_rows)
    turn_summary = []
    for turn_id in sorted(turn_df["turn_id"].unique()):
        subset = turn_df[turn_df["turn_id"] == turn_id]
        turn_summary.append(
            {
                "turn_id": int(turn_id),
                "pass_count": int(subset["turn_passed"].sum()),
                "turn_count": int(len(subset)),
                "turn_pass_rate": round(100.0 * float(subset["turn_passed"].mean()), 2),
                "avg_turn_score": round(float(subset["score_total"].mean()), 2),
            }
        )

    return {
        "stage_id": stage_id,
        "label": label,
        "suite_label": f"{len(scenario_df)}-scenario combined suite",
        "scenario_count": int(len(scenario_df)),
        "scenario_pass_count": int(scenario_df["scenario_passed"].sum()),
        "scenario_pass_rate": round(100.0 * float(scenario_df["scenario_passed"].mean()), 2),
        "avg_conversation_score": round(float(scenario_df["conversation_score"].mean()), 2),
        "turn_summary": turn_summary,
    }


def _save_stage_pass_chart(stage_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(stage_df["label"], stage_df["scenario_pass_rate"], color=["#4f81bd", "#c0504d", "#9bbb59"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Scenario Pass Rate (%)")
    ax.set_title("Progressive Pass Rate During Agent Evolution")
    ax.tick_params(axis="x", rotation=18)
    for idx, value in enumerate(stage_df["scenario_pass_rate"]):
        ax.text(idx, value + 1.5, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "progressive_stage_pass_rate.png", dpi=180)
    plt.close(fig)


def _save_combined_pass_chart(combined_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(combined_df["label"], combined_df["scenario_pass_rate"], color=["#f39c12", "#27ae60"])
    ax.set_ylim(90, 100)
    ax.set_ylabel("Scenario Pass Rate (%)")
    ax.set_title("Expanded Suite Recovery After Adaptive Evolution")
    for idx, value in enumerate(combined_df["scenario_pass_rate"]):
        ax.text(idx, value + 0.15, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "combined_suite_recovery.png", dpi=180)
    plt.close(fig)


def _save_turn_recovery_chart(before: Dict[str, Any], after: Dict[str, Any], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for stage, color in [(before, "#c0504d"), (after, "#27ae60")]:
        ax.plot(
            [item["turn_id"] for item in stage["turn_summary"]],
            [item["turn_pass_rate"] for item in stage["turn_summary"]],
            marker="o",
            linewidth=2,
            label=stage["label"],
            color=color,
        )
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Turn Pass Rate (%)")
    ax.set_title("Open-Suite Turn Recovery After Adaptive Evolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "open_suite_turn_recovery.png", dpi=180)
    plt.close(fig)


def _save_failure_reduction_chart(before: Dict[str, Any], after: Dict[str, Any], figures_dir: Path) -> None:
    categories = sorted(set(before["failure_counts"]) | set(after["failure_counts"]))
    before_values = [before["failure_counts"].get(category, 0) for category in categories]
    after_values = [after["failure_counts"].get(category, 0) for category in categories]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(categories))
    ax.bar([item - 0.18 for item in x], before_values, width=0.36, label="Before fix", color="#c0504d")
    ax.bar([item + 0.18 for item in x], after_values, width=0.36, label="After fix", color="#27ae60")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("Failure Count")
    ax.set_title("Open-Suite Failure Categories Before vs After Adaptive Evolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "open_suite_failure_reduction.png", dpi=180)
    plt.close(fig)


def _save_root_cause_chart(root_cause_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(root_cause_df["label"], root_cause_df["count"], color="#4f81bd")
    ax.set_xlabel("Matched Failure Count")
    ax.set_title("Root-Cause Signatures Extracted From Open Failures")
    for idx, value in enumerate(root_cause_df["count"]):
        ax.text(value + 0.05, idx, str(value), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "root_cause_signatures.png", dpi=180)
    plt.close(fig)


def _save_progress_timeline_chart(timeline_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = range(len(timeline_df))
    ax1.plot(x, timeline_df["scenario_pass_rate"], color="#2e7d32", marker="o", linewidth=2, label="Scenario pass rate")
    ax1.set_ylabel("Scenario Pass Rate (%)", color="#2e7d32")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor="#2e7d32")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(timeline_df["label"], rotation=20, ha="right")
    ax1.set_title("Fine-tuned + RAG Progressive Optimization Timeline")

    ax2 = ax1.twinx()
    ax2.plot(x, timeline_df["avg_conversation_score"], color="#1565c0", marker="s", linewidth=2, label="Average conversation score")
    ax2.set_ylabel("Average Conversation Score", color="#1565c0")
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", labelcolor="#1565c0")

    fig.tight_layout()
    fig.savefig(figures_dir / "ft_rag_progression_timeline.png", dpi=180)
    plt.close(fig)


def _save_progress_turn_chart(turn_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {1: "#1b5e20", 2: "#f57c00", 3: "#6a1b9a"}
    ordered_labels = list(dict.fromkeys(turn_df["label"].tolist()))
    x = range(len(ordered_labels))
    for turn_id in [1, 2, 3]:
        subset = turn_df[turn_df["turn_id"] == turn_id]
        ax.plot(
            x,
            subset["turn_pass_rate"],
            marker="o",
            linewidth=2,
            color=colors[turn_id],
            label=f"Turn {turn_id}",
        )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Turn Pass Rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered_labels, rotation=20, ha="right")
    ax.set_title("Fine-tuned + RAG Turn-Level Progression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "ft_rag_turn_progression.png", dpi=180)
    plt.close(fig)


def _save_suite_growth_chart(timeline_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#90caf9" if group == "main_suite" else "#ffcc80" for group in timeline_df["suite_group"]]
    ax.bar(timeline_df["label"], timeline_df["scenario_count"], color=colors)
    ax.set_ylabel("Scenario Count")
    ax.set_title("Benchmark Growth Across Optimization Stages")
    ax.tick_params(axis="x", rotation=20)
    for idx, value in enumerate(timeline_df["scenario_count"]):
        ax.text(idx, value + 1, f"{value}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "suite_growth_timeline.png", dpi=180)
    plt.close(fig)


def _write_markdown_report(
    output_dir: Path,
    timeline_df: pd.DataFrame,
    open_loop_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    root_cause_df: pd.DataFrame,
    mutation_df: pd.DataFrame,
    profile: Dict[str, Any],
) -> None:
    report_path = output_dir / "agent_evolution_report.md"
    integration_points = [
        "[agent_evolution.py](/home/bshe/Documents/git-research/pfagent/text-to-sim/src/agent_evolution.py)",
        "[prompt_builder.py](/home/bshe/Documents/git-research/pfagent/text-to-sim/src/prompt_builder.py)",
        "[rag_chatbot.py](/home/bshe/Documents/git-research/pfagent/text-to-sim/src/chatbots/openai/rag_chatbot.py)",
        "[agent_evolution_profile.json](/home/bshe/Documents/git-research/pfagent/text-to-sim/data_files/agent_evolution_profile.json)",
    ]
    report = f"""# Agent Evolution Report

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
- {integration_points[0]}
- {integration_points[1]}
- {integration_points[2]}
- {integration_points[3]}

## Full Fine-Tuned + RAG Timeline

{_frame_to_markdown(timeline_df)}

## Open-Scenario Adaptive Evolution Loop

{_frame_to_markdown(open_loop_df)}

## Expanded Suite Summary

{_frame_to_markdown(combined_df)}

## Root-Cause Signatures

{_frame_to_markdown(root_cause_df)}

## Activated Mutation Packs

{_frame_to_markdown(mutation_df)}

## Key Findings

- The earliest retained 100-scenario benchmark started at `51.43` average conversation score and `0%` scenario pass rate for `Fine-tuned + RAG`.
- The same 100-scenario suite was later saturated at `100%`, showing that the first major workflow abstraction closed the original gap.
- Expanding from `100` to `132`, `140`, `146`, and then `152` scenarios created a more realistic progression curve instead of a single static benchmark.
- The 146-scenario generalized suite was the main retained point where performance dropped but still stayed above `95%`.
- Introducing four open-ended scenarios caused the pass rate on that mini-suite to drop to `0%`, while the combined 156-scenario suite remained at `97.44%` because the failure set was small but qualitatively important.
- The adaptive evolution mechanism recovered the open suite to `100%` by translating observed failures into reusable prompt rules and parser hooks.
- The generated profile activated {len(profile["active_mutation_packs"])} mutation packs and produced {len(profile["prompt_guidance"])} adaptive runtime guidance bullets.

## Figures

- [Full Fine-tuned + RAG progression timeline](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/ft_rag_progression_timeline.png)
- [Fine-tuned + RAG turn progression](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/ft_rag_turn_progression.png)
- [Benchmark growth timeline](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/suite_growth_timeline.png)
- [Progressive stage pass rate](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/progressive_stage_pass_rate.png)
- [Expanded suite recovery](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/combined_suite_recovery.png)
- [Open-suite turn recovery](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/open_suite_turn_recovery.png)
- [Open-suite failure reduction](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/open_suite_failure_reduction.png)
- [Root-cause signatures](/home/bshe/Documents/git-research/pfagent/verification/optimization/agent_evolution_20260404/figures/root_cause_signatures.png)
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    output_dir = _ensure_dir(DEFAULT_OUTPUT_DIR)
    figures_dir = _ensure_dir(output_dir / "figures")
    tables_dir = _ensure_dir(output_dir / "tables")

    progress_summaries = [_summarize_progress_stage(stage) for stage in FULL_PROGRESS_STAGES]
    stage_summaries = [_summarize_stage(stage) for stage in OPEN_LOOP_STAGES]
    open_before = next(stage for stage in stage_summaries if stage["stage_id"] == "open_generalization_gap")
    open_after = next(stage for stage in stage_summaries if stage["stage_id"] == "open_generalization_recovered")
    structured_baseline = next(stage for stage in stage_summaries if stage["stage_id"] == "structured_baseline")

    expanded_before = _combine_stages(
        "expanded_before_fix",
        "Expanded 156-scenario suite before adaptive evolution",
        [structured_baseline, open_before],
    )
    expanded_after = _combine_stages(
        "expanded_after_fix",
        "Expanded 156-scenario suite after adaptive evolution",
        [structured_baseline, open_after],
    )

    profile = build_evolution_profile_from_results(
        [Path(open_before["results_path"])],
        profile_version="2026-04-04-open-generalization-v1",
    )
    save_evolution_profile(profile, DEFAULT_PROFILE_PATH)
    save_evolution_profile(profile, output_dir / "agent_evolution_profile.generated.json")

    timeline_df = pd.DataFrame(
        [
            {
                "stage_id": stage["stage_id"],
                "label": stage["label"],
                "suite_group": stage["suite_group"],
                "suite_label": stage["suite_label"],
                "scenario_count": stage["scenario_count"],
                "scenario_pass_count": stage["scenario_pass_count"],
                "scenario_pass_rate": stage["scenario_pass_rate"],
                "avg_conversation_score": stage["avg_conversation_score"],
                "turn_1_pass_rate": stage["turn_1_pass_rate"],
                "turn_2_pass_rate": stage["turn_2_pass_rate"],
                "turn_3_pass_rate": stage["turn_3_pass_rate"],
                "notes": stage["notes"],
            }
            for stage in progress_summaries
        ]
    )
    open_loop_df = pd.DataFrame(
        [
            {
                "stage_id": stage["stage_id"],
                "label": stage["label"],
                "scenario_count": stage["scenario_count"],
                "scenario_pass_count": stage["scenario_pass_count"],
                "scenario_pass_rate": stage["scenario_pass_rate"],
                "avg_conversation_score": stage["avg_conversation_score"],
                "turn_1_pass_rate": stage["turn_1_pass_rate"],
                "turn_2_pass_rate": stage["turn_2_pass_rate"],
                "turn_3_pass_rate": stage["turn_3_pass_rate"],
                "notes": stage["notes"],
            }
            for stage in progress_summaries
            if stage["suite_group"] == "open_suite"
        ]
    )
    stage_df = pd.DataFrame(
        [
            {
                "stage_id": stage["stage_id"],
                "label": stage["label"],
                "suite_label": stage["suite_label"],
                "scenario_count": stage["scenario_count"],
                "scenario_pass_count": stage["scenario_pass_count"],
                "scenario_pass_rate": stage["scenario_pass_rate"],
                "avg_conversation_score": stage["avg_conversation_score"],
                "notes": stage["notes"],
            }
            for stage in stage_summaries
        ]
    )
    combined_df = pd.DataFrame(
        [
            {
                "stage_id": stage["stage_id"],
                "label": stage["label"],
                "scenario_count": stage["scenario_count"],
                "scenario_pass_count": stage["scenario_pass_count"],
                "scenario_pass_rate": stage["scenario_pass_rate"],
                "avg_conversation_score": stage["avg_conversation_score"],
            }
            for stage in [expanded_before, expanded_after]
        ]
    )
    root_cause_df = pd.DataFrame(profile["root_cause_summary"])
    mutation_df = pd.DataFrame(
        [
            {
                "pack_id": pack_id,
                "title": MUTATION_LIBRARY[pack_id].title,
                "prompt_guidance_count": len(MUTATION_LIBRARY[pack_id].prompt_guidance),
            }
            for pack_id in profile["active_mutation_packs"]
        ]
    )
    failure_df = pd.DataFrame(
        [
            {"failure_category": category, "before_fix": open_before["failure_counts"].get(category, 0), "after_fix": open_after["failure_counts"].get(category, 0)}
            for category in sorted(set(open_before["failure_counts"]) | set(open_after["failure_counts"]))
        ]
    )
    turn_df = pd.DataFrame(
        [
            {
                "suite_label": open_before["suite_label"],
                "phase": "before_fix",
                "turn_id": item["turn_id"],
                "turn_pass_rate": item["turn_pass_rate"],
            }
            for item in open_before["turn_summary"]
        ]
        + [
            {
                "suite_label": open_after["suite_label"],
                "phase": "after_fix",
                "turn_id": item["turn_id"],
                "turn_pass_rate": item["turn_pass_rate"],
            }
            for item in open_after["turn_summary"]
        ]
    )
    timeline_turn_df = pd.DataFrame(
        [
            {
                "stage_id": stage["stage_id"],
                "label": stage["label"],
                "suite_group": stage["suite_group"],
                "turn_id": turn_id,
                "turn_pass_rate": stage[f"turn_{turn_id}_pass_rate"],
            }
            for stage in progress_summaries
            for turn_id in [1, 2, 3]
        ]
    )

    timeline_df.to_csv(tables_dir / "ft_rag_progression_timeline.csv", index=False)
    timeline_turn_df.to_csv(tables_dir / "ft_rag_turn_progression.csv", index=False)
    stage_df.to_csv(tables_dir / "progressive_stage_summary.csv", index=False)
    combined_df.to_csv(tables_dir / "expanded_suite_summary.csv", index=False)
    root_cause_df.to_csv(tables_dir / "root_cause_summary.csv", index=False)
    mutation_df.to_csv(tables_dir / "mutation_pack_summary.csv", index=False)
    failure_df.to_csv(tables_dir / "open_suite_failure_summary.csv", index=False)
    turn_df.to_csv(tables_dir / "open_suite_turn_summary.csv", index=False)

    _save_progress_timeline_chart(timeline_df, figures_dir)
    _save_progress_turn_chart(timeline_turn_df, figures_dir)
    _save_suite_growth_chart(timeline_df, figures_dir)
    _save_stage_pass_chart(stage_df, figures_dir)
    _save_combined_pass_chart(combined_df, figures_dir)
    _save_turn_recovery_chart(open_before, open_after, figures_dir)
    _save_failure_reduction_chart(open_before, open_after, figures_dir)
    _save_root_cause_chart(root_cause_df, figures_dir)
    _write_markdown_report(output_dir, timeline_df, open_loop_df, stage_df, combined_df, root_cause_df, mutation_df, profile)

    optimization_log = {
        "generated_at": datetime.now().isoformat(),
        "model_key": MODEL_KEY,
        "model_label": MODEL_LABEL,
        "workflow": [
            "stress_test_open_scenarios",
            "collect_verification_failures",
            "extract_root_cause_signatures",
            "activate_mutation_packs",
            "update_runtime_profile",
            "revalidate_same_scenarios",
        ],
        "full_progression": progress_summaries,
        "stage_summaries": stage_summaries,
        "combined_summaries": [expanded_before, expanded_after],
        "profile_path": str(DEFAULT_PROFILE_PATH),
        "profile_summary": {
            "profile_version": profile["profile_version"],
            "active_mutation_packs": profile["active_mutation_packs"],
            "root_cause_summary": profile["root_cause_summary"],
        },
    }
    (output_dir / "progressive_optimization_log.json").write_text(
        json.dumps(optimization_log, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
