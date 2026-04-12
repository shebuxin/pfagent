from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "base_openai": "Base OpenAI",
    "rag": "RAG",
    "fine_tuned": "Fine-tuned",
    "fine_tuned_rag": "Fine-tuned + RAG",
}


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


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _turn_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_key, scenarios in results["models"].items():
        for scenario in scenarios:
            for turn in scenario["turns"]:
                rows.append(
                    {
                        "model_key": model_key,
                        "model_label": MODEL_LABELS.get(model_key, model_key),
                        "scenario_id": scenario["scenario_id"],
                        "blueprint": scenario["blueprint"],
                        "case_family": scenario["case_family"],
                        "case_source": scenario["case_source"],
                        "turn_id": turn["turn_id"],
                        "turn_passed": turn["turn_passed"],
                        "score_total": turn["score_total"],
                        "format_score": turn["format_score"],
                        "grounding_score": turn["grounding_score"],
                        "continuity_score": turn["continuity_score"],
                        "execution_score": turn["execution_score"],
                        "semantic_score": turn["semantic_score"],
                        "artifact_score": turn["artifact_score"],
                        "format_valid": turn["format_valid"],
                        "execution_passed": turn["execution_passed"],
                        "semantic_passed": turn["semantic_passed"],
                        "artifact_passed": turn["artifact_passed"],
                        "response_used_fallback": turn.get("used_template_fallback", False),
                        "failure_categories": ",".join(turn["failure_categories"]),
                    }
                )
    return rows


def _scenario_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_key, scenarios in results["models"].items():
        for scenario in scenarios:
            turn_scores = [turn["score_total"] for turn in scenario["turns"]]
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": MODEL_LABELS.get(model_key, model_key),
                    "scenario_id": scenario["scenario_id"],
                    "blueprint": scenario["blueprint"],
                    "case_family": scenario["case_family"],
                    "case_source": scenario["case_source"],
                    "scenario_passed": all(turn["turn_passed"] for turn in scenario["turns"]),
                    "conversation_score": round(sum(turn_scores) / len(turn_scores), 4),
                }
            )
    return rows


def _save_overall_score_chart(summary_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = summary_df["model_label"].tolist()
    values = summary_df["avg_conversation_score"].tolist()
    colors = ["#6c8ebf", "#7cb342", "#ff8f00", "#c62828"]
    ax.bar(labels, values, color=colors[: len(labels)])
    # Headroom above 100 so the value labels don't collide with the title.
    ax.set_ylim(0, 115)
    ax.set_ylabel("Average Conversation Score")
    ax.set_title("Average Verification Score by Model", pad=12)
    for idx, value in enumerate(values):
        ax.text(idx, value + 2, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "overall_score_by_model.png", dpi=180)
    plt.close(fig)


def _save_pass_rate_chart(summary_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = summary_df["model_label"].tolist()
    values = summary_df["scenario_pass_rate"].tolist()
    colors = ["#5e35b1", "#00897b", "#c62828", "#ffb300"]
    ax.bar(labels, values, color=colors[: len(labels)])
    # Headroom above 100 so the value labels don't collide with the title.
    ax.set_ylim(0, 115)
    ax.set_ylabel("Scenario Full-Pass Rate (%)")
    ax.set_title("Conversation-Level Pass Rate by Model", pad=12)
    for idx, value in enumerate(values):
        ax.text(idx, value + 2, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "scenario_pass_rate_by_model.png", dpi=180)
    plt.close(fig)


def _save_turn_pass_chart(turn_df: pd.DataFrame, figure_dir: Path) -> None:
    grouped = (
        turn_df.groupby(["model_label", "turn_id"])["turn_passed"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_label in grouped["model_label"].unique():
        subset = grouped[grouped["model_label"] == model_label]
        ax.plot(
            subset["turn_id"],
            subset["turn_passed"] * 100.0,
            marker="o",
            linewidth=2,
            label=model_label,
        )
    ax.set_xticks([1, 2, 3])
    # Headroom above 100 so the high-performing models' lines don't sit
    # under the title and the legend can fit below them.
    ax.set_ylim(-5, 115)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Turn Pass Rate (%)")
    ax.set_title("Turn-by-Turn Pass Rate", pad=12)
    # Place the legend outside the plot area on the right so it cannot
    # overlap the lines or the title.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "turn_pass_rate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_failure_chart(turn_df: pd.DataFrame, figure_dir: Path) -> None:
    exploded = (
        turn_df.assign(
            failure_category=turn_df["failure_categories"].str.split(",")
        )
        .explode("failure_category")
    )
    exploded = exploded[exploded["failure_category"].astype(bool)]
    if exploded.empty:
        return

    grouped = (
        exploded.groupby(["model_label", "failure_category"])
        .size()
        .reset_index(name="count")
    )
    categories = sorted(grouped["failure_category"].unique())
    models = [label for label in MODEL_LABELS.values() if label in grouped["model_label"].unique()]
    matrix = np.zeros((len(models), len(categories)))
    for row in grouped.itertuples(index=False):
        matrix[models.index(row.model_label), categories.index(row.failure_category)] = row.count

    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.1), 5))
    x = np.arange(len(categories))
    width = 0.22
    colors = ["#6c8ebf", "#7cb342", "#ff8f00", "#c62828"]
    for idx, model_label in enumerate(models):
        ax.bar(x + (idx - 1) * width, matrix[idx], width=width, label=model_label, color=colors[idx % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.set_ylabel("Failure Count")
    ax.set_title("Failure Categories by Model", pad=12)
    # Add 12% headroom above the tallest bar so the legend has space
    # without colliding with the title or the bar tops.
    ymax = float(matrix.max()) if matrix.size else 0.0
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
    # Move the legend outside the plot to the right -- inside-the-plot
    # placement was overlapping the centered title at the top.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "failure_categories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_heatmap(turn_df: pd.DataFrame, figure_dir: Path) -> None:
    pivot = turn_df.pivot_table(
        index="blueprint",
        columns="model_label",
        values="score_total",
        aggfunc="mean",
    ).sort_index()
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.6)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Average Turn Score by Blueprint and Model")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Average Score")
    fig.tight_layout()
    fig.savefig(figure_dir / "blueprint_model_heatmap.png", dpi=180)
    plt.close(fig)


def generate_reports(results: Dict[str, Any], output_root: Path) -> Dict[str, str]:
    report_dir = _ensure_dir(output_root / "reports")
    figure_dir = _ensure_dir(report_dir / "figures")
    table_dir = _ensure_dir(report_dir / "tables")

    turn_df = pd.DataFrame(_turn_rows(results))
    scenario_df = pd.DataFrame(_scenario_rows(results))

    model_summary = (
        scenario_df.groupby(["model_key", "model_label"])
        .agg(
            scenarios=("scenario_id", "count"),
            scenario_pass_rate=("scenario_passed", lambda s: round(100.0 * float(s.mean()), 2)),
            avg_conversation_score=("conversation_score", lambda s: round(float(s.mean()), 2)),
        )
        .reset_index()
        .sort_values("avg_conversation_score", ascending=False)
    )

    turn_summary = (
        turn_df.groupby(["model_label", "turn_id"])
        .agg(
            turn_pass_rate=("turn_passed", lambda s: round(100.0 * float(s.mean()), 2)),
            avg_turn_score=("score_total", lambda s: round(float(s.mean()), 2)),
        )
        .reset_index()
    )

    family_summary = (
        scenario_df.groupby(["model_label", "case_family", "case_source"])
        .agg(
            scenario_pass_rate=("scenario_passed", lambda s: round(100.0 * float(s.mean()), 2)),
            avg_conversation_score=("conversation_score", lambda s: round(float(s.mean()), 2)),
        )
        .reset_index()
    )

    failure_summary = (
        turn_df.assign(failure_category=turn_df["failure_categories"].str.split(","))
        .explode("failure_category")
    )
    failure_summary = failure_summary[failure_summary["failure_category"].astype(bool)]
    if not failure_summary.empty:
        failure_summary = (
            failure_summary.groupby(["model_label", "failure_category"])
            .size()
            .reset_index(name="count")
            .sort_values(["model_label", "count"], ascending=[True, False])
        )

    model_summary.to_csv(table_dir / "model_summary.csv", index=False)
    turn_summary.to_csv(table_dir / "turn_summary.csv", index=False)
    family_summary.to_csv(table_dir / "family_summary.csv", index=False)
    turn_df.to_csv(table_dir / "turn_level_results.csv", index=False)
    scenario_df.to_csv(table_dir / "scenario_level_results.csv", index=False)
    if not failure_summary.empty:
        failure_summary.to_csv(table_dir / "failure_summary.csv", index=False)

    _save_overall_score_chart(model_summary, figure_dir)
    _save_pass_rate_chart(model_summary, figure_dir)
    _save_turn_pass_chart(turn_df, figure_dir)
    _save_failure_chart(turn_df, figure_dir)
    _save_heatmap(turn_df, figure_dir)

    markdown_lines = [
        "# Verification Summary",
        "",
        "## Evaluation System",
        "",
        "- `Format` (10): exactly one Python code block, no missing code.",
        "- `Grounding` (25): case loading correctness plus prompt-specific literals and required API usage.",
        "- `Continuity` (15): follow-up turns preserve required earlier modifications.",
        "- `Execution` (20): generated code runs successfully in the ANDES runtime.",
        "- `Semantic` (25): `RESULT_JSON` matches the oracle result within tolerance.",
        "- `Artifact` (5): required plot files are created and reported correctly.",
        "",
        "## Model Summary",
        "",
        _frame_to_markdown(model_summary),
        "",
        "## Turn Summary",
        "",
        _frame_to_markdown(turn_summary),
        "",
        "## Family Summary",
        "",
        _frame_to_markdown(family_summary),
    ]
    if not failure_summary.empty:
        markdown_lines.extend(
            [
                "",
                "## Failure Summary",
                "",
                _frame_to_markdown(failure_summary),
            ]
        )

    summary_path = report_dir / "verification_summary.md"
    summary_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    (report_dir / "model_summary.json").write_text(
        json.dumps(model_summary.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    return {
        "report_dir": str(report_dir),
        "summary_markdown": str(summary_path),
        "model_summary_csv": str(table_dir / "model_summary.csv"),
        "turn_summary_csv": str(table_dir / "turn_summary.csv"),
        "family_summary_csv": str(table_dir / "family_summary.csv"),
        "overall_score_figure": str(figure_dir / "overall_score_by_model.png"),
        "scenario_pass_rate_figure": str(figure_dir / "scenario_pass_rate_by_model.png"),
    }
