from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = [
    "Base OpenAI",
    "Fine-tuned",
    "RAG",
    "Fine-tuned + RAG",
]

MODEL_COLORS = {
    "Base OpenAI": "#9e9e9e",
    "Fine-tuned": "#ef6c00",
    "RAG": "#2e7d32",
    "Fine-tuned + RAG": "#1565c0",
}


def _frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _ordered_models(values: list[str]) -> list[str]:
    present = set(values)
    return [name for name in MODEL_ORDER if name in present]


def _save_model_bar(model_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = model_df["model_label"].tolist()
    values = model_df["scenario_pass_rate"].tolist()
    colors = [MODEL_COLORS[label] for label in labels]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Scenario Pass Rate (%)")
    ax.set_title("164-Scenario Pass Rate by Model")
    for idx, value in enumerate(values):
        ax.text(idx, value + 1, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "model_pass_rate.png", dpi=180)
    plt.close(fig)


def _save_turn_line(model_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in model_df.iterrows():
        label = row["model_label"]
        values = [row["turn1_pass_rate"], row["turn2_pass_rate"], row["turn3_pass_rate"]]
        ax.plot([1, 2, 3], values, marker="o", linewidth=2, label=label, color=MODEL_COLORS[label])
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Turn Pass Rate (%)")
    ax.set_title("Turn-by-Turn Pass Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "turn_pass_rate_lines.png", dpi=180)
    plt.close(fig)


def _save_family_heatmap(family_df: pd.DataFrame, figure_dir: Path) -> None:
    family_df = family_df.copy()
    family_df["family_key"] = family_df["case_family"] + " (" + family_df["case_source"] + ")"
    pivot = family_df.pivot(index="family_key", columns="model_label", values="scenario_pass_rate")
    pivot = pivot[_ordered_models(pivot.columns.tolist())]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Scenario Pass Rate by Case Family")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pass Rate (%)")
    fig.tight_layout()
    fig.savefig(figure_dir / "family_pass_rate_heatmap.png", dpi=180)
    plt.close(fig)


def _save_blueprint_bar(blueprint_df: pd.DataFrame, figure_dir: Path) -> None:
    difficult = (
        blueprint_df.groupby("blueprint")["scenario_pass_rate"]
        .mean()
        .sort_values()
        .head(8)
        .index
    )
    subset = blueprint_df[blueprint_df["blueprint"].isin(difficult)].copy()
    subset["blueprint"] = pd.Categorical(subset["blueprint"], categories=list(difficult), ordered=True)
    subset = subset.sort_values(["blueprint", "model_label"])

    fig, ax = plt.subplots(figsize=(12, 5))
    models = _ordered_models(subset["model_label"].tolist())
    x = range(len(difficult))
    width = 0.18
    offsets = {
        model: (idx - (len(models) - 1) / 2) * width
        for idx, model in enumerate(models)
    }
    for model in models:
        rows = subset[subset["model_label"] == model].set_index("blueprint").reindex(difficult).reset_index()
        ax.bar(
            [idx + offsets[model] for idx in x],
            rows["scenario_pass_rate"],
            width=width,
            label=model,
            color=MODEL_COLORS[model],
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(difficult, rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Scenario Pass Rate (%)")
    ax.set_title("Most Difficult Blueprints")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "hardest_blueprints.png", dpi=180)
    plt.close(fig)


def _save_open_scenarios(open_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    scenario_ids = sorted(open_df["scenario_id"].unique())
    models = _ordered_models(open_df["model_label"].tolist())
    x = range(len(scenario_ids))
    width = 0.18
    offsets = {
        model: (idx - (len(models) - 1) / 2) * width
        for idx, model in enumerate(models)
    }
    for model in models:
        rows = open_df[open_df["model_label"] == model].set_index("scenario_id").reindex(scenario_ids).reset_index()
        ax.bar(
            [idx + offsets[model] for idx in x],
            rows["conversation_score"],
            width=width,
            label=model,
            color=MODEL_COLORS[model],
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(scenario_ids)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Conversation Score")
    ax.set_title("Open Scenarios 161-164")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "open_scenarios_161_164.png", dpi=180)
    plt.close(fig)


def build_report(output_dir: Path) -> None:
    scenario_df = pd.read_csv(output_dir / "combined_scenario_level_results.csv")
    turn_df = pd.read_csv(output_dir / "combined_turn_level_results.csv")
    model_df = pd.read_csv(output_dir / "combined_model_summary.csv")
    open_df = pd.read_csv(output_dir / "open_scenarios_161_164.csv")

    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    family_df = (
        scenario_df.groupby(["model_label", "case_family", "case_source"], as_index=False)
        .agg(
            scenario_pass_rate=("scenario_passed", lambda s: round(100.0 * s.astype(str).str.lower().eq("true").mean(), 2)),
            avg_conversation_score=("conversation_score", lambda s: round(float(s.mean()), 2)),
        )
    )
    family_df.to_csv(table_dir / "family_summary.csv", index=False)

    blueprint_df = (
        scenario_df.groupby(["model_label", "blueprint"], as_index=False)
        .agg(
            scenario_pass_rate=("scenario_passed", lambda s: round(100.0 * s.astype(str).str.lower().eq("true").mean(), 2)),
            avg_conversation_score=("conversation_score", lambda s: round(float(s.mean()), 2)),
            scenarios=("scenario_id", "count"),
        )
    )
    blueprint_df.to_csv(table_dir / "blueprint_summary.csv", index=False)

    failure_df = (
        turn_df.assign(failure_category=turn_df["failure_categories"].fillna("").str.split(","))
        .explode("failure_category")
    )
    failure_df = failure_df[failure_df["failure_category"].astype(str).str.len() > 0]
    failure_df = (
        failure_df.groupby(["model_label", "failure_category"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    failure_df.to_csv(table_dir / "failure_summary.csv", index=False)

    _save_model_bar(model_df, figure_dir)
    _save_turn_line(model_df, figure_dir)
    _save_family_heatmap(family_df, figure_dir)
    _save_blueprint_bar(blueprint_df, figure_dir)
    _save_open_scenarios(open_df, figure_dir)

    hardest_blueprints = (
        blueprint_df.groupby("blueprint")["scenario_pass_rate"]
        .mean()
        .sort_values()
        .head(8)
        .reset_index()
    )
    hardest_open = open_df.sort_values(["scenario_id", "model_label"])

    lines = [
        "# 164-Scenario Four-Model Verification Digest",
        "",
        "## Snapshot",
        "",
        "- Evaluation scope: `164 scenarios x 4 models x 3 turns`.",
        "- Best scenario pass rate in this run: `RAG = Fine-tuned + RAG = 60.98%`.",
        "- `Fine-tuned` reached `34.76%`; `Base OpenAI` remained at `0.00%`.",
        "- The main bottleneck for `RAG` and `Fine-tuned + RAG` was no longer execution; it was concentrated in `turn 3` grounding for targeted case-edit and outage reasoning.",
        "",
        "## Model Summary",
        "",
        _frame_to_markdown(model_df),
        "",
        "## Key Takeaways",
        "",
        "- `RAG` and `Fine-tuned + RAG` were identical on this 164-scenario suite: full pass on turns 1-2, then a drop to `60.98%` on turn 3.",
        "- `Fine-tuned` improved over `Base OpenAI` but still showed broad execution, semantic, and grounding failures.",
        "- The newly added open scenarios `161-164` all failed for every model, which explains why the suite now behaves more realistically than earlier near-100% runs.",
        "",
        "## Case-Family Summary",
        "",
        _frame_to_markdown(family_df),
        "",
        "## Hardest Blueprints",
        "",
        _frame_to_markdown(hardest_blueprints),
        "",
        "## Open Scenarios 161-164",
        "",
        _frame_to_markdown(hardest_open),
        "",
        "## Failure Concentration",
        "",
        _frame_to_markdown(failure_df),
        "",
        "## Figures",
        "",
        "- [model_pass_rate.png](figures/model_pass_rate.png)",
        "- [turn_pass_rate_lines.png](figures/turn_pass_rate_lines.png)",
        "- [family_pass_rate_heatmap.png](figures/family_pass_rate_heatmap.png)",
        "- [hardest_blueprints.png](figures/hardest_blueprints.png)",
        "- [open_scenarios_161_164.png](figures/open_scenarios_161_164.png)",
        "",
        "## Tables",
        "",
        "- [family_summary.csv](tables/family_summary.csv)",
        "- [blueprint_summary.csv](tables/blueprint_summary.csv)",
        "- [failure_summary.csv](tables/failure_summary.csv)",
    ]
    (output_dir / "aggregate_digest.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="verification/artifacts_recheck_20260404_164_aggregate",
        help="Directory containing the combined aggregate CSV files.",
    )
    args = parser.parse_args()
    build_report(Path(args.output_dir))


if __name__ == "__main__":
    main()
