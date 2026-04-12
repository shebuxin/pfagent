from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_TO_SIM_ROOT = REPO_ROOT / "text-to-sim"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

import andes

from src.andes_manual import bootstrap_default_andes_manual
from src.chatbot_factory import DEFAULT_BASE_MODEL, DEFAULT_FINETUNED_MODEL
from src.chatbots.openai.rag_chatbot import RAGChatbot, RAGConfig, validate_response_code
from src.chatbots.openai.simple_chatbot import SimpleChatConfig, SimpleChatbot

from verification.oracle import compute_oracle_for_scenario
from verification.reporting import generate_reports
from verification.suite import (
    FULL_SUITE_SCENARIO_COUNT,
    OPEN_GENERALIZATION_SCENARIO_COUNT,
    build_open_generalization_suite,
    build_verification_suite,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".svg")
MODEL_SPECS = {
    "base_openai": {"label": "Base OpenAI"},
    "rag": {"label": "RAG"},
    "fine_tuned": {"label": "Fine-tuned"},
    "fine_tuned_rag": {"label": "Fine-tuned + RAG"},
}

logging.getLogger("andes").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

PLOT_CAPTURE_PREAMBLE = """
try:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _verification_output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(_verification_output_dir, exist_ok=True)
    _verification_existing = [
        name for name in os.listdir(_verification_output_dir)
        if name.startswith("plot_") and name.endswith(".png")
    ]
    _verification_plot_counter = len(_verification_existing)

    def _verification_safe_show(*args, **kwargs):
        global _verification_plot_counter
        saved_paths = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            _verification_plot_counter += 1
            plot_path = os.path.join(_verification_output_dir, f"plot_{_verification_plot_counter}.png")
            fig.savefig(plot_path, bbox_inches="tight")
            saved_paths.append(plot_path)
        if saved_paths:
            print("Saved plot(s):")
            for path in saved_paths:
                print(f"- {path}")
        plt.close("all")

    plt.show = _verification_safe_show
except Exception:
    pass
"""


def _extract_python_blocks(response: str) -> List[str]:
    return re.findall(r"```python\s*\n(.*?)```", response or "", re.DOTALL)


def _extract_python_code(response: str) -> str:
    blocks = _extract_python_blocks(response)
    return blocks[0].strip() if blocks else ""


def _response_is_single_code_block(response: str) -> bool:
    if not response:
        return False
    stripped = response.strip()
    if len(_extract_python_blocks(stripped)) != 1:
        return False
    return bool(re.fullmatch(r"```python\s*\n.*?```\s*", stripped, re.DOTALL))


def _build_uploaded_runtime_context(prompt: str, filename: str, workspace_dir: Path) -> str:
    return (
        f"{prompt}\n\n"
        "Runtime file context:\n"
        f"- Working directory for execution: {workspace_dir}\n"
        "- Uploaded files available during execution:\n"
        f"- {filename}\n"
        "- Use this exact filename directly in andes.load(...).\n"
        "- Do not wrap the uploaded filename with andes.get_case(...).\n"
    )


async def create_chatbot(
    model_key: str,
    api_key: str,
    base_model: str,
    fine_tuned_model: str,
):
    if model_key == "base_openai":
        chatbot = SimpleChatbot(
            SimpleChatConfig(
                openai_api_key=api_key,
                chat_model=base_model,
                code_compilation_check=True,
                max_compilation_retries=2,
            )
        )
        chatbot.load_system_prompt(session_id="verification_session", custom_instructions="")
        return chatbot

    if model_key == "fine_tuned":
        chatbot = SimpleChatbot(
            SimpleChatConfig(
                openai_api_key=api_key,
                chat_model=fine_tuned_model,
                code_compilation_check=True,
                max_compilation_retries=2,
            )
        )
        chatbot.load_system_prompt(session_id="verification_session", custom_instructions="")
        return chatbot

    if model_key == "rag":
        chatbot = RAGChatbot(
            RAGConfig(
                openai_api_key=api_key,
                chat_model=base_model,
                code_compilation_check=True,
                max_compilation_retries=2,
                allow_template_fallback=False,
            )
        )
        await bootstrap_default_andes_manual(chatbot)
        chatbot.load_system_prompt(session_id="verification_session", custom_instructions="")
        return chatbot

    if model_key == "fine_tuned_rag":
        chatbot = RAGChatbot(
            RAGConfig(
                openai_api_key=api_key,
                chat_model=fine_tuned_model,
                code_compilation_check=True,
                max_compilation_retries=2,
                allow_template_fallback=False,
            )
        )
        await bootstrap_default_andes_manual(chatbot)
        chatbot.load_system_prompt(session_id="verification_session", custom_instructions="")
        return chatbot

    raise ValueError(f"Unsupported model key: {model_key}")


def _copy_uploaded_case(scenario: Dict[str, Any], workspace_dir: Path) -> None:
    if scenario["case_source"] != "uploaded":
        return

    source_case = Path(andes.get_case(scenario["source_case_path"]))
    target_case = workspace_dir / scenario["uploaded_filename"]
    shutil.copyfile(source_case, target_case)


def _prepare_workspace(
    run_root: Path,
    model_key: str,
    scenario: Dict[str, Any],
    turn_id: int,
) -> Path:
    workspace_dir = run_root / "raw" / model_key / scenario["scenario_id"] / f"turn_{turn_id:02d}" / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    _copy_uploaded_case(scenario, workspace_dir)
    return workspace_dir


def _find_image_artifacts(workspace_dir: Path) -> List[str]:
    paths: List[str] = []
    for path in workspace_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(str(path.relative_to(workspace_dir)))
    return sorted(paths)


def _execute_generated_code(code: str, workspace_dir: Path) -> Dict[str, Any]:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    code_path = workspace_dir / "generated_code.py"
    code_path.write_text(f"{PLOT_CAPTURE_PREAMBLE}\n{code}", encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pfagent-verification")
    result = subprocess.run(
        [sys.executable, code_path.name],
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    output = ""
    if stdout:
        output += f"STDOUT:\n{stdout}\n"
    if stderr:
        output += f"STDERR:\n{stderr}\n"
    if not output:
        output = "No output captured."

    result_json = None
    parse_error = None
    result_line = None
    for line in stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            result_line = line[len("RESULT_JSON="):].strip()
    if result_line:
        try:
            result_json = json.loads(result_line)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
    else:
        parse_error = "RESULT_JSON marker not found in stdout."

    return {
        "execution_passed": result.returncode == 0,
        "return_code": result.returncode,
        "execution_output": output,
        "result_json": result_json,
        "result_json_parse_error": parse_error,
        "image_artifacts": _find_image_artifacts(workspace_dir),
    }


def _score_check_group(code: str, checks: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
    if not checks:
        return 1.0, []

    total_weight = sum(float(item.get("weight", 1.0)) for item in checks)
    matched_weight = 0.0
    issues: List[str] = []
    for item in checks:
        pattern = item["pattern"]
        weight = float(item.get("weight", 1.0))
        if re.search(pattern, code, re.MULTILINE | re.DOTALL):
            matched_weight += weight
        else:
            issues.append(f"Missing code requirement: {item['label']}")
    return (matched_weight / total_weight if total_weight else 1.0), issues


def _find_forbidden_hits(code: str, patterns: List[str]) -> List[str]:
    hits: List[str] = []
    for pattern in patterns:
        if re.search(pattern, code, re.MULTILINE | re.DOTALL):
            hits.append(f"Forbidden pattern detected: {pattern}")
    return hits


def _compare_values(actual: Any, expected: Any, path: str) -> List[str]:
    mismatches: List[str] = []
    if isinstance(expected, float):
        try:
            if abs(float(actual) - expected) > 1e-4:
                mismatches.append(f"{path}: expected {expected}, got {actual}")
        except Exception:
            mismatches.append(f"{path}: expected numeric value {expected}, got {actual}")
        return mismatches

    if isinstance(expected, int):
        try:
            if int(actual) != expected:
                mismatches.append(f"{path}: expected {expected}, got {actual}")
        except Exception:
            mismatches.append(f"{path}: expected integer value {expected}, got {actual}")
        return mismatches

    if isinstance(expected, str):
        if str(actual) != expected:
            mismatches.append(f"{path}: expected {expected}, got {actual}")
        return mismatches

    if isinstance(expected, list):
        if not isinstance(actual, list):
            mismatches.append(f"{path}: expected list, got {type(actual).__name__}")
            return mismatches
        if len(actual) != len(expected):
            mismatches.append(f"{path}: expected list length {len(expected)}, got {len(actual)}")
            return mismatches
        for idx, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            mismatches.extend(_compare_values(actual_item, expected_item, f"{path}[{idx}]"))
        return mismatches

    if actual != expected:
        mismatches.append(f"{path}: expected {expected}, got {actual}")
    return mismatches


def _compare_result_json(actual: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> Tuple[float, List[str]]:
    if actual is None:
        return 0.0, ["RESULT_JSON was missing or could not be parsed."]

    mismatches: List[str] = []
    matched = 0
    total = len(expected)
    for key, expected_value in expected.items():
        if key not in actual:
            mismatches.append(f"Missing RESULT_JSON key: {key}")
            continue
        value_mismatches = _compare_values(actual[key], expected_value, key)
        if value_mismatches:
            mismatches.extend(value_mismatches)
        else:
            matched += 1
    return (matched / total if total else 1.0), mismatches


def _artifact_pass(turn: Dict[str, Any], execution: Dict[str, Any], actual_result: Optional[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    if not turn.get("expects_plot"):
        return True, []

    expected_plot = turn["plot_filename"]
    actual_plot = actual_result.get("plot_file") if isinstance(actual_result, dict) else None
    artifact_paths = execution["image_artifacts"]
    issues: List[str] = []
    if actual_plot != expected_plot:
        issues.append(f"Plot filename mismatch: expected {expected_plot}, got {actual_plot}")
    if expected_plot not in artifact_paths:
        issues.append(f"Expected plot artifact not found: {expected_plot}")
    return len(issues) == 0, issues


def _classify_failure_categories(
    format_valid: bool,
    grounding_issues: List[str],
    continuity_issues: List[str],
    validation_issues: List[str],
    execution_passed: bool,
    semantic_issues: List[str],
    artifact_issues: List[str],
) -> List[str]:
    categories: List[str] = []
    if not format_valid:
        categories.append("format")
    if grounding_issues or validation_issues:
        categories.append("grounding")
    if continuity_issues:
        categories.append("continuity")
    if not execution_passed:
        categories.append("execution")
    if semantic_issues:
        categories.append("semantic")
    if artifact_issues:
        categories.append("artifact")
    return categories


def _evaluate_turn(
    turn: Dict[str, Any],
    scenario: Dict[str, Any],
    prompt: str,
    response: str,
    expected_result: Dict[str, Any],
    execution: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    format_valid = _response_is_single_code_block(response)
    code = _extract_python_code(response)
    format_issues = [] if format_valid else ["Response was not exactly one fenced Python code block."]

    grounding_ratio, grounding_issues = _score_check_group(code, turn["current_code_checks"])
    continuity_ratio, continuity_issues = _score_check_group(code, turn["carry_forward_checks"])
    forbidden_issues = _find_forbidden_hits(code, turn["forbidden_patterns"])

    validation_ok, validation_errors = validate_response_code(response, user_context=prompt)
    validation_issues = [] if validation_ok else validation_errors

    execution_passed = bool(execution and execution["execution_passed"])
    execution_output = execution["execution_output"] if execution else None
    actual_result = execution["result_json"] if execution else None

    semantic_ratio, semantic_issues = _compare_result_json(actual_result, expected_result) if execution else (0.0, ["Execution was not attempted."])
    artifact_ok, artifact_issues = _artifact_pass(turn, execution or {"image_artifacts": []}, actual_result)

    format_score = 10.0 if format_valid else 0.0
    grounding_score = round(25.0 * grounding_ratio, 2)
    continuity_score = round(15.0 * continuity_ratio, 2)
    execution_score = 20.0 if execution_passed else 0.0
    semantic_score = round(25.0 * semantic_ratio, 2)
    artifact_score = 5.0 if artifact_ok else 0.0
    score_total = round(
        format_score + grounding_score + continuity_score + execution_score + semantic_score + artifact_score,
        2,
    )

    all_issues = (
        format_issues
        + grounding_issues
        + continuity_issues
        + forbidden_issues
        + validation_issues
        + ([] if execution_passed else [f"Execution failed. Return code: {execution['return_code']}" if execution else "Execution was not attempted."])
        + semantic_issues
        + artifact_issues
    )
    failure_categories = _classify_failure_categories(
        format_valid,
        grounding_issues + forbidden_issues,
        continuity_issues,
        validation_issues,
        execution_passed,
        semantic_issues,
        artifact_issues,
    )

    turn_passed = (
        format_valid
        and not grounding_issues
        and not continuity_issues
        and not forbidden_issues
        and not validation_issues
        and execution_passed
        and not semantic_issues
        and not artifact_issues
    )

    return {
        "turn_id": turn["turn_id"],
        "prompt": prompt,
        "response": response,
        "code": code,
        "expected_result_json": expected_result,
        "actual_result_json": actual_result,
        "format_valid": format_valid,
        "format_score": format_score,
        "grounding_score": grounding_score,
        "continuity_score": continuity_score,
        "execution_score": execution_score,
        "semantic_score": semantic_score,
        "artifact_score": artifact_score,
        "score_total": score_total,
        "turn_passed": turn_passed,
        "execution_passed": execution_passed,
        "execution_output": execution_output,
        "artifact_paths": execution["image_artifacts"] if execution else [],
        "artifact_passed": artifact_ok,
        "semantic_passed": semantic_ratio == 1.0,
        "issues": all_issues,
        "failure_categories": failure_categories,
        "result_json_parse_error": execution["result_json_parse_error"] if execution else "Execution not attempted.",
        "used_template_fallback": False,
    }


def _persist_turn_artifacts(
    run_root: Path,
    model_key: str,
    scenario_id: str,
    turn_id: int,
    turn_result: Dict[str, Any],
) -> None:
    artifact_dir = run_root / "raw" / model_key / scenario_id / f"turn_{turn_id:02d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "prompt.txt").write_text(turn_result["prompt"], encoding="utf-8")
    (artifact_dir / "response.md").write_text(turn_result["response"], encoding="utf-8")
    (artifact_dir / "generated_code.py").write_text(turn_result["code"], encoding="utf-8")
    (artifact_dir / "execution_output.txt").write_text(turn_result["execution_output"] or "", encoding="utf-8")
    (artifact_dir / "expected_result.json").write_text(
        json.dumps(turn_result["expected_result_json"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "actual_result.json").write_text(
        json.dumps(turn_result["actual_result_json"], indent=2, sort_keys=True) if turn_result["actual_result_json"] is not None else "null",
        encoding="utf-8",
    )
    (artifact_dir / "turn_result.json").write_text(
        json.dumps(turn_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )


async def run_verification_suite(
    api_key: str,
    scenario_count: int,
    model_keys: List[str],
    output_root: Path,
    base_model: str,
    fine_tuned_model: str,
    suite_name: str,
) -> Dict[str, Any]:
    if suite_name == "open_generalization":
        scenarios = build_open_generalization_suite()[: max(0, scenario_count)]
    else:
        scenarios = build_verification_suite(scenario_count)
    run_root = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "scenario_suite.json").write_text(
        json.dumps(scenarios, indent=2),
        encoding="utf-8",
    )
    scenario_oracles = {
        scenario["scenario_id"]: compute_oracle_for_scenario(scenario)
        for scenario in scenarios
    }

    results: Dict[str, Any] = {
        "run_root": str(run_root),
        "suite_name": suite_name,
        "scenario_count": len(scenarios),
        "models": {},
    }

    for model_key in model_keys:
        model_scenarios: List[Dict[str, Any]] = []
        for scenario_index, scenario in enumerate(scenarios, start=1):
            chatbot = await create_chatbot(model_key, api_key, base_model, fine_tuned_model)
            oracle_results = scenario_oracles[scenario["scenario_id"]]
            scenario_turn_results: List[Dict[str, Any]] = []
            try:
                for turn, expected_result in zip(scenario["turns"], oracle_results):
                    workspace_dir = _prepare_workspace(run_root, model_key, scenario, turn["turn_id"])
                    prompt = turn["prompt"]
                    if scenario["case_source"] == "uploaded":
                        prompt = _build_uploaded_runtime_context(prompt, scenario["uploaded_filename"], workspace_dir)

                    response = await chatbot.chat(prompt)
                    execution = None
                    code = _extract_python_code(response)
                    if code:
                        execution = _execute_generated_code(code, workspace_dir)

                    turn_result = _evaluate_turn(
                        turn=turn,
                        scenario=scenario,
                        prompt=prompt,
                        response=response,
                        expected_result=expected_result,
                        execution=execution,
                    )
                    turn_result["used_template_fallback"] = bool(
                        getattr(chatbot, "last_response_used_template_fallback", False)
                    )
                    if turn_result["used_template_fallback"]:
                        turn_result["issues"].append("Response used deterministic template fallback.")
                        if "grounding" not in turn_result["failure_categories"]:
                            turn_result["failure_categories"].append("grounding")
                        turn_result["turn_passed"] = False
                    scenario_turn_results.append(turn_result)
                    _persist_turn_artifacts(run_root, model_key, scenario["scenario_id"], turn["turn_id"], turn_result)
            finally:
                close_method = getattr(chatbot, "close", None)
                if callable(close_method):
                    close_method()

            scenario_result = {
                "scenario_id": scenario["scenario_id"],
                "blueprint": scenario["blueprint"],
                "case_family": scenario["case_family"],
                "case_source": scenario["case_source"],
                "source_case_path": scenario["source_case_path"],
                "uploaded_filename": scenario["uploaded_filename"],
                "turns": scenario_turn_results,
            }
            model_scenarios.append(scenario_result)
            scenario_dir = run_root / "raw" / model_key / scenario["scenario_id"]
            scenario_dir.mkdir(parents=True, exist_ok=True)
            (scenario_dir / "scenario_result.json").write_text(
                json.dumps(scenario_result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(
                f"[{MODEL_SPECS[model_key]['label']}] completed scenario {scenario_index}/{len(scenarios)}: "
                f"{scenario['scenario_id']}",
                flush=True,
            )

        results["models"][model_key] = model_scenarios

    results_path = run_root / "verification_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["results_path"] = str(results_path)

    report_paths = generate_reports(results, run_root)
    results["report_paths"] = report_paths
    (run_root / "verification_manifest.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full multi-turn ANDES verification suite across PFAGENT model modes."
    )
    parser.add_argument(
        "--suite",
        choices=["full", "open_generalization"],
        default="full",
        help=f"Which scenario suite to run. `full` defaults to {FULL_SUITE_SCENARIO_COUNT} scenarios; `open_generalization` contains {OPEN_GENERALIZATION_SCENARIO_COUNT}.",
    )
    parser.add_argument(
        "--scenario-count",
        type=int,
        default=FULL_SUITE_SCENARIO_COUNT,
        help=f"How many scenarios to run. Defaults to {FULL_SUITE_SCENARIO_COUNT}.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS.keys()),
        default=list(MODEL_SPECS.keys()),
        help="Model modes to execute.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "verification" / "artifacts"),
        help="Directory for raw results and reports.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base OpenAI model for the Base OpenAI mode.",
    )
    parser.add_argument(
        "--fine-tuned-model",
        default=DEFAULT_FINETUNED_MODEL,
        help="Fine-tuned model id for Fine-tuned and Fine-tuned + RAG modes.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. If omitted, OPENAI_API_KEY will be used.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("An OpenAI API key is required. Pass --api-key or set OPENAI_API_KEY.")

    results = asyncio.run(
        run_verification_suite(
            api_key=api_key,
            scenario_count=args.scenario_count,
            model_keys=args.models,
            output_root=Path(args.output_root),
            base_model=args.base_model,
            fine_tuned_model=args.fine_tuned_model,
            suite_name=args.suite,
        )
    )

    print(f"Verification run completed. Results: {results['results_path']}")
    print(f"Summary report: {results['report_paths']['summary_markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
