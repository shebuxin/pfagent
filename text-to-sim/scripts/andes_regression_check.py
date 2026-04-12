import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TEXT_TO_SIM_ROOT.parent
DEFAULT_RUNTIME_DIR = TEXT_TO_SIM_ROOT / "regression_runtime"
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

import andes
from src.andes_manual import bootstrap_default_andes_manual
from src.chatbot_factory import DEFAULT_FINETUNED_MODEL
from src.chatbots.openai.graphrag_chatbot import GraphRAGChatbot, GraphRAGConfig
from src.chatbots.openai.rag_chatbot import (
    RAGChatbot,
    RAGConfig,
    build_andes_fallback_response,
    validate_response_code,
)


def extract_python_code(response: str) -> str:
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    return match.group(1).strip() if match else ""


def build_uploaded_runtime_context(user_prompt: str, uploaded_files: List[str]) -> str:
    available_files = "\n".join(f"- {name}" for name in uploaded_files)
    return (
        f"{user_prompt}\n\n"
        "Runtime file context:\n"
        f"- Working directory for execution: {DEFAULT_RUNTIME_DIR}\n"
        "- Uploaded files available during execution:\n"
        f"{available_files}\n"
        "- Use these filenames directly in generated Python code when needed.\n"
        "- Case-loading rule: if using an uploaded file, load it directly with andes.load(\"<exact_filename>\", ...), and do NOT wrap it with andes.get_case(...).\n"
        "- Case-loading rule: only use andes.get_case(...) for ANDES built-in benchmark cases.\n"
        "- Preferred uploaded-case template: script_dir=os.getcwd(); case=os.path.join(script_dir, \"<exact_filename>\"); ssa=andes.load(case, setup=True, no_output=True, log=False)"
    )


def validate_scenario_response(
    scenario: Dict[str, Any],
    contextual_prompt: str,
    response: str,
) -> Tuple[bool, List[str], str]:
    code = extract_python_code(response)
    issues: List[str] = []

    if not code:
        issues.append("No Python code block was returned.")
        return False, issues, code

    is_valid, validation_errors = validate_response_code(response, user_context=contextual_prompt)
    if not is_valid:
        issues.extend(validation_errors)

    for required_snippet in scenario.get("required_snippets", []):
        if required_snippet not in code:
            issues.append(f"Missing required snippet: {required_snippet}")

    for required_pattern in scenario.get("required_patterns", []):
        if not re.search(required_pattern, code, re.MULTILINE):
            issues.append(f"Missing required pattern: {required_pattern}")

    for forbidden_snippet in scenario.get("forbidden_snippets", []):
        if forbidden_snippet in code:
            issues.append(f"Forbidden snippet detected: {forbidden_snippet}")

    if scenario.get("disallow_fallback_match", True):
        fallback_response = build_andes_fallback_response(contextual_prompt)
        fallback_code = extract_python_code(fallback_response)
        if fallback_code and re.sub(r"\s+", "", fallback_code) == re.sub(r"\s+", "", code):
            issues.append("Response matched the deterministic fallback template instead of model-personalized generation.")

    return len(issues) == 0, issues, code


def get_runtime_upload_sources() -> Dict[str, str]:
    return {
        "uploaded_ieee39.xlsx": "ieee39/ieee39.xlsx",
        "grid39_for_review.xlsx": "ieee39/ieee39.xlsx",
        "alt39_profile.xlsx": "ieee39/ieee39.xlsx",
        "case39_minmax.xlsx": "ieee39/ieee39.xlsx",
        "study_ieee14_uploaded.xlsx": "ieee14/ieee14_full.xlsx",
        "alt14_review.xlsx": "ieee14/ieee14_full.xlsx",
        "north_ieee14_case.xlsx": "ieee14/ieee14_full.xlsx",
        "kundur_uploaded_study.xlsx": "kundur/kundur_full.xlsx",
        "pjm5_uploaded.json": "5bus/pjm5bus.json",
    }


def build_builtin_forbidden_snippets() -> List[str]:
    return list(get_runtime_upload_sources().keys()) + ["unittest"]


def build_uploaded_forbidden_snippets(uploaded_filename: str) -> List[str]:
    return [
        *[name for name in get_runtime_upload_sources() if name != uploaded_filename],
        f'andes.get_case("{uploaded_filename}")',
        f"andes.get_case('{uploaded_filename}')",
        "unittest",
    ]


def build_scenarios() -> List[Dict[str, Any]]:
    uploaded_case = "uploaded_ieee39.xlsx"
    uploaded_case_ieee14 = "study_ieee14_uploaded.xlsx"
    uploaded_case_review = "grid39_for_review.xlsx"
    uploaded_case_ieee39_alt = "alt39_profile.xlsx"
    uploaded_case_ieee39_minmax = "case39_minmax.xlsx"
    uploaded_case_ieee14_alt = "alt14_review.xlsx"
    uploaded_case_ieee14_custom = "north_ieee14_case.xlsx"
    uploaded_case_kundur = "kundur_uploaded_study.xlsx"
    uploaded_case_pjm = "pjm5_uploaded.json"
    builtin_forbidden_uploads = build_builtin_forbidden_snippets()
    return [
        {
            "name": "builtin_voltage_summary",
            "prompt": "Generate runnable Python code only. Use an ANDES built-in IEEE 14 case, run power flow, and print slack bus voltage plus the top-3 highest bus voltages.",
            "required_snippets": ["andes.get_case(", "ieee14/ieee14_full.xlsx", "PFlow.run()", "Slack"],
            "forbidden_snippets": builtin_forbidden_uploads + ["unittest"],
            "uploaded_files": [],
        },
        {
            "name": "builtin_add_load_before_setup_personalized",
            "prompt": "Generate runnable Python code only. Use the built-in kundur_full case, add one new PQ load at bus 9 before setup with p0=0.02 and q0=0.015, run power flow, and report any bus voltage outside [0.94, 1.06].",
            "required_snippets": ["andes.get_case(", "kundur/kundur_full.xlsx", ".add(", ".setup()", "PFlow.run()"],
            "required_patterns": [r"\b9\b", r"\b0\.02\b", r"\b0\.015\b", r"\b0\.94\b", r"\b1\.06\b"],
            "forbidden_snippets": builtin_forbidden_uploads + ["unittest"],
            "uploaded_files": [],
        },
        {
            "name": "builtin_line_angle_top3",
            "prompt": "Generate runnable Python code only. Use the built-in pjm5bus case, run power flow, and print the top-3 lines by absolute sending-end phase angle.",
            "required_snippets": ["andes.get_case(", "5bus/pjm5bus.json", "PFlow.run()", "Line"],
            "required_patterns": [r"Line\.a1\.e", r"argsort|sorted|argpartition", r"\b3\b"],
            "forbidden_snippets": builtin_forbidden_uploads + ["unittest"],
            "uploaded_files": [],
        },
        {
            "name": "builtin_ieee39_low_voltage_filter",
            "prompt": "Generate runnable Python code only. Use the built-in IEEE 39 case, run power flow, and print every bus below 0.95 p.u.",
            "required_snippets": ["andes.get_case(", "ieee39/ieee39.xlsx", "PFlow.run()"],
            "required_patterns": [r"\b0\.95\b", r"Bus\.v\.v|bus_v"],
            "forbidden_snippets": builtin_forbidden_uploads + ["unittest"],
            "uploaded_files": [],
        },
        {
            "name": "builtin_ieee14_above_threshold_count",
            "prompt": "Generate runnable Python code only. Use the built-in IEEE 14 full case, run power flow, and count how many buses are above 1.02 p.u.",
            "required_snippets": ["andes.get_case(", "ieee14/ieee14_full.xlsx", "PFlow.run()"],
            "required_patterns": [r"\b1\.02\b", r"len\(|sum\(", r"Bus\.v\.v|bus_v"],
            "forbidden_snippets": builtin_forbidden_uploads + ["unittest"],
            "uploaded_files": [],
        },
        {
            "name": "builtin_ieee14_two_lowest_buses",
            "prompt": "Generate runnable Python code only. Use the built-in IEEE 14 full case, run power flow, and print the two lowest-voltage buses.",
            "required_snippets": ["andes.get_case(", "ieee14/ieee14_full.xlsx", "PFlow.run()"],
            "required_patterns": [r"argsort|sorted", r"\b2\b", r"Bus\.v\.v|bus_v"],
            "forbidden_snippets": builtin_forbidden_uploads + ["unittest"],
            "uploaded_files": [],
        },
        {
            "name": "builtin_kundur_add_load_min_voltage_bus",
            "prompt": "Generate runnable Python code only. Use the built-in kundur_full case, add a new PQ load at bus 4 before setup with p0=0.03 and q0=0.01, run power flow, and print the minimum-voltage bus.",
            "required_snippets": ["andes.get_case(", "kundur/kundur_full.xlsx", ".add(", ".setup()", "PFlow.run()"],
            "required_patterns": [r"\b4\b", r"\b0\.03\b", r"\b0\.01\b", r"argmin|min\("],
            "forbidden_snippets": builtin_forbidden_uploads + ["unittest"],
            "uploaded_files": [],
        },
        {
            "name": "builtin_ieee39_voltage_plot",
            "prompt": "Generate runnable Python code only. Use the built-in IEEE 39 case, run power flow, and plot the bus voltage profile.",
            "required_snippets": ["andes.get_case(", "ieee39/ieee39.xlsx", "PFlow.run()"],
            "required_patterns": [r"plt\.plot\(|\.plot\(", r"Bus\.v\.v|bus_v"],
            "forbidden_snippets": builtin_forbidden_uploads + ["unittest"],
            "uploaded_files": [],
        },
        {
            "name": "builtin_pjm5bus_line_angle_threshold",
            "prompt": "Generate runnable Python code only. Use the built-in pjm5bus case, run power flow, and print every line whose absolute sending-end phase angle is above 0.10 radians.",
            "required_snippets": ["andes.get_case(", "5bus/pjm5bus.json", "PFlow.run()", "Line"],
            "required_patterns": [r"Line\.a1\.e", r"\b0\.10\b|\b0\.1\b", r"if|where"],
            "forbidden_snippets": builtin_forbidden_uploads + ["unittest"],
            "uploaded_files": [],
        },
        {
            "name": "builtin_ieee39_slack_and_two_lowest",
            "prompt": "Generate runnable Python code only. Use the built-in IEEE 39 case, run power flow, and print the slack bus voltage together with the two lowest-voltage buses.",
            "required_snippets": ["andes.get_case(", "ieee39/ieee39.xlsx", "PFlow.run()", "Slack"],
            "required_patterns": [r"argsort|sorted", r"\b2\b", r"Slack"],
            "forbidden_snippets": builtin_forbidden_uploads + ["unittest"],
            "uploaded_files": [],
        },
        {
            "name": "uploaded_voltage_extremes",
            "prompt": f"Generate runnable Python code only. Use my uploaded file {uploaded_case} in current directory, run power flow, and print the buses with the maximum and minimum voltages.",
            "required_snippets": [uploaded_case, "andes.load(", "PFlow.run()"],
            "forbidden_snippets": build_uploaded_forbidden_snippets(uploaded_case),
            "uploaded_files": [uploaded_case],
        },
        {
            "name": "uploaded_voltage_plot",
            "prompt": f"Generate runnable Python code only. Use my uploaded file {uploaded_case}, run power flow, and plot the voltage profile.",
            "required_snippets": [uploaded_case, "andes.load(", "PFlow.run()"],
            "required_patterns": [r"\bplt\.plot\(|\.plot\("],
            "forbidden_snippets": build_uploaded_forbidden_snippets(uploaded_case),
            "uploaded_files": [uploaded_case],
        },
        {
            "name": "uploaded_ieee14_threshold_filter",
            "prompt": f"Generate runnable Python code only. Use my uploaded file {uploaded_case_ieee14}, run power flow, and print every bus below 1.0 p.u.",
            "required_snippets": [uploaded_case_ieee14, "andes.load(", "PFlow.run()"],
            "required_patterns": [r"\b1\.0\b", r"Bus\.v\.v|bus_v"],
            "forbidden_snippets": build_uploaded_forbidden_snippets(uploaded_case_ieee14),
            "uploaded_files": [uploaded_case_ieee14],
        },
        {
            "name": "uploaded_slack_and_min_voltage",
            "prompt": f"Generate runnable Python code only. Use my uploaded file {uploaded_case_review}, run power flow, and print the slack bus voltage together with the minimum-voltage bus.",
            "required_snippets": [uploaded_case_review, "andes.load(", "PFlow.run()", "Slack"],
            "required_patterns": [r"argmin|min\(", r"Slack"],
            "forbidden_snippets": build_uploaded_forbidden_snippets(uploaded_case_review),
            "uploaded_files": [uploaded_case_review],
        },
        {
            "name": "uploaded_kundur_add_load_personalized",
            "prompt": f"Generate runnable Python code only. Use my uploaded file {uploaded_case_kundur}, add a PQ load at bus 6 before setup with p0=0.025 and q0=0.02, run power flow, and report buses outside [0.93, 1.07].",
            "required_snippets": [uploaded_case_kundur, "andes.load(", ".add(", ".setup()", "PFlow.run()"],
            "required_patterns": [r"\b6\b", r"\b0\.025\b", r"\b0\.02\b", r"\b0\.93\b", r"\b1\.07\b"],
            "forbidden_snippets": build_uploaded_forbidden_snippets(uploaded_case_kundur),
            "uploaded_files": [uploaded_case_kundur],
        },
        {
            "name": "uploaded_pjm5bus_line_angle_top2",
            "prompt": f"Generate runnable Python code only. Use my uploaded file {uploaded_case_pjm}, run power flow, and print the top-2 lines by absolute sending-end phase angle.",
            "required_snippets": [uploaded_case_pjm, "andes.load(", "PFlow.run()", "Line"],
            "required_patterns": [r"Line\.a1\.e", r"argsort|sorted|argpartition", r"\b2\b"],
            "forbidden_snippets": build_uploaded_forbidden_snippets(uploaded_case_pjm),
            "uploaded_files": [uploaded_case_pjm],
        },
        {
            "name": "uploaded_ieee39_above_1_04_count",
            "prompt": f"Generate runnable Python code only. Use my uploaded file {uploaded_case_ieee39_alt}, run power flow, and count how many buses are above 1.04 p.u.",
            "required_snippets": [uploaded_case_ieee39_alt, "andes.load(", "PFlow.run()"],
            "required_patterns": [r"\b1\.04\b", r"len\(|sum\(", r"Bus\.v\.v|bus_v"],
            "forbidden_snippets": build_uploaded_forbidden_snippets(uploaded_case_ieee39_alt),
            "uploaded_files": [uploaded_case_ieee39_alt],
        },
        {
            "name": "uploaded_ieee14_bar_plot",
            "prompt": f"Generate runnable Python code only. Use my uploaded file {uploaded_case_ieee14_alt}, run power flow, and make a bar plot of the bus voltage profile.",
            "required_snippets": [uploaded_case_ieee14_alt, "andes.load(", "PFlow.run()"],
            "required_patterns": [r"plt\.bar\(|\.bar\(", r"Bus\.v\.v|bus_v"],
            "forbidden_snippets": build_uploaded_forbidden_snippets(uploaded_case_ieee14_alt),
            "uploaded_files": [uploaded_case_ieee14_alt],
        },
        {
            "name": "uploaded_ieee39_slack_and_top2_highest",
            "prompt": f"Generate runnable Python code only. Use my uploaded file {uploaded_case_ieee39_minmax}, run power flow, and print the slack bus voltage together with the top-2 highest bus voltages.",
            "required_snippets": [uploaded_case_ieee39_minmax, "andes.load(", "PFlow.run()", "Slack"],
            "required_patterns": [r"argsort|sorted", r"\b2\b", r"Slack"],
            "forbidden_snippets": build_uploaded_forbidden_snippets(uploaded_case_ieee39_minmax),
            "uploaded_files": [uploaded_case_ieee39_minmax],
        },
        {
            "name": "uploaded_ieee14_max_min_custom_filename",
            "prompt": f"Generate runnable Python code only. Use my uploaded file {uploaded_case_ieee14_custom}, run power flow, and print the maximum-voltage bus together with the minimum-voltage bus.",
            "required_snippets": [uploaded_case_ieee14_custom, "andes.load(", "PFlow.run()"],
            "required_patterns": [r"argmax|max\(", r"argmin|min\("],
            "forbidden_snippets": build_uploaded_forbidden_snippets(uploaded_case_ieee14_custom),
            "uploaded_files": [uploaded_case_ieee14_custom],
        },
    ]


def prepare_runtime_files(runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for target_name, source_case_relpath in get_runtime_upload_sources().items():
        source_case = Path(andes.get_case(source_case_relpath))
        if not source_case.exists():
            raise FileNotFoundError(f"Unable to locate built-in ANDES case for upload simulation: {source_case}")
        shutil.copyfile(source_case, runtime_dir / target_name)


def execute_generated_code(code: str, runtime_dir: Path) -> Tuple[bool, str]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    code_path = runtime_dir / "generated_regression_case.py"
    code_path.write_text(code, encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pfagent-regression")

    result = subprocess.run(
        [sys.executable, str(code_path.name)],
        cwd=runtime_dir,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    output = ""
    if result.stdout:
        output += f"STDOUT:\n{result.stdout}\n"
    if result.stderr:
        output += f"STDERR:\n{result.stderr}\n"
    if not output:
        output = "No output captured."

    return result.returncode == 0, output


async def create_chatbot(chatbot_type: str, chat_model: str, allow_template_fallback: bool):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the regression check.")

    if chatbot_type == "rag":
        chatbot = RAGChatbot(
            RAGConfig(
                openai_api_key=api_key,
                chat_model=chat_model,
                allow_template_fallback=allow_template_fallback,
            )
        )
    elif chatbot_type == "graphrag":
        chatbot = GraphRAGChatbot(
            GraphRAGConfig(
                openai_api_key=api_key,
                neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
                neo4j_password=os.environ.get("NEO4J_PASSWORD"),
                chat_model=chat_model,
            )
        )
    else:
        raise ValueError(f"Unsupported chatbot type: {chatbot_type}")

    await bootstrap_default_andes_manual(chatbot)
    chatbot.load_system_prompt(session_id="regression_session", custom_instructions="")
    return chatbot


async def run_regression(
    chatbot_type: str,
    chat_model: str,
    execute_code_flag: bool,
    allow_template_fallback: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    prepare_runtime_files(DEFAULT_RUNTIME_DIR)

    for scenario in build_scenarios():
        chatbot = await create_chatbot(chatbot_type, chat_model, allow_template_fallback=allow_template_fallback)
        try:
            prompt = scenario["prompt"]
            if scenario["uploaded_files"]:
                prompt = build_uploaded_runtime_context(prompt, scenario["uploaded_files"])

            response = await chatbot.chat(prompt)
            passed, issues, code = validate_scenario_response(scenario, prompt, response)
            used_template_fallback = getattr(chatbot, "last_response_used_template_fallback", False)
            if scenario.get("disallow_fallback_match", True) and used_template_fallback:
                issues.append("Response used deterministic template fallback; expected model-personalized generation.")
            execution_passed = None
            execution_output = None

            if execute_code_flag and code:
                try:
                    execution_passed, execution_output = execute_generated_code(code, DEFAULT_RUNTIME_DIR)
                    if not execution_passed:
                        issues.append("Generated code failed during execution.")
                except Exception as exc:
                    execution_passed = False
                    execution_output = f"Execution harness error: {exc}"
                    issues.append("Execution harness failed.")

            results.append(
                {
                    "scenario": scenario["name"],
                    "passed": passed and (execution_passed is not False),
                    "issues": issues,
                    "response": response,
                    "code": code,
                    "used_template_fallback": used_template_fallback,
                    "execution_passed": execution_passed,
                    "execution_output": execution_output,
                }
            )
        finally:
            close_method = getattr(chatbot, "close", None)
            if callable(close_method):
                close_method()

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ANDES generation regression checks.")
    parser.add_argument(
        "--chatbot-type",
        choices=["rag", "graphrag"],
        default="rag",
        help="Choose which chatbot implementation to validate.",
    )
    parser.add_argument(
        "--chat-model",
        default=DEFAULT_FINETUNED_MODEL,
        help="Model name to use for generation. Defaults to the repo's fine-tuned model.",
    )
    parser.add_argument(
        "--execute-generated-code",
        action="store_true",
        help="Execute generated code for each scenario after validating the response shape.",
    )
    parser.add_argument(
        "--allow-template-fallback",
        action="store_true",
        help="Allow deterministic fallback templates during regression. Disabled by default so tests measure model-driven generation.",
    )
    parser.add_argument(
        "--output",
        default=str(TEXT_TO_SIM_ROOT / "regression_results.json"),
        help="Path for saving the JSON regression report.",
    )
    args = parser.parse_args()

    results = asyncio.run(
        run_regression(
            args.chatbot_type,
            args.chat_model,
            args.execute_generated_code,
            allow_template_fallback=args.allow_template_fallback,
        )
    )
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed = sum(1 for item in results if item["passed"])
    total = len(results)
    print(f"Regression summary: {passed}/{total} scenarios passed")
    print(f"Detailed report saved to: {output_path}")

    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"- {status}: {item['scenario']}")
        for issue in item["issues"]:
            print(f"  * {issue}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
