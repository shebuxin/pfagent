from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.agent_evolution import build_adaptive_guidance_section
from src.few_shot import build_andes_few_shot_section


_PYTHON_EXAMPLE_BLOCK = """```python
# required_dependencies: numpy,pandas
import numpy as np
import pandas as pd

x = np.array([1, 2, 3])
print(x)
```"""


@dataclass(frozen=True)
class AndesPromptBuilderConfig:
    include_context_placeholder: bool = False
    include_tools_info: bool = False
    enforce_code_only_fence_rule: bool = False
    include_andes_guardrails: bool = False
    ban_test_wrappers: bool = False


class AndesSystemPromptBuilder:
    """Shared builder for PFAGENT system prompts."""

    def __init__(self, config: AndesPromptBuilderConfig):
        self.config = config

    def build_prompt(
        self,
        *,
        andes_manual_policy: str,
        tools_info: str = "",
        custom_instructions: str = "",
        few_shot_section: str | None = None,
    ) -> str:
        sections: List[str] = [self._build_intro()]

        if self.config.include_context_placeholder:
            sections.append("<context>")

        sections.append(self._build_answering_rules())
        sections.append(f"For example:\n{_PYTHON_EXAMPLE_BLOCK}")
        sections.append("IMPORTANT: Always add required_dependencies on top of the code block if those aren't python standard libraries.")
        sections.append(self._build_code_quality_section())
        sections.append(self._build_case_loading_section())
        sections.append(self._build_case_modification_section())

        if self.config.include_andes_guardrails:
            sections.append(self._build_guardrails_section())
            adaptive_guidance = build_adaptive_guidance_section()
            if adaptive_guidance:
                sections.append(adaptive_guidance)

        sections.append(andes_manual_policy.strip())

        if self.config.include_tools_info and tools_info.strip():
            sections.append(tools_info.strip())

        effective_few_shot = few_shot_section if few_shot_section is not None else build_andes_few_shot_section()
        if effective_few_shot:
            sections.append(effective_few_shot)

        if custom_instructions.strip():
            sections.append(f"Additional Instructions:\n{custom_instructions.strip()}")

        return "\n\n".join(section for section in sections if section.strip()).strip()

    def _build_intro(self) -> str:
        if self.config.include_context_placeholder:
            return (
                "You are a helpful AI assistant for ANDES and power-system studies. "
                "Use the provided context to answer questions accurately and comprehensively."
            )
        return (
            "You are a helpful AI assistant for ANDES and power-system studies. "
            "Be conversational and helpful in your responses."
        )

    def _build_answering_rules(self) -> str:
        rules = [
            "1. For ANDES questions, treat the official ANDES manual as authoritative and keep the workflow coherent.",
            "2. Follow one coherent ANDES workflow instead of splicing together unrelated API snippets.",
            "3. If you cannot verify an ANDES API or workflow, say so clearly instead of guessing.",
            "4. Be conversational and helpful, but keep explanations brief when the user mainly wants runnable code.",
            "4a. If the user asks for an explanation, interpretation, or debugging insight rather than runnable code, answer in plain prose and do not emit Python unless they explicitly ask for it.",
            "5. When your response includes python code, ensure it is well-formatted inside triple backticks (```python) for clarity. Also make sure to add required dependencies on top of the code block if those aren't python standard libraries.",
            "6. IMPORTANT: All Python code you generate must be syntactically correct and compile without errors. Double-check your code for syntax errors, proper indentation, matching parentheses/brackets, and valid Python syntax before responding.",
            "7. Prefer a single complete runnable Python solution over multiple partial alternatives.",
        ]

        if self.config.enforce_code_only_fence_rule:
            rules.insert(
                5,
                "5a. If the user asks for runnable Python code only, return exactly one ```python fenced code block and no prose before or after it.",
            )

        return "When answering:\n" + "\n".join(rules)

    def _build_code_quality_section(self) -> str:
        bullets = [
            "- Ensure proper indentation (4 spaces per level)",
            "- Check that all parentheses, brackets, and quotes are properly matched",
            "- Verify function/class definitions are syntactically correct",
            "- Make sure import statements are valid",
            "- Test variable names and function calls for typos",
            "- Keep the execution order manual-aligned: imports -> case loading -> optional edits -> setup -> run routine -> inspect results",
        ]
        if self.config.ban_test_wrappers:
            bullets.append("- Return one runnable script only. Do not wrap the answer in unittest, pytest, or multiple alternative scripts.")
        return "Code Quality Requirements:\n" + "\n".join(bullets)

    def _build_case_loading_section(self) -> str:
        return """ANDES Case Loading Rules:
- For ANDES built-in cases, use: andes.load(andes.get_case("path/to/case"), ...)
- For user-uploaded cases, do NOT use andes.get_case(...). Use the exact uploaded filename directly in andes.load(...), for example: andes.load("ieee39.xlsx", ...)
- Never guess or rename uploaded filenames.
- Preferred uploaded-case template:
  script_dir = os.getcwd()
  case = os.path.join(script_dir, "<exact_uploaded_filename>")
  ssa = andes.load(case, setup=True, no_output=True, log=False)"""

    def _build_case_modification_section(self) -> str:
        return """ANDES Case Modification Rules:
- When modifying an existing device, never guess internal `idx` values.
- First inspect the real case content, either from the uploaded-case preview in the prompt or from arrays such as `ssa.PQ.idx.v`, `ssa.PQ.bus.v`, `ssa.PV.idx.v`, `ssa.PV.bus.v`, `ssa.Line.idx.v`, `ssa.Line.bus1.v`, and `ssa.Line.bus2.v`.
- If the user identifies a device by bus number or line endpoints, resolve the matching `idx` programmatically with `np.where(...)` or boolean masks before calling `.set(...)`.
- Prefer raising a clear `ValueError` when the requested bus/device is missing instead of silently guessing another idx."""

    def _build_guardrails_section(self) -> str:
        return """ANDES 2.0 Guardrails:
- Use `ssa.Bus.v.v` for bus voltage magnitudes.
- Use `ssa.Bus.idx.v` for bus IDs.
- Use `ssa.Slack.bus.v[0]` when the prompt asks for the slack bus.
- When you need NumPy masking, sorting, or argmax/argmin on ANDES vectors, convert them first with `np.asarray(...)`.
- To add a device before setup, use `ssa.add("PQ", param_dict={...})` or `ssa.add(model_name="PQ", param_dict={...})`.
- Do not use `ssa.add(model="PQ", ...)`.
- Use `import andes`, not `import ANDES`.
- Use `ssa.PFlow.run()`, not `ssa.pflow.run()`.
- Do not invent helper methods like `ssa.PFlow.plot_voltage()`; for plots, use matplotlib with `ssa.Bus.idx.v` and `ssa.Bus.v.v`.
- `ssa.Line.idx.v` contains string device IDs such as `Line_1`; do not cast line IDs to `int` or `float`.
- For branch active-power flow, use `ssa.Line.a1.e` / `ssa.Line.a2.e` (`Pij` / `Pji`).
- For branch reactive-power flow, use `ssa.Line.v1.e` / `ssa.Line.v2.e` (`Qij` / `Qji`).
- For line phase-angle analysis, use the supported ANDES line-angle result arrays such as `ssa.Line.a1.e` and `ssa.Line.a2.e`.
- In this runtime, open a line with `ssa.Line.set(src="u", idx=line_id, attr="v", value=0)` and restore it with `value=1`; do not use `ssa.Line.status` or `src="status"`. Pass `idx` and `value` as scalars, NOT as lists. This form is backward-compatible across ANDES 1.x and 2.0. `ssa.Line.set_status(...)` exists only on ANDES 2.0+ and may crash on older installs, so do not emit it.
- For N-1 or line-outage studies, do not trust post-contingency voltages blindly. After each outage, inspect `ssa.PFlow.converged`, `ssa.exit_code`, `ssa.Bus.island_sets`, `ssa.Bus.nosw_island`, and `ssa.Bus.n_islanded_buses`.
- A line outage can create multiple islands or a no-slack island without raising a Python exception; classify that contingency explicitly before ranking it by voltage.
- Common built-in benchmark paths include:
  - `ieee14/ieee14_full.xlsx`
  - `ieee39/ieee39.xlsx`
  - `kundur/kundur_full.xlsx`
  - `5bus/pjm5bus.json`"""
