from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = TEXT_TO_SIM_ROOT / "data_files" / "agent_evolution_profile.json"


@dataclass(frozen=True)
class MutationPack:
    pack_id: str
    title: str
    prompt_guidance: Sequence[str]
    pattern_overrides: Dict[str, Sequence[str]]
    marker_overrides: Dict[str, Sequence[str]]


@dataclass(frozen=True)
class RootCauseSignature:
    signature_id: str
    label: str
    prompt_patterns: Sequence[str]
    execution_patterns: Sequence[str]
    issue_patterns: Sequence[str]
    activate_packs: Sequence[str]


MUTATION_LIBRARY: Dict[str, MutationPack] = {
    "targeted_device_resolution": MutationPack(
        pack_id="targeted_device_resolution",
        title="Resolve targeted devices by bus before editing",
        prompt_guidance=[
            "When the user identifies a demand or PV record by bus, resolve the matching device idx from the real ANDES arrays before calling `.set(...)`.",
            "Do not treat the bus number or array position as the device idx for `PQ`, `PV`, or `Line` edits.",
            "When modifying an existing case, inspect the uploaded case preview or ANDES arrays first instead of guessing idx values.",
        ],
        pattern_overrides={
            "target_pq_bus": [
                r"demand record belongs to bus (\d+)",
                r"record belongs to bus (\d+)",
                r"demand record sits on bus (\d+)",
                r"record sits on bus (\d+)",
                r"load belongs to bus (\d+)",
            ],
            "target_pv_bus": [
                r"generator-side voltage-control record associated with bus (\d+)",
                r"voltage-control record associated with bus (\d+)",
            ],
        },
        marker_overrides={},
    ),
    "pq_percentage_scaling": MutationPack(
        pack_id="pq_percentage_scaling",
        title="Interpret percentage demand changes as scale factors",
        prompt_guidance=[
            "Interpret phrases like `4% heavier` or `3% higher` on the same demand record as multiplying both `p0` and `q0` by `1.04` or `1.03`.",
        ],
        pattern_overrides={
            "target_pq_percent": [
                r"([0-9]*\.?[0-9]+)% heavier",
                r"([0-9]*\.?[0-9]+)% higher",
                r"([0-9]*\.?[0-9]+)% larger",
                r"([0-9]*\.?[0-9]+)% more on both active and reactive",
            ],
        },
        marker_overrides={
            "pq_carry_markers": [
                "same demand record",
                "that same demand record",
                "that demand record",
                "same demand",
                "demand components",
            ]
        },
    ),
    "pv_regulator_aliases": MutationPack(
        pack_id="pv_regulator_aliases",
        title="Handle regulator-target phrasing for PV edits",
        prompt_guidance=[
            "Treat phrases such as `regulator target`, `voltage-control record`, or `generator-side voltage-control record` as PV `v0` adjustment requests.",
        ],
        pattern_overrides={
            "target_pv_v0": [
                r"raise .* regulator target to ([0-9]*\.?[0-9]+)",
                r"raise that regulator target to ([0-9]*\.?[0-9]+)",
                r"move that regulator target to ([0-9]*\.?[0-9]+)",
            ],
        },
        marker_overrides={
            "pv_carry_markers": [
                "that regulator",
                "same voltage-control record",
                "that same voltage-control record",
                "regulator target",
                "that generator voltage-target change",
                "keep that regulator change",
            ]
        },
    ),
    "corridor_outage_aliases": MutationPack(
        pack_id="corridor_outage_aliases",
        title="Interpret corridor phrasing as line outages",
        prompt_guidance=[
            "Treat `corridor between buses A and B`, `trip the branch`, and `out of service` phrasing as line-outage requests that should map to `ssa.Line.set(src=\"u\", idx=[line_id], attr=\"v\", value=[0])` after resolving the real line idx.",
        ],
        pattern_overrides={
            "line_outage_by_pair": [
                r"put the transmission corridor between buses (\d+) and (\d+) out of service",
                r"corridor between buses (\d+) and (\d+) out of service",
                r"knock the (\d+)-(\d+) corridor out of service",
            ]
        },
        marker_overrides={},
    ),
    "n1_outage_set_aliases": MutationPack(
        pack_id="n1_outage_set_aliases",
        title="Interpret outage-set language as N-1 screening",
        prompt_guidance=[
            "Treat `outage set`, `screening set`, or `stressed case` wording as N-1 screening over the listed candidate bus pairs, always restarting from the same modified case.",
        ],
        pattern_overrides={},
        marker_overrides={
            "candidate_line_markers": [
                "outage set",
                "screening set",
                "stressed case",
            ],
            "structured_activation_markers": [
                "outage set",
                "screening set",
                "stressed case",
            ],
        },
    ),
    "string_device_idx_guardrail": MutationPack(
        pack_id="string_device_idx_guardrail",
        title="Preserve string-valued ANDES device identifiers",
        prompt_guidance=[
            "ANDES device identifiers such as `PQ_2` or `Line_3` may be strings; preserve them as returned by ANDES instead of forcing integer conversion.",
        ],
        pattern_overrides={},
        marker_overrides={},
    ),
    "line_outage_api_guardrail": MutationPack(
        pack_id="line_outage_api_guardrail",
        title="Use line-status APIs instead of guessed PFlow setters",
        prompt_guidance=[
            "For contingency studies, update line status through `ssa.Line.set(src=\"u\", idx=[line_id], attr=\"v\", value=[0])`. Do not invent helpers such as `ssa.PFlow.set(..., attr='in_service', ...)`.",
            "After each outage, inspect `ssa.PFlow.converged`, `ssa.exit_code`, `ssa.Bus.island_sets`, `ssa.Bus.nosw_island`, and `ssa.Bus.n_islanded_buses` before ranking contingencies by bus voltage.",
        ],
        pattern_overrides={},
        marker_overrides={},
    ),
    "runnable_code_contract": MutationPack(
        pack_id="runnable_code_contract",
        title="Return one runnable Python block for code requests",
        prompt_guidance=[
            "When the user asks for code, return exactly one runnable Python code block instead of prose-only instructions.",
            "Do not require the user to reconstruct code manually from mixed narrative text.",
        ],
        pattern_overrides={},
        marker_overrides={},
    ),
    "followup_case_continuity": MutationPack(
        pack_id="followup_case_continuity",
        title="Carry the same case and targeted device across follow-ups",
        prompt_guidance=[
            "For follow-up requests, continue from the same case, same targeted device, and same prior edits unless the user explicitly switches to a new case.",
        ],
        pattern_overrides={},
        marker_overrides={
            "structured_activation_markers": [
                "same case",
                "same line",
                "keep the earlier change",
                "keep the previous change",
                "same demand record",
                "same regulator",
            ],
        },
    ),
}


ROOT_CAUSE_SIGNATURES: Sequence[RootCauseSignature] = (
    RootCauseSignature(
        signature_id="device_idx_cast_to_int",
        label="String device idx was cast to int",
        prompt_patterns=[],
        execution_patterns=[
            r"invalid literal for int\(\) with base 10: (?:np\.str_\()?['\"]PQ_",
            r"invalid literal for int\(\) with base 10: (?:np\.str_\()?['\"]PV_",
            r"invalid literal for int\(\) with base 10: (?:np\.str_\()?['\"]Line_",
        ],
        issue_patterns=[],
        activate_packs=("string_device_idx_guardrail",),
    ),
    RootCauseSignature(
        signature_id="positional_idx_used_as_device_idx",
        label="Bus number or array index was used as device idx",
        prompt_patterns=[],
        execution_patterns=[r"device not exist with idx=\d+"],
        issue_patterns=[
            r"wrong device idx",
            r"inspect (?:the )?case",
            r"look(?:ed)? up (?:the )?case",
            r"guess(?:ed)? idx",
            r"used bus number as device idx",
        ],
        activate_packs=("targeted_device_resolution", "string_device_idx_guardrail"),
    ),
    RootCauseSignature(
        signature_id="open_ended_pq_percentage_language",
        label="Open-ended percentage demand phrasing was not grounded",
        prompt_patterns=[r"% heavier", r"% higher", r"same demand record"],
        execution_patterns=[],
        issue_patterns=[r"heavier load", r"percentage demand", r"same demand record"],
        activate_packs=("pq_percentage_scaling", "targeted_device_resolution"),
    ),
    RootCauseSignature(
        signature_id="open_ended_pv_regulator_language",
        label="Open-ended regulator phrasing was not grounded",
        prompt_patterns=[r"regulator target", r"voltage-control record associated", r"generator-side voltage-control record"],
        execution_patterns=[],
        issue_patterns=[r"pv_bus: expected", r"pv_idx: expected", r"regulator", r"voltage-control"],
        activate_packs=("pv_regulator_aliases", "targeted_device_resolution"),
    ),
    RootCauseSignature(
        signature_id="corridor_outage_language",
        label="Corridor phrasing was not mapped to line outage status edits",
        prompt_patterns=[r"corridor", r"out of service"],
        execution_patterns=[r"PFlow' object has no attribute 'set'", r"KeyError: 'in_service'"],
        issue_patterns=[r"corridor outage", r"branch trip", r"out of service"],
        activate_packs=("corridor_outage_aliases", "line_outage_api_guardrail"),
    ),
    RootCauseSignature(
        signature_id="n1_outage_set_language",
        label="Outage-set language was not mapped to N-1 screening",
        prompt_patterns=[r"outage set", r"screening set", r"stressed case"],
        execution_patterns=[r"PFlow' object has no attribute 'set'"],
        issue_patterns=[r"n-1", r"outage set", r"screening set", r"stressed case"],
        activate_packs=("n1_outage_set_aliases", "line_outage_api_guardrail"),
    ),
    RootCauseSignature(
        signature_id="response_not_runnable",
        label="Response was not directly runnable",
        prompt_patterns=[],
        execution_patterns=[],
        issue_patterns=[r"not runnable", r"could not run", r"no code block", r"plain text", r"response not runnable"],
        activate_packs=("runnable_code_contract",),
    ),
    RootCauseSignature(
        signature_id="followup_context_lost",
        label="Follow-up continuity was lost across turns",
        prompt_patterns=[],
        execution_patterns=[],
        issue_patterns=[r"forgot previous", r"follow-up continuity", r"lost prior state", r"same case", r"keep previous change"],
        activate_packs=("followup_case_continuity",),
    ),
)


def _default_profile() -> Dict[str, Any]:
    return {
        "profile_version": "manual-default",
        "source_runs": [],
        "active_mutation_packs": [],
        "prompt_guidance": [],
        "pattern_overrides": {},
        "marker_overrides": {},
        "root_cause_summary": [],
    }


def load_agent_evolution_profile(profile_path: Path | None = None) -> Dict[str, Any]:
    path = profile_path or DEFAULT_PROFILE_PATH
    if not path.exists():
        return _default_profile()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _default_profile()
    if not isinstance(data, dict):
        return _default_profile()
    merged = _default_profile()
    merged.update(data)
    return merged


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def get_profile_pattern_overrides(key: str, defaults: Sequence[str], profile_path: Path | None = None) -> List[str]:
    profile = load_agent_evolution_profile(profile_path)
    extras = profile.get("pattern_overrides", {}).get(key, [])
    return _dedupe(list(defaults) + list(extras))


def get_profile_marker_overrides(key: str, defaults: Sequence[str], profile_path: Path | None = None) -> List[str]:
    profile = load_agent_evolution_profile(profile_path)
    extras = profile.get("marker_overrides", {}).get(key, [])
    return _dedupe(list(defaults) + list(extras))


def build_adaptive_guidance_section(profile_path: Path | None = None) -> str:
    profile = load_agent_evolution_profile(profile_path)
    bullets = _dedupe(profile.get("prompt_guidance", []))
    if not bullets:
        return ""
    return "Adaptive Evolution Rules:\n" + "\n".join(f"- {bullet}" for bullet in bullets)


def _iter_turn_results(results: Dict[str, Any]) -> Iterable[tuple[str, Dict[str, Any], Dict[str, Any]]]:
    for model_key, scenarios in results.get("models", {}).items():
        for scenario in scenarios:
            for turn in scenario.get("turns", []):
                yield model_key, scenario, turn


def _signature_matches(signature: RootCauseSignature, turn: Dict[str, Any]) -> bool:
    prompt = turn.get("prompt", "")
    execution_output = turn.get("execution_output", "") or ""
    issues_text = "\n".join(turn.get("issues", []))
    matched_prompt = any(re.search(pattern, prompt, flags=re.IGNORECASE) for pattern in signature.prompt_patterns)
    matched_execution = any(re.search(pattern, execution_output, flags=re.IGNORECASE) for pattern in signature.execution_patterns)
    matched_issues = any(re.search(pattern, issues_text, flags=re.IGNORECASE) for pattern in signature.issue_patterns)
    return matched_prompt or matched_execution or matched_issues


def _build_evolution_profile_from_records(
    failure_records: Sequence[Dict[str, Any]],
    *,
    profile_version: str,
    source_runs: Sequence[str],
) -> Dict[str, Any]:
    root_cause_counter: Counter[str] = Counter()
    root_cause_examples: Dict[str, List[str]] = defaultdict(list)
    active_packs: set[str] = set()

    for record in failure_records:
        if record.get("turn_passed", False):
            continue
        scenario_id = record.get("scenario_id", "runtime_feedback")
        turn_id = int(record.get("turn_id", 0) or 0)
        for signature in ROOT_CAUSE_SIGNATURES:
            if not _signature_matches(signature, record):
                continue
            root_cause_counter[signature.signature_id] += 1
            if len(root_cause_examples[signature.signature_id]) < 3:
                root_cause_examples[signature.signature_id].append(
                    f"{scenario_id}/turn_{turn_id:02d}"
                )
            active_packs.update(signature.activate_packs)

    prompt_guidance: List[str] = []
    pattern_overrides: Dict[str, List[str]] = defaultdict(list)
    marker_overrides: Dict[str, List[str]] = defaultdict(list)
    for pack_id in sorted(active_packs):
        pack = MUTATION_LIBRARY[pack_id]
        prompt_guidance.extend(pack.prompt_guidance)
        for key, patterns in pack.pattern_overrides.items():
            pattern_overrides[key].extend(patterns)
        for key, markers in pack.marker_overrides.items():
            marker_overrides[key].extend(markers)

    root_cause_summary = []
    for signature in ROOT_CAUSE_SIGNATURES:
        count = int(root_cause_counter.get(signature.signature_id, 0))
        if count == 0:
            continue
        root_cause_summary.append(
            {
                "signature_id": signature.signature_id,
                "label": signature.label,
                "count": count,
                "example_turns": root_cause_examples[signature.signature_id],
                "activated_packs": list(signature.activate_packs),
            }
        )

    return {
        "profile_version": profile_version,
        "source_runs": list(source_runs),
        "active_mutation_packs": sorted(active_packs),
        "prompt_guidance": _dedupe(prompt_guidance),
        "pattern_overrides": {key: _dedupe(values) for key, values in pattern_overrides.items()},
        "marker_overrides": {key: _dedupe(values) for key, values in marker_overrides.items()},
        "root_cause_summary": root_cause_summary,
    }


def build_evolution_profile_from_failures(
    failure_records: Sequence[Dict[str, Any]],
    *,
    profile_version: str,
    source_runs: Sequence[str] | None = None,
) -> Dict[str, Any]:
    return _build_evolution_profile_from_records(
        failure_records,
        profile_version=profile_version,
        source_runs=list(source_runs or []),
    )


def build_evolution_profile_from_results(
    results_paths: Sequence[Path],
    *,
    profile_version: str,
) -> Dict[str, Any]:
    failure_records: List[Dict[str, Any]] = []
    for results_path in results_paths:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        for _model_key, scenario, turn in _iter_turn_results(results):
            if turn.get("turn_passed", False):
                continue
            failure_records.append(
                {
                    "scenario_id": scenario.get("scenario_id", "verification"),
                    "turn_id": turn.get("turn_id", 0),
                    "prompt": turn.get("prompt", ""),
                    "execution_output": turn.get("execution_output", "") or "",
                    "issues": list(turn.get("issues", [])),
                    "turn_passed": False,
                }
            )
    return _build_evolution_profile_from_records(
        failure_records,
        profile_version=profile_version,
        source_runs=[str(path) for path in results_paths],
    )


def merge_evolution_profiles(
    base_profile: Dict[str, Any],
    delta_profile: Dict[str, Any],
    *,
    profile_version: str | None = None,
) -> Dict[str, Any]:
    merged = _default_profile()
    merged["profile_version"] = profile_version or delta_profile.get("profile_version") or base_profile.get("profile_version") or "merged"
    merged["source_runs"] = _dedupe(list(base_profile.get("source_runs", [])) + list(delta_profile.get("source_runs", [])))
    merged["active_mutation_packs"] = _dedupe(
        list(base_profile.get("active_mutation_packs", [])) + list(delta_profile.get("active_mutation_packs", []))
    )
    merged["prompt_guidance"] = _dedupe(
        list(base_profile.get("prompt_guidance", [])) + list(delta_profile.get("prompt_guidance", []))
    )

    pattern_overrides: Dict[str, List[str]] = defaultdict(list)
    for profile in (base_profile, delta_profile):
        for key, values in profile.get("pattern_overrides", {}).items():
            pattern_overrides[key].extend(values)
    merged["pattern_overrides"] = {key: _dedupe(values) for key, values in pattern_overrides.items()}

    marker_overrides: Dict[str, List[str]] = defaultdict(list)
    for profile in (base_profile, delta_profile):
        for key, values in profile.get("marker_overrides", {}).items():
            marker_overrides[key].extend(values)
    merged["marker_overrides"] = {key: _dedupe(values) for key, values in marker_overrides.items()}

    root_summary: Dict[str, Dict[str, Any]] = {}
    for profile in (base_profile, delta_profile):
        for item in profile.get("root_cause_summary", []):
            signature_id = item.get("signature_id")
            if not signature_id:
                continue
            existing = root_summary.setdefault(
                signature_id,
                {
                    "signature_id": signature_id,
                    "label": item.get("label", signature_id),
                    "count": 0,
                    "example_turns": [],
                    "activated_packs": [],
                },
            )
            existing["count"] += int(item.get("count", 0))
            existing["example_turns"] = _dedupe(existing["example_turns"] + list(item.get("example_turns", [])))[:5]
            existing["activated_packs"] = _dedupe(existing["activated_packs"] + list(item.get("activated_packs", [])))
    merged["root_cause_summary"] = list(root_summary.values())
    return merged


def save_evolution_profile(profile: Dict[str, Any], profile_path: Path | None = None) -> Path:
    path = profile_path or DEFAULT_PROFILE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    return path
