from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .agent_evolution import (
    DEFAULT_PROFILE_PATH,
    build_evolution_profile_from_failures,
    load_agent_evolution_profile,
    merge_evolution_profiles,
    save_evolution_profile,
)
from .code_blocks import extract_python_code_segments


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
USER_FEEDBACK_ROOT = TEXT_TO_SIM_ROOT / "data_files" / "user_feedback"
SESSION_LOG_ROOT = USER_FEEDBACK_ROOT / "sessions"
ANALYSIS_ROOT = USER_FEEDBACK_ROOT / "analysis"
MASTER_EVENT_LOG = USER_FEEDBACK_ROOT / "events.jsonl"
MASTER_ANALYSIS_LOG = USER_FEEDBACK_ROOT / "analysis_log.jsonl"

ERROR_TOKENS: Sequence[str] = (
    "Error",
    "Exception",
    "Traceback",
    "SyntaxError",
    "NameError",
    "TypeError",
    "ValueError",
    "ImportError",
    "ModuleNotFoundError",
    "AttributeError",
    "KeyError",
    "IndexError",
    "FileNotFoundError",
    "PermissionError",
    "ZeroDivisionError",
)


def _ensure_directories(base_dir: Path | None = None) -> Dict[str, Path]:
    root = Path(base_dir) if base_dir else USER_FEEDBACK_ROOT
    session_root = root / "sessions"
    analysis_root = root / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    session_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "sessions": session_root,
        "analysis": analysis_root,
        "events": root / "events.jsonl",
        "analysis_log": root / "analysis_log.jsonl",
    }


def _session_log_path(session_id: str, base_dir: Path | None = None) -> Path:
    paths = _ensure_directories(base_dir)
    return paths["sessions"] / f"session_{session_id}.json"


def _analysis_path(session_id: str, base_dir: Path | None = None) -> Path:
    paths = _ensure_directories(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return paths["analysis"] / f"analysis_{session_id}_{timestamp}.json"


def _default_session_log(session_id: str, chatbot_type: str | None = None) -> Dict[str, Any]:
    now = datetime.now().isoformat()
    return {
        "session_id": session_id,
        "created_at": now,
        "last_updated_at": now,
        "chatbot_type": chatbot_type or "",
        "turns": [],
        "executions": [],
        "code_feedback": [],
        "session_feedback": [],
        "analysis_history": [],
    }


def load_session_log(session_id: str, *, base_dir: Path | None = None, chatbot_type: str | None = None) -> Dict[str, Any]:
    path = _session_log_path(session_id, base_dir)
    if not path.exists():
        return _default_session_log(session_id, chatbot_type=chatbot_type)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _default_session_log(session_id, chatbot_type=chatbot_type)
    if not isinstance(data, dict):
        return _default_session_log(session_id, chatbot_type=chatbot_type)
    merged = _default_session_log(session_id, chatbot_type=chatbot_type)
    merged.update(data)
    if chatbot_type and not merged.get("chatbot_type"):
        merged["chatbot_type"] = chatbot_type
    return merged


def save_session_log(session_log: Dict[str, Any], *, base_dir: Path | None = None) -> Path:
    path = _session_log_path(session_log["session_id"], base_dir)
    session_log["last_updated_at"] = datetime.now().isoformat()
    path.write_text(json.dumps(session_log, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _upsert_turn(turns: List[Dict[str, Any]], turn_payload: Dict[str, Any]) -> None:
    for index, existing in enumerate(turns):
        if existing.get("turn_id") == turn_payload.get("turn_id"):
            turns[index] = turn_payload
            return
    turns.append(turn_payload)


def execution_output_has_error(output: str) -> bool:
    normalized = output or ""
    if normalized.startswith("Error (exit code"):
        return True
    return any(token in normalized for token in ERROR_TOKENS)


def _infer_feedback_issues(feedback_text: str, root_cause_hint: str = "", *, has_code_block: bool | None = None) -> List[str]:
    combined = " ".join(part for part in [feedback_text or "", root_cause_hint or ""] if part).strip()
    lowered = combined.lower()
    issues: List[str] = []

    if combined:
        issues.append(combined)

    if has_code_block is False:
        issues.append("response not runnable")
    if any(token in lowered for token in ("not runnable", "can't run", "cannot run", "could not run", "plain text", "no code block")):
        issues.append("response not runnable")
    if any(token in lowered for token in ("wrong idx", "device idx", "inspect the case", "look up the case", "lookup the case", "guess idx")):
        issues.append("wrong device idx / inspect the case")
    if any(token in lowered for token in ("forgot previous", "same case", "keep previous change", "follow-up", "follow up")):
        issues.append("follow-up continuity")
    if any(token in lowered for token in ("corridor", "branch trip", "out of service", "outage")):
        issues.append("corridor outage")
    if any(token in lowered for token in ("regulator", "voltage-control", "voltage control", "pv setpoint")):
        issues.append("regulator target")
    if "%" in combined and any(token in lowered for token in ("heavier", "higher", "load", "demand")):
        issues.append("percentage demand change")

    deduped: List[str] = []
    seen = set()
    for issue in issues:
        normalized = issue.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def record_chat_turn(
    session_id: str,
    *,
    turn_id: int,
    user_message: str,
    assistant_message: str,
    contextual_user_message: str = "",
    chatbot_type: str = "",
    turn_type: str = "user",
    active_case: Dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> Path:
    session_log = load_session_log(session_id, base_dir=base_dir, chatbot_type=chatbot_type)
    turn_payload = {
        "turn_id": int(turn_id),
        "turn_type": turn_type,
        "timestamp": datetime.now().isoformat(),
        "chatbot_type": chatbot_type or session_log.get("chatbot_type", ""),
        "user_message": user_message,
        "contextual_user_message": contextual_user_message,
        "assistant_message": assistant_message,
        "has_python_code": bool(extract_python_code_segments(assistant_message)),
        "active_case": active_case or {},
    }
    _upsert_turn(session_log["turns"], turn_payload)
    path = save_session_log(session_log, base_dir=base_dir)
    _append_jsonl(
        _ensure_directories(base_dir)["events"],
        {
            "event_type": "chat_turn",
            "session_id": session_id,
            **turn_payload,
        },
    )
    return path


def record_execution_result(
    session_id: str,
    *,
    turn_id: int,
    message_index: int,
    code_id: str,
    executed_code: str,
    output: str,
    active_case: Dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> Path:
    session_log = load_session_log(session_id, base_dir=base_dir)
    execution_payload = {
        "event_id": f"{code_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        "turn_id": int(turn_id),
        "message_index": int(message_index),
        "code_id": code_id,
        "timestamp": datetime.now().isoformat(),
        "executed_code": executed_code,
        "output": output,
        "status": "error" if execution_output_has_error(output) else "success",
        "active_case": active_case or {},
    }
    session_log["executions"].append(execution_payload)
    path = save_session_log(session_log, base_dir=base_dir)
    _append_jsonl(
        _ensure_directories(base_dir)["events"],
        {
            "event_type": "execution",
            "session_id": session_id,
            **execution_payload,
        },
    )
    return path


def record_code_feedback(
    session_id: str,
    *,
    turn_id: int,
    message_index: int,
    code_id: str,
    verdict: str,
    feedback_text: str = "",
    root_cause_hint: str = "",
    assistant_message: str = "",
    base_dir: Path | None = None,
) -> Path:
    session_log = load_session_log(session_id, base_dir=base_dir)
    has_code_block = bool(extract_python_code_segments(assistant_message)) if assistant_message else None
    feedback_payload = {
        "event_id": f"{code_id}_{verdict}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        "turn_id": int(turn_id),
        "message_index": int(message_index),
        "code_id": code_id,
        "timestamp": datetime.now().isoformat(),
        "verdict": verdict,
        "feedback_text": feedback_text,
        "root_cause_hint": root_cause_hint,
        "derived_issues": _infer_feedback_issues(feedback_text, root_cause_hint, has_code_block=has_code_block),
    }
    session_log["code_feedback"].append(feedback_payload)
    path = save_session_log(session_log, base_dir=base_dir)
    _append_jsonl(
        _ensure_directories(base_dir)["events"],
        {
            "event_type": "code_feedback",
            "session_id": session_id,
            **feedback_payload,
        },
    )
    return path


def record_session_feedback(
    session_id: str,
    *,
    feedback_payload: Dict[str, Any],
    base_dir: Path | None = None,
) -> Path:
    session_log = load_session_log(session_id, base_dir=base_dir)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "feedback": feedback_payload,
    }
    session_log["session_feedback"].append(payload)
    path = save_session_log(session_log, base_dir=base_dir)
    _append_jsonl(
        _ensure_directories(base_dir)["events"],
        {
            "event_type": "session_feedback",
            "session_id": session_id,
            **payload,
        },
    )
    return path


def _build_failure_records(session_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = {int(turn["turn_id"]): turn for turn in session_log.get("turns", [])}
    failure_records: Dict[int, Dict[str, Any]] = {}

    def ensure_record(turn_id: int) -> Dict[str, Any]:
        turn = turns.get(turn_id, {})
        record = failure_records.setdefault(
            turn_id,
            {
                "scenario_id": f"user_session_{session_log.get('session_id', 'unknown')}",
                "turn_id": turn_id,
                "prompt": turn.get("user_message", ""),
                "execution_output": "",
                "issues": [],
                "turn_passed": False,
            },
        )
        return record

    for execution in session_log.get("executions", []):
        if execution.get("status") != "error":
            continue
        turn_id = int(execution.get("turn_id", 0) or 0)
        if turn_id <= 0:
            continue
        record = ensure_record(turn_id)
        output = execution.get("output", "")
        if output:
            existing = record.get("execution_output", "")
            record["execution_output"] = f"{existing}\n\n{output}".strip()
        record["issues"].append("execution error")

    for feedback in session_log.get("code_feedback", []):
        if feedback.get("verdict") != "failure":
            continue
        turn_id = int(feedback.get("turn_id", 0) or 0)
        if turn_id <= 0:
            continue
        record = ensure_record(turn_id)
        record["issues"].append("user marked failure")
        record["issues"].extend(feedback.get("derived_issues", []))

    for turn_id, turn in turns.items():
        if turn.get("turn_type") != "user":
            failure_records.pop(turn_id, None)
            continue
        if turn_id not in failure_records:
            continue
        if not turn.get("has_python_code", False):
            failure_records[turn_id]["issues"].append("response not runnable")

    normalized: List[Dict[str, Any]] = []
    for turn_id in sorted(failure_records):
        record = failure_records[turn_id]
        deduped_issues: List[str] = []
        seen = set()
        for issue in record.get("issues", []):
            normalized_issue = str(issue).strip()
            if not normalized_issue or normalized_issue in seen:
                continue
            seen.add(normalized_issue)
            deduped_issues.append(normalized_issue)
        record["issues"] = deduped_issues
        normalized.append(record)
    return normalized


def analyze_session_feedback_loop(
    session_id: str,
    *,
    profile_path: Path | None = None,
    base_dir: Path | None = None,
) -> Dict[str, Any]:
    session_log = load_session_log(session_id, base_dir=base_dir)
    session_log_path = _session_log_path(session_id, base_dir)
    failure_records = _build_failure_records(session_log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if failure_records:
        delta_profile = build_evolution_profile_from_failures(
            failure_records,
            profile_version=f"user-feedback-{timestamp}",
            source_runs=[str(session_log_path)],
        )
        base_profile = load_agent_evolution_profile(profile_path)
        merged_profile = merge_evolution_profiles(
            base_profile,
            delta_profile,
            profile_version=f"runtime-feedback-merged-{timestamp}",
        )
        saved_profile_path = save_evolution_profile(merged_profile, profile_path or DEFAULT_PROFILE_PATH)
    else:
        delta_profile = build_evolution_profile_from_failures(
            [],
            profile_version=f"user-feedback-{timestamp}",
            source_runs=[str(session_log_path)],
        )
        merged_profile = load_agent_evolution_profile(profile_path)
        saved_profile_path = profile_path or DEFAULT_PROFILE_PATH

    analysis = {
        "session_id": session_id,
        "analyzed_at": datetime.now().isoformat(),
        "chatbot_type": session_log.get("chatbot_type", ""),
        "failure_turn_count": len(failure_records),
        "failure_records": failure_records,
        "delta_profile": delta_profile,
        "merged_profile_path": str(saved_profile_path),
        "activated_packs": delta_profile.get("active_mutation_packs", []),
        "root_cause_summary": delta_profile.get("root_cause_summary", []),
    }
    analysis_path = _analysis_path(session_id, base_dir)
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    session_log.setdefault("analysis_history", []).append(
        {
            "timestamp": analysis["analyzed_at"],
            "analysis_path": str(analysis_path),
            "failure_turn_count": len(failure_records),
            "activated_packs": delta_profile.get("active_mutation_packs", []),
        }
    )
    save_session_log(session_log, base_dir=base_dir)

    paths = _ensure_directories(base_dir)
    _append_jsonl(paths["analysis_log"], analysis)
    return analysis
