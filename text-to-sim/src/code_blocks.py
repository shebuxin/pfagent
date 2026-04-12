from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import List, Tuple


@dataclass
class ExtractedCodeBlock:
    code: str
    start: int
    end: int
    fenced: bool


_FENCED_BLOCK_PATTERN = re.compile(
    r"```(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\r?\n(?P<code>.*?)(?:\r?\n)?```",
    re.DOTALL,
)


def looks_like_python_script(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or stripped.startswith("```"):
        return False

    signals = (
        "import ",
        "from ",
        "print(",
        "andes.",
        "ssa =",
        "case =",
        "plt.",
        "np.",
        "os.path.join",
        "def ",
        "class ",
    )
    hit_count = sum(1 for signal in signals if signal in stripped)
    return hit_count >= 2 or ("import " in stripped and "\n" in stripped)


def is_code_only_request(user_context: str) -> bool:
    normalized_user_context = (user_context or "").lower()
    if "runnable python code only" in normalized_user_context or "code only" in normalized_user_context:
        return True

    patterns = (
        r"exactly one runnable python code block",
        r"one complete runnable python script only",
        r"one runnable python script only",
        r"single\s+```python",
        r"single\s+`python block",
        r"no prose",
        r"nothing else",
    )
    return any(re.search(pattern, normalized_user_context) for pattern in patterns)


def ensure_python_code_block(response_text: str, user_context: str = "") -> Tuple[str, List[str]]:
    normalized = response_text or ""
    notes: List[str] = []

    if re.search(r"```(?:python|py)\b", normalized, re.IGNORECASE) and normalized.count("```") % 2 == 1:
        normalized = normalized.rstrip() + "\n```"
        notes.append("Closed an unfinished Python code fence.")

    if _extract_fenced_python_code_segments(normalized):
        return normalized, notes

    if is_code_only_request(user_context) and looks_like_python_script(normalized):
        normalized = f"```python\n{normalized.strip()}\n```"
        notes.append("Wrapped a plain Python response in a ```python``` code block.")

    return normalized, notes


def extract_python_code_segments(text: str) -> List[ExtractedCodeBlock]:
    normalized = text or ""
    if re.search(r"```(?:python|py)\b", normalized, re.IGNORECASE) and normalized.count("```") % 2 == 1:
        normalized = normalized.rstrip() + "\n```"

    segments = _extract_fenced_python_code_segments(normalized)
    if segments:
        return segments

    stripped = normalized.strip()
    extracted_plain_script = _extract_plain_python_script(stripped)
    if extracted_plain_script:
        start = normalized.find(extracted_plain_script)
        return [
            ExtractedCodeBlock(
                code=extracted_plain_script,
                start=max(start, 0),
                end=max(start, 0) + len(extracted_plain_script),
                fenced=False,
            )
        ]

    if looks_like_python_script(stripped):
        start = normalized.find(stripped)
        return [
            ExtractedCodeBlock(
                code=stripped,
                start=max(start, 0),
                end=max(start, 0) + len(stripped),
                fenced=False,
            )
        ]

    return []


def _extract_fenced_python_code_segments(text: str) -> List[ExtractedCodeBlock]:
    segments: List[ExtractedCodeBlock] = []

    for match in _FENCED_BLOCK_PATTERN.finditer(text):
        lang = (match.group("lang") or "").strip().lower()
        code = match.group("code").strip()
        if lang in {"python", "py"} or (not lang and looks_like_python_script(code)):
            segments.append(
                ExtractedCodeBlock(
                    code=code,
                    start=match.start(),
                    end=match.end(),
                    fenced=True,
                )
            )
    return segments


def _extract_plain_python_script(text: str) -> str:
    stripped = (text or "").strip()
    if not looks_like_python_script(stripped):
        return ""

    if _parses_as_python(stripped):
        return stripped

    lines = stripped.splitlines()
    best_end = 0
    for end in range(1, len(lines) + 1):
        candidate = "\n".join(lines[:end]).rstrip()
        if not candidate:
            continue
        if _parses_as_python(candidate):
            best_end = end

    if best_end <= 0 or best_end >= len(lines):
        return ""

    trailing_lines = lines[best_end:]
    first_trailing = next((line.strip() for line in trailing_lines if line.strip()), "")
    if not first_trailing or not _looks_like_non_code_trailer(first_trailing):
        return ""

    return "\n".join(lines[:best_end]).rstrip()


def _parses_as_python(candidate: str) -> bool:
    try:
        ast.parse(candidate)
        return True
    except SyntaxError:
        return False


def _looks_like_non_code_trailer(line: str) -> bool:
    lowered = (line or "").strip().lower()
    if not lowered:
        return False

    markers = (
        "**root cause",
        "**what changed",
        "**remaining assumption",
        "**local validation",
        "root cause:",
        "what changed:",
        "remaining assumption:",
        "local validation:",
        "the error was",
        "this fixes",
        "explanation:",
        "reason:",
        "saved plot",
        "output:",
        "feedback loop:",
    )
    return lowered.startswith("**") or lowered.startswith("```") or lowered.startswith(markers)


def extract_python_code_blocks(text: str) -> List[str]:
    return [segment.code for segment in extract_python_code_segments(text)]


def strip_python_code_from_message(text: str) -> str:
    normalized, _ = ensure_python_code_block(text or "")
    segments = extract_python_code_segments(normalized)
    if not segments:
        return normalized

    if len(segments) == 1 and not segments[0].fenced:
        return ""

    pieces: List[str] = []
    cursor = 0
    for segment in segments:
        pieces.append(normalized[cursor:segment.start])
        cursor = segment.end
    pieces.append(normalized[cursor:])
    stripped = "".join(pieces).strip()
    return re.sub(r"\n{3,}", "\n\n", stripped)


def replace_python_code_block(text: str, block_index: int, new_code: str) -> str:
    normalized, _ = ensure_python_code_block(text or "")
    segments = extract_python_code_segments(normalized)
    if block_index < 0 or block_index >= len(segments):
        return normalized

    replacement = f"```python\n{new_code}\n```"
    segment = segments[block_index]
    if not segment.fenced and len(segments) == 1:
        return replacement

    return normalized[:segment.start] + replacement + normalized[segment.end:]
