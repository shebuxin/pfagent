import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .code_blocks import ensure_python_code_block

logger = logging.getLogger(__name__)

DEFAULT_CODEX_FIX_MODEL = os.environ.get("OPENAI_CODEX_FIX_MODEL", "gpt-5.2-codex")
DEFAULT_REPO_CONTEXT_K = int(os.environ.get("OPENAI_CODEX_FIX_REPO_CONTEXT_K", "4"))
DEFAULT_HISTORY_TURNS = int(os.environ.get("OPENAI_CODEX_FIX_HISTORY_TURNS", "3"))
MAX_REPO_FILE_BYTES = int(os.environ.get("OPENAI_CODEX_FIX_MAX_FILE_BYTES", "200000"))
DEFAULT_CODEX_FIX_MAX_TOKENS = int(os.environ.get("OPENAI_CODEX_FIX_MAX_TOKENS", "3200"))
DEFAULT_CODEX_FIX_REASONING_EFFORT = os.environ.get("OPENAI_CODEX_FIX_REASONING_EFFORT", "medium")
DEFAULT_CODEX_FIX_TEXT_VERBOSITY = os.environ.get("OPENAI_CODEX_FIX_TEXT_VERBOSITY", "medium")

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]{2,}")
_SPLIT_RE = re.compile(r"[._/:-]+")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "while",
    "when",
    "then",
    "else",
    "true",
    "false",
    "none",
    "please",
    "provide",
    "corrected",
    "python",
    "code",
    "error",
    "output",
    "request",
    "running",
    "encountered",
    "failed",
    "fix",
    "what",
    "how",
    "your",
    "they",
    "them",
    "there",
    "have",
    "does",
    "did",
    "using",
    "used",
    "use",
    "call",
    "calls",
    "value",
    "line",
    "lines",
}
_REPO_CACHE: Dict[Tuple[str, int, int], "RepoContextRetriever"] = {}


def build_basic_error_fix_prompt(code: str, output: str) -> str:
    return f"""I encountered an error while running Python code. Please analyze the error and provide a corrected version of the code.

**Original Code:**
```python
{code}
```

**Error Output:**
```text
{output}
```

**Request:**
Please provide the corrected Python code that fixes this error. Focus on:
1. Identifying the root cause of the error
2. Providing syntactically correct code
3. Adding any necessary imports or dependencies
4. Including proper error handling if applicable
5. Adding comments explaining the fix if it's not obvious

Please respond with the corrected code in a ```python``` code block, and briefly explain what was wrong and how you fixed it."""


@dataclass(frozen=True)
class RepoSnippet:
    path: str
    start_line: int
    end_line: int
    content: str
    score: float


@dataclass(frozen=True)
class _IndexedRepoChunk:
    path: str
    start_line: int
    end_line: int
    content: str
    path_lower: str
    content_lower: str


@dataclass
class CodexFixerConfig:
    openai_api_key: str
    repo_root: str
    model: str = DEFAULT_CODEX_FIX_MODEL
    max_tokens: int = DEFAULT_CODEX_FIX_MAX_TOKENS
    repo_context_k: int = DEFAULT_REPO_CONTEXT_K
    history_turns: int = DEFAULT_HISTORY_TURNS
    chunk_lines: int = 50
    overlap_lines: int = 10
    reasoning_effort: str = DEFAULT_CODEX_FIX_REASONING_EFFORT
    text_verbosity: str = DEFAULT_CODEX_FIX_TEXT_VERBOSITY


def clear_repo_context_cache() -> None:
    _REPO_CACHE.clear()


def _truncate_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _build_fix_normalization_context(request: Mapping[str, Any]) -> str:
    sections: List[str] = []
    user_message = str(request.get("user_message", "") or "").strip()
    if user_message:
        sections.append(user_message)

    assistant_message = str(request.get("assistant_message", "") or "").strip()
    if assistant_message:
        sections.append(assistant_message)

    failed_code = str(request.get("failed_code", "") or "").strip()
    if failed_code:
        sections.append(f"Previously failing code:\n```python\n{failed_code}\n```")

    error_output = str(request.get("error_output", "") or "").strip()
    if error_output:
        sections.append(f"Runtime error:\n```text\n{error_output}\n```")

    return "\n\n".join(section for section in sections if section)


def normalize_error_fix_response(response_text: str, request: Mapping[str, Any]) -> Tuple[str, List[str]]:
    normalized, notes = ensure_python_code_block(
        response_text,
        user_context=str(request.get("user_message", "") or ""),
    )

    guardrail_context = _build_fix_normalization_context(request)
    if not guardrail_context:
        return normalized, notes

    try:
        from .chatbots.openai.rag_chatbot import normalize_andes_response

        normalized, andes_notes = normalize_andes_response(
            normalized,
            user_context=guardrail_context,
        )
        notes.extend(andes_notes)
    except Exception as exc:
        logger.warning("Failed to apply shared ANDES guardrails to fixer response: %s", exc)

    return normalized, notes


def _extract_response_text(response: Any) -> str:
    """Normalize LangChain/OpenAI Responses API payloads into plain text."""
    if isinstance(response, str):
        return response

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output_items = getattr(response, "output", None)
    if isinstance(output_items, list):
        parts: List[str] = []
        for item in output_items:
            item_type = getattr(item, "type", None)
            if isinstance(item, dict):
                item_type = item.get("type", item_type)
                content_blocks = item.get("content", [])
            else:
                content_blocks = getattr(item, "content", [])

            if item_type == "message" and isinstance(content_blocks, list):
                for block in content_blocks:
                    if isinstance(block, dict):
                        text_value = block.get("text")
                        if isinstance(text_value, str) and text_value.strip():
                            parts.append(text_value)
                    else:
                        text_value = getattr(block, "text", None)
                        if isinstance(text_value, str) and text_value.strip():
                            parts.append(text_value)
                continue

            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str) and text_attr.strip():
                parts.append(text_attr)
        if parts:
            return "\n".join(parts).strip()

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                    continue
                if isinstance(item.get("content"), str):
                    parts.append(item["content"])
                    continue
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str):
                parts.append(text_attr)
                continue
            content_attr = getattr(item, "content", None)
            if isinstance(content_attr, str):
                parts.append(content_attr)
                continue
        return "\n".join(part for part in parts if part).strip()

    text_attr = getattr(response, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    return ""


def _summarize_response_for_debug(response: Any, max_preview_chars: int = 500) -> str:
    content = getattr(response, "content", response)
    summary_parts = [f"response_type={type(response).__name__}", f"content_type={type(content).__name__}"]
    status = getattr(response, "status", None)
    if status is not None:
        summary_parts.append(f"status={status}")
    incomplete_details = getattr(response, "incomplete_details", None)
    incomplete_reason = getattr(incomplete_details, "reason", None)
    if incomplete_reason:
        summary_parts.append(f"incomplete_reason={incomplete_reason}")

    if isinstance(content, list):
        item_summaries: List[str] = []
        preview_parts: List[str] = []
        for index, item in enumerate(content[:6]):
            item_type = type(item).__name__
            if isinstance(item, dict):
                dict_keys = ",".join(sorted(str(key) for key in item.keys())[:8])
                item_summaries.append(f"{index}:{item_type}[{dict_keys}]")
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    preview_parts.append(text_value.strip())
                elif isinstance(item.get("content"), str) and item["content"].strip():
                    preview_parts.append(item["content"].strip())
            else:
                item_summaries.append(f"{index}:{item_type}")
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str) and text_attr.strip():
                    preview_parts.append(text_attr.strip())
                elif isinstance(item, str) and item.strip():
                    preview_parts.append(item.strip())
        if item_summaries:
            summary_parts.append(f"items={' | '.join(item_summaries)}")
        preview_text = "\n".join(part for part in preview_parts if part).strip()
        if preview_text:
            summary_parts.append(f"preview={_truncate_text(preview_text, max_preview_chars)!r}")
        return "; ".join(summary_parts)

    if isinstance(content, str):
        summary_parts.append(f"preview={_truncate_text(content, max_preview_chars)!r}")
        return "; ".join(summary_parts)

    text_attr = getattr(response, "text", None)
    if isinstance(text_attr, str):
        summary_parts.append(f"text_attr={_truncate_text(text_attr, max_preview_chars)!r}")

    output_items = getattr(response, "output", None)
    if isinstance(output_items, list):
        output_summaries: List[str] = []
        for index, item in enumerate(output_items[:6]):
            item_type = getattr(item, "type", None)
            if isinstance(item, dict):
                item_type = item.get("type", item_type)
            output_summaries.append(f"{index}:{type(item).__name__}:{item_type}")
        if output_summaries:
            summary_parts.append(f"output_items={' | '.join(output_summaries)}")

    return "; ".join(summary_parts)


def _response_hit_output_cap(response: Any) -> bool:
    incomplete_details = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete_details, "reason", None)
    return reason == "max_output_tokens"


def _extract_signal_terms(text: str) -> List[str]:
    terms: List[str] = []
    seen = set()
    for raw_match in _TOKEN_RE.findall(text or ""):
        for piece in [raw_match, *_SPLIT_RE.split(raw_match)]:
            token = piece.strip().lower()
            if len(token) < 3 or token in _STOPWORDS:
                continue
            if token not in seen:
                seen.add(token)
                terms.append(token)
    return terms


def _repo_file_paths(repo_root: Path) -> List[Path]:
    include_files = [
        repo_root / "README.md",
        repo_root / "CONTRIBUTING.md",
        repo_root / "Makefile",
        repo_root / "text-to-sim" / "README.md",
        repo_root / "text-to-sim" / "main.py",
        repo_root / "verification" / "README.md",
        repo_root / "verification" / "REVIEWER_GUIDE.md",
    ]
    include_dirs = [
        repo_root / "docs",
        repo_root / "text-to-sim" / "src",
        repo_root / "text-to-sim" / "tests",
        repo_root / "verification",
        repo_root / "knowledge" / "rag" / "code_examples",
    ]
    allowed_suffixes = {".py", ".md", ".txt", ".toml", ".yml", ".yaml"}
    excluded_parts = {
        ".git",
        "__pycache__",
        "code_executions",
        "final",
        "optimization",
        "generated_examples",
        "old_generated_examples",
        "data_files",
    }

    results = set()
    for file_path in include_files:
        if file_path.exists() and file_path.is_file():
            results.add(file_path.resolve())

    for directory in include_dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if any(part in excluded_parts for part in path.relative_to(repo_root).parts):
                continue
            if path.suffix.lower() not in allowed_suffixes:
                continue
            try:
                if path.stat().st_size > MAX_REPO_FILE_BYTES:
                    continue
            except OSError:
                continue
            results.add(path.resolve())

    return sorted(results)


def _build_repo_chunks(repo_root: str, chunk_lines: int, overlap_lines: int) -> List[_IndexedRepoChunk]:
    root = Path(repo_root).resolve()
    chunks: List[_IndexedRepoChunk] = []
    step = max(1, chunk_lines - overlap_lines)

    for path in _repo_file_paths(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        lines = text.splitlines()
        if not lines:
            continue

        rel_path = str(path.relative_to(root))
        for start in range(0, len(lines), step):
            end = min(len(lines), start + chunk_lines)
            snippet = "\n".join(lines[start:end]).strip()
            if not snippet:
                continue
            chunks.append(
                _IndexedRepoChunk(
                    path=rel_path,
                    start_line=start + 1,
                    end_line=end,
                    content=snippet,
                    path_lower=rel_path.lower(),
                    content_lower=snippet.lower(),
                )
            )
            if end >= len(lines):
                break

    return chunks


class RepoContextRetriever:
    def __init__(self, repo_root: str, chunk_lines: int = 50, overlap_lines: int = 10):
        self.repo_root = str(Path(repo_root).resolve())
        self.chunk_lines = chunk_lines
        self.overlap_lines = overlap_lines
        self._chunks = _build_repo_chunks(self.repo_root, chunk_lines, overlap_lines)

    def retrieve(self, query: str, k: int = DEFAULT_REPO_CONTEXT_K) -> List[RepoSnippet]:
        query_terms = _extract_signal_terms(query)
        if not query_terms:
            return []

        scored: List[RepoSnippet] = []
        for chunk in self._chunks:
            score = 0.0
            for term in query_terms:
                path_hits = chunk.path_lower.count(term)
                content_hits = chunk.content_lower.count(term)
                if path_hits:
                    score += 10.0 + min(path_hits - 1, 2) * 3.0
                if content_hits:
                    score += min(content_hits, 6) * (4.0 if len(term) >= 8 else 2.0)
            if score <= 0:
                continue
            scored.append(
                RepoSnippet(
                    path=chunk.path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                    score=score,
                )
            )

        scored.sort(key=lambda item: (-item.score, item.path, item.start_line))
        return scored[:k]


def _get_repo_retriever(repo_root: str, chunk_lines: int, overlap_lines: int) -> RepoContextRetriever:
    key = (str(Path(repo_root).resolve()), chunk_lines, overlap_lines)
    retriever = _REPO_CACHE.get(key)
    if retriever is None:
        retriever = RepoContextRetriever(key[0], chunk_lines=chunk_lines, overlap_lines=overlap_lines)
        _REPO_CACHE[key] = retriever
    return retriever


def _format_recent_history(
    chat_history: Sequence[Tuple[str, str]],
    message_index: Optional[int],
    max_turns: int,
) -> str:
    if not chat_history:
        return ""

    end = len(chat_history)
    if isinstance(message_index, int):
        end = min(end, message_index + 1)
    start = max(0, end - max_turns)

    sections: List[str] = []
    for turn_number, (user_msg, assistant_msg) in enumerate(chat_history[start:end], start=start + 1):
        sections.append(
            "\n".join(
                [
                    f"Turn {turn_number} user:",
                    _truncate_text(user_msg, 900),
                    "",
                    f"Turn {turn_number} assistant:",
                    _truncate_text(assistant_msg, 1200),
                ]
            )
        )
    return "\n\n".join(sections)


def _format_runtime_file_context(runtime_data_dir: str, runtime_files: Sequence[str]) -> str:
    if not runtime_files:
        return ""
    listed_files = "\n".join(f"- {name}" for name in runtime_files[:12])
    return "\n".join(
        [
            f"- Runtime data directory: {runtime_data_dir}",
            "- Files currently available to the execution environment:",
            listed_files,
        ]
    )


def _format_active_case(active_case: Mapping[str, Any]) -> str:
    source = str(active_case.get("source", "")).strip()
    value = str(active_case.get("value", "")).strip()
    if not source or not value:
        return ""
    return "\n".join(
        [
            f"- Last successful case source: {source}",
            f"- Last successful case identifier: {value}",
            "- Preserve continuity unless the user explicitly changed cases.",
        ]
    )


def _format_repo_snippets(snippets: Sequence[RepoSnippet]) -> str:
    if not snippets:
        return ""

    sections = ["## Retrieved repository context"]
    for index, snippet in enumerate(snippets, 1):
        sections.append(
            "\n".join(
                [
                    f"### Repo snippet {index}: {snippet.path}:{snippet.start_line}-{snippet.end_line}",
                    "```text",
                    _truncate_text(snippet.content, 2200),
                    "```",
                ]
            )
        )
    return "\n\n".join(sections)


def _format_validation_attempt(request: Mapping[str, Any]) -> str:
    attempt_number = request.get("validation_attempt")
    validation_output = str(request.get("validation_output", "") or "").strip()
    previous_candidate_code = str(request.get("previous_candidate_code", "") or "").strip()

    if attempt_number is None and not validation_output and not previous_candidate_code:
        return ""

    sections = ["## Local validation feedback"]
    if attempt_number is not None:
        sections.append(f"- This is retry attempt {attempt_number} after a local execution check failed.")
    sections.append(
        "- The app will execute your returned code in the current local session environment. "
        "Use the validation failure below to produce a revised, runnable fix."
    )

    if previous_candidate_code:
        sections.extend(
            [
                "### Previously generated candidate",
                "```python",
                previous_candidate_code,
                "```",
            ]
        )

    if validation_output:
        sections.extend(
            [
                "### Validation failure output",
                "```text",
                validation_output,
                "```",
            ]
        )

    return "\n".join(sections)


def build_codex_fix_system_prompt(custom_instructions: str = "") -> str:
    prompt = """
You are Codex, a repository-aware debugging assistant for the PFAGENT Streamlit app.

Your job is to fix failing Python code with minimal, runnable changes.
Prefer repository-grounded patterns over invented APIs, especially for ANDES usage.
Preserve the user's intent, outputs, and artifacts unless they directly caused the failure.
If the repository context contradicts the failing code, follow the repository pattern.
Assume your code will be executed locally after you answer. If prior local validation failed, incorporate that runtime evidence directly instead of making a superficial edit.

Always return:
1. One corrected ```python``` code block.
2. A short explanation with:
   - Root cause
   - What changed
   - Any assumption that remains
""".strip()

    custom = (custom_instructions or "").strip()
    if custom:
        prompt += f"\n\nAdditional user instructions:\n{custom}"
    return prompt


def build_codex_fix_user_message(
    request: Mapping[str, Any],
    repo_snippets: Sequence[RepoSnippet],
    history_turns: int = DEFAULT_HISTORY_TURNS,
    fallback_reason: str = "",
) -> str:
    sections = [
        "A generated Python script failed inside the PFAGENT Streamlit app. "
        "Fix it using the runtime context and repository snippets below when they are relevant.",
    ]

    fallback_reason = str(fallback_reason or "").strip()
    if fallback_reason:
        sections.append(
            "\n".join(
                [
                    "## Fallback repair mode",
                    "- The dedicated Codex fixer was unavailable for this attempt.",
                    "- The current chat agent must act as a focused debugging assistant for this one repair only.",
                    "- Ignore earlier unrelated turns unless they are explicitly restated below.",
                    f"- Fallback reason: {fallback_reason}",
                ]
            )
        )

    user_message = str(request.get("user_message", "") or "").strip()
    if user_message:
        sections.append(f"## Original user request\n{_truncate_text(user_message, 1800)}")

    chat_history = request.get("recent_chat_history") or []
    recent_history = _format_recent_history(
        chat_history,
        request.get("message_index"),
        max_turns=history_turns,
    )
    if recent_history:
        sections.append(f"## Recent conversation context\n{recent_history}")

    runtime_context = _format_runtime_file_context(
        str(request.get("runtime_data_dir", "") or ""),
        request.get("runtime_files") or [],
    )
    if runtime_context:
        sections.append(f"## Runtime file context\n{runtime_context}")

    uploaded_case_preview = str(request.get("uploaded_case_preview", "") or "").strip()
    if uploaded_case_preview:
        sections.append(f"## Uploaded case preview\n{_truncate_text(uploaded_case_preview, 2500)}")

    active_case = request.get("active_case")
    if isinstance(active_case, Mapping):
        active_case_context = _format_active_case(active_case)
        if active_case_context:
            sections.append(f"## ANDES continuity context\n{active_case_context}")

    repo_context = _format_repo_snippets(repo_snippets)
    if repo_context:
        sections.append(repo_context)

    validation_context = _format_validation_attempt(request)
    if validation_context:
        sections.append(validation_context)

    failed_code = str(request.get("failed_code", "") or "").rstrip()
    sections.append(
        "\n".join(
            [
                "## Failing code",
                "```python",
                failed_code,
                "```",
            ]
        )
    )

    error_output = str(request.get("error_output", "") or "").rstrip()
    sections.append(
        "\n".join(
            [
                "## Execution output",
                "```text",
                error_output,
                "```",
            ]
        )
    )

    sections.append(
        "\n".join(
            [
                "## Fix requirements",
                "- Keep the solution aligned with existing repository conventions when possible.",
                "- Do not invent unsupported ANDES helper methods or attributes.",
                "- Preserve plots, files, and outputs that the user asked for unless they caused the crash.",
                "- If the error came from the app's generated code, return a clean replacement that the user can run directly.",
                "- Prefer a fix that will survive immediate local execution, not just a plausible textual rewrite.",
                "- Treat this as a fresh debugging task: do not answer an earlier user request unless it is restated in the Original user request section.",
            ]
        )
    )

    return "\n\n".join(section for section in sections if section.strip())


def build_repo_aware_fix_prompt(
    request: Mapping[str, Any],
    repo_root: str,
    *,
    repo_context_k: int = DEFAULT_REPO_CONTEXT_K,
    history_turns: int = DEFAULT_HISTORY_TURNS,
    chunk_lines: int = 50,
    overlap_lines: int = 10,
    fallback_reason: str = "",
) -> str:
    retriever = _get_repo_retriever(
        repo_root=str(Path(repo_root).resolve()),
        chunk_lines=chunk_lines,
        overlap_lines=overlap_lines,
    )
    retrieval_query = "\n".join(
        filter(
            None,
            [
                str(request.get("user_message", "") or ""),
                str(request.get("failed_code", "") or ""),
                str(request.get("error_output", "") or ""),
            ],
        )
    )
    repo_snippets = retriever.retrieve(retrieval_query, k=repo_context_k)
    return build_codex_fix_user_message(
        request,
        repo_snippets,
        history_turns=history_turns,
        fallback_reason=fallback_reason,
    )


async def run_isolated_chat_repair(chatbot: Any, prompt: str) -> str:
    conversation_history = getattr(chatbot, "conversation_history", None)
    if not isinstance(conversation_history, list):
        return await chatbot.chat(prompt)

    original_history = list(conversation_history)
    try:
        chatbot.conversation_history = []
        return await chatbot.chat(prompt)
    finally:
        chatbot.conversation_history = original_history


class RepoAwareCodexFixer:
    def __init__(self, config: CodexFixerConfig):
        from openai import AsyncOpenAI

        self.config = config
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.repo_retriever = _get_repo_retriever(
            repo_root=config.repo_root,
            chunk_lines=config.chunk_lines,
            overlap_lines=config.overlap_lines,
        )

    async def _request_fix(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_output_tokens: int,
        reasoning_effort: str,
    ) -> Any:
        request_kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": reasoning_effort},
        }
        if self.config.text_verbosity:
            request_kwargs["text"] = {"verbosity": self.config.text_verbosity}

        try:
            return await self.client.responses.create(**request_kwargs)
        except Exception as exc:
            error_text = str(exc)
            retry_kwargs = dict(request_kwargs)
            should_retry = False

            if "text.verbosity" in error_text and retry_kwargs.get("text", {}).get("verbosity") != "medium":
                retry_kwargs["text"] = {"verbosity": "medium"}
                should_retry = True

            if "reasoning" in error_text and retry_kwargs.get("reasoning", {}).get("effort") != "medium":
                retry_kwargs["reasoning"] = {"effort": "medium"}
                should_retry = True

            if should_retry:
                logger.warning(
                    "Retrying Codex fixer request with compatibility-safe Responses API settings after error: %s",
                    error_text,
                )
                return await self.client.responses.create(**retry_kwargs)
            raise

    async def fix_error(self, request: Mapping[str, Any]) -> Tuple[str, str]:
        retrieval_query = "\n".join(
            filter(
                None,
                [
                    str(request.get("user_message", "") or ""),
                    str(request.get("failed_code", "") or ""),
                    str(request.get("error_output", "") or ""),
                ],
            )
        )
        repo_snippets = self.repo_retriever.retrieve(
            retrieval_query,
            k=self.config.repo_context_k,
        )
        user_prompt = build_codex_fix_user_message(
            request,
            repo_snippets,
            history_turns=self.config.history_turns,
        )
        system_prompt = build_codex_fix_system_prompt(
            str(request.get("custom_instructions", "") or "")
        )

        response = await self._request_fix(
            system_prompt,
            user_prompt,
            max_output_tokens=self.config.max_tokens,
            reasoning_effort=self.config.reasoning_effort,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Codex fixer raw response summary: %s", _summarize_response_for_debug(response))

        content = _extract_response_text(response)
        if not content and _response_hit_output_cap(response):
            retry_max_tokens = max(self.config.max_tokens * 2, 4800)
            logger.warning(
                "Codex fixer response hit max_output_tokens without returning text; retrying with max_output_tokens=%s",
                retry_max_tokens,
            )
            response = await self._request_fix(
                system_prompt,
                user_prompt,
                max_output_tokens=retry_max_tokens,
                reasoning_effort="low",
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Codex fixer retry response summary: %s", _summarize_response_for_debug(response))
            content = _extract_response_text(response)

        if not content:
            raise RuntimeError(
                "Codex fixer returned no assistant text. "
                f"status={getattr(response, 'status', None)!r}, "
                f"incomplete_reason={getattr(getattr(response, 'incomplete_details', None), 'reason', None)!r}"
            )

        normalized_content, normalization_notes = normalize_error_fix_response(
            content,
            request,
        )
        if normalization_notes:
            logger.info("Applied Codex fixer response normalization: %s", "; ".join(normalization_notes))
        return normalized_content, user_prompt


def create_codex_error_fixer(
    openai_api_key: str,
    repo_root: str,
    model: str = DEFAULT_CODEX_FIX_MODEL,
) -> RepoAwareCodexFixer:
    config = CodexFixerConfig(
        openai_api_key=openai_api_key,
        repo_root=repo_root,
        model=model or DEFAULT_CODEX_FIX_MODEL,
    )
    return RepoAwareCodexFixer(config)
