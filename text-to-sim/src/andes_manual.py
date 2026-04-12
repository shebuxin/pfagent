import os
import re
from functools import lru_cache
from typing import Dict, List, Tuple

import PyPDF2


DEFAULT_ANDES_MANUAL_DOC_ID_PREFIX = "andes_manual_builtin"
DEFAULT_MANUAL_RETRIEVAL_WINDOW_PAGES = int(os.environ.get("ANDES_MANUAL_WINDOW_PAGES", "8"))
DEFAULT_MANUAL_RETRIEVAL_MAX_WINDOWS = int(os.environ.get("ANDES_MANUAL_MAX_WINDOWS", "2"))
DEFAULT_MANUAL_RETRIEVAL_MAX_CHARS = int(os.environ.get("ANDES_MANUAL_MAX_CHARS", "9000"))

_MANUAL_QUERY_STOPWORDS = {
    "about",
    "after",
    "against",
    "align",
    "along",
    "also",
    "analysis",
    "answer",
    "anything",
    "around",
    "available",
    "before",
    "between",
    "built",
    "builtin",
    "case",
    "check",
    "code",
    "current",
    "data",
    "default",
    "directly",
    "document",
    "execution",
    "explain",
    "file",
    "files",
    "follow",
    "following",
    "from",
    "generate",
    "generated",
    "help",
    "into",
    "load",
    "loaded",
    "need",
    "only",
    "output",
    "plot",
    "please",
    "print",
    "python",
    "question",
    "response",
    "result",
    "results",
    "retrieval",
    "runtime",
    "same",
    "section",
    "sections",
    "show",
    "that",
    "their",
    "them",
    "there",
    "these",
    "this",
    "those",
    "through",
    "uploaded",
    "using",
    "user",
    "when",
    "with",
    "without",
    "would",
    "your",
}


STRICT_ANDES_CODE_POLICY = """
ANDES Code Generation Policy:
- Treat the ANDES manual as the authoritative source of truth for APIs, argument names, workflow order, and result access.
- Build one coherent solution that follows a single manual-supported workflow. Do not stitch together incompatible snippets from unrelated parts of the manual.
- Use only ANDES APIs, attributes, routine names, and arguments that are supported by the manual context, the built-in case catalog, or the few-shot examples.
- If the retrieved manual context is insufficient to support a requested ANDES API call, say what is missing instead of guessing.
- Prefer the smallest runnable script that directly answers the request.
- Keep the workflow aligned with the manual: imports -> case loading -> optional model edits -> setup -> routine execution -> result access -> plotting/printing.
- For ANDES built-in benchmark cases, use exact `andes.get_case("...")` paths.
- For user-uploaded cases, use the exact uploaded filename directly in `andes.load(...)` and never wrap uploaded files with `andes.get_case(...)`.
""".strip()


RAG_ANDES_MANUAL_POLICY = f"""
ANDES Manual Policy:
- The official ANDES manual is preloaded by default and retrieval runs over the full manual, not just a small hand-picked subset.
- Prefer solutions that are directly supported by the retrieved manual windows and the provided few-shot examples.
- When multiple manual-supported approaches seem possible, choose the simplest one that is most faithful to the manual.
- If the manual context does not support a requested step, say so clearly instead of inventing behavior.

{STRICT_ANDES_CODE_POLICY}
""".strip()


BASE_ANDES_MANUAL_POLICY = f"""
ANDES Manual Policy:
- For ANDES-related requests, prefer workflows and API usage that are consistent with the official ANDES manual.
- If you cannot verify that an ANDES API, attribute, or argument exists, say so instead of inventing it.
- Keep imports, case loading, setup, routine execution, and result access aligned with standard ANDES usage.

{STRICT_ANDES_CODE_POLICY}
""".strip()


def _normalize_manual_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _strip_runtime_context(query: str) -> str:
    cleaned_query = query or ""
    for marker in ("Runtime file context:", "ANDES continuity context:"):
        if marker in cleaned_query:
            cleaned_query = cleaned_query.split(marker, 1)[0]
    return cleaned_query.strip()


def _extract_manual_query_terms(query: str) -> List[str]:
    cleaned_query = _strip_runtime_context(query)
    if not cleaned_query:
        return []

    terms = set()
    for api_name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", cleaned_query):
        terms.add(api_name.lower())

    for file_name in re.findall(r"[A-Za-z0-9_\-/]+\.(?:xlsx|xls|raw|json|csv|dyr|dat|txt|seq|rcd|m|mat)", cleaned_query):
        terms.add(file_name.lower())
        parts = re.split(r"[/\\._-]+", file_name.lower())
        for part in parts:
            if len(part) >= 3 and part not in _MANUAL_QUERY_STOPWORDS:
                terms.add(part)

    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", cleaned_query.lower()):
        if token not in _MANUAL_QUERY_STOPWORDS:
            terms.add(token)

    if "power flow" in cleaned_query.lower():
        terms.update({"power flow", "pflow"})

    return sorted(terms, key=lambda item: (-len(item), item))


def _score_manual_page(page_text: str, query: str, query_terms: List[str]) -> int:
    normalized_text = page_text.lower()
    normalized_query = " ".join(_strip_runtime_context(query).lower().split())
    score = 0

    if normalized_query and normalized_query in normalized_text:
        score += 30

    for term in query_terms:
        if not term:
            continue
        hit_count = normalized_text.count(term)
        if hit_count == 0:
            continue

        if "." in term or "/" in term:
            weight = 10
        elif any(char.isdigit() for char in term):
            weight = 7
        elif len(term) >= 8:
            weight = 5
        else:
            weight = 3

        score += weight * min(hit_count, 4)

    return score


def get_andes_manual_pdf_path() -> str:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    text_to_sim_root = os.path.dirname(module_dir)
    repo_root = os.path.dirname(text_to_sim_root)

    candidate_paths = [
        # New canonical location under knowledge/ (2026-04-10 refactor).
        os.path.join(repo_root, "knowledge", "rag", "andes_manual.pdf"),
        # Legacy raw extraction copy (kept as a fallback).
        os.path.join(
            repo_root,
            "knowledge",
            "raw",
            "manual_extraction",
            "docs-andes-app-en-stable.pdf",
        ),
    ]

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Unable to find andes_manual.pdf under knowledge/rag/ "
        "or knowledge/raw/manual_extraction/."
    )


@lru_cache(maxsize=1)
def load_andes_manual_pages() -> Tuple[Tuple[int, str], ...]:
    manual_path = get_andes_manual_pdf_path()
    reader = PyPDF2.PdfReader(manual_path)

    pages: List[Tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text() or ""
        normalized_text = _normalize_manual_text(extracted_text)
        if normalized_text:
            pages.append((page_number, normalized_text))

    if not pages:
        raise ValueError("The ANDES manual PDF was found but no text could be extracted.")

    return tuple(pages)


def build_default_andes_manual_documents(
    window_pages: int = 12,
    overlap_pages: int = 3,
) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
    pages = list(load_andes_manual_pages())
    safe_window_pages = max(window_pages, 1)
    safe_overlap_pages = max(min(overlap_pages, safe_window_pages - 1), 0)
    stride = max(safe_window_pages - safe_overlap_pages, 1)

    documents: List[str] = []
    doc_ids: List[str] = []
    metadata: List[Dict[str, str]] = []

    for start_index in range(0, len(pages), stride):
        page_window = pages[start_index:start_index + safe_window_pages]
        if not page_window:
            continue

        start_page = page_window[0][0]
        end_page = page_window[-1][0]
        window_blocks = [
            f"[ANDES Manual Page {page_number}]\n{page_text}"
            for page_number, page_text in page_window
        ]
        document_text = (
            f"Official ANDES Manual pages {start_page}-{end_page}\n\n"
            + "\n\n".join(window_blocks)
        )

        documents.append(document_text)
        doc_ids.append(f"{DEFAULT_ANDES_MANUAL_DOC_ID_PREFIX}_{start_page}_{end_page}")
        metadata.append(
            {
                "source": "andes_manual",
                "title": f"Official ANDES Manual pages {start_page}-{end_page}",
                "page_range": f"{start_page}-{end_page}",
            }
        )

    return documents, doc_ids, metadata


def retrieve_relevant_andes_manual_windows(
    query: str,
    window_pages: int = DEFAULT_MANUAL_RETRIEVAL_WINDOW_PAGES,
    max_windows: int = DEFAULT_MANUAL_RETRIEVAL_MAX_WINDOWS,
    max_chars_per_window: int = DEFAULT_MANUAL_RETRIEVAL_MAX_CHARS,
) -> List[Dict[str, str]]:
    pages = list(load_andes_manual_pages())
    cleaned_query = _strip_runtime_context(query)
    if not cleaned_query:
        return []

    query_terms = _extract_manual_query_terms(cleaned_query)
    scored_pages: List[Tuple[int, int]] = []
    for index, (_, page_text) in enumerate(pages):
        score = _score_manual_page(page_text, cleaned_query, query_terms)
        if score > 0:
            scored_pages.append((score, index))

    if not scored_pages:
        fallback_terms = ["andes.load", "andes.get_case", "pflow", "power flow", "setup"]
        for index, (_, page_text) in enumerate(pages):
            score = _score_manual_page(page_text, cleaned_query, fallback_terms)
            if score > 0:
                scored_pages.append((score, index))

    if not scored_pages:
        return []

    chosen_ranges: List[Tuple[int, int]] = []
    backward_window = max(window_pages // 3, 0)

    for _, page_index in sorted(scored_pages, key=lambda item: (-item[0], item[1])):
        start_index = max(0, page_index - backward_window)
        end_index = min(len(pages), start_index + max(window_pages, 1))
        start_index = max(0, end_index - max(window_pages, 1))

        overlaps_existing = any(
            not (end_index <= existing_start or start_index >= existing_end)
            for existing_start, existing_end in chosen_ranges
        )
        if overlaps_existing:
            continue

        chosen_ranges.append((start_index, end_index))
        if len(chosen_ranges) >= max(max_windows, 1):
            break

    windows: List[Dict[str, str]] = []
    for start_index, end_index in sorted(chosen_ranges):
        page_window = pages[start_index:end_index]
        start_page = page_window[0][0]
        end_page = page_window[-1][0]
        content = "\n\n".join(
            f"[ANDES Manual Page {page_number}]\n{page_text}"
            for page_number, page_text in page_window
        )
        if len(content) > max_chars_per_window:
            content = (
                content[:max_chars_per_window].rsplit(" ", 1)[0]
                + "\n\n[Manual excerpt truncated to keep the prompt coherent.]"
            )

        windows.append(
            {
                "title": f"Official ANDES Manual pages {start_page}-{end_page}",
                "page_range": f"{start_page}-{end_page}",
                "content": content,
            }
        )

    return windows


def build_retrieved_andes_manual_context(query: str) -> str:
    windows = retrieve_relevant_andes_manual_windows(query)
    if not windows:
        return ""

    sections = [
        "## Relevant Context from the full ANDES manual (retrieved over the complete manual):"
    ]
    for index, window in enumerate(windows, start=1):
        sections.append(
            f"### ANDES Manual Window {index} (pages {window['page_range']})\n{window['content']}"
        )
    return "\n\n".join(sections)


async def bootstrap_default_andes_manual(chatbot) -> int:
    if getattr(chatbot, "default_andes_manual_loaded", False):
        return getattr(chatbot, "default_andes_manual_count", 0)

    page_count = len(load_andes_manual_pages())
    chatbot.default_andes_manual_loaded = True
    chatbot.default_andes_manual_count = page_count
    chatbot.default_andes_manual_page_count = page_count
    return page_count
