import difflib
import os
from functools import lru_cache
from typing import List, Tuple


_SUPPORTED_CASE_EXTENSIONS = (
    ".json",
    ".xlsx",
    ".xls",
    ".raw",
    ".dyr",
    ".m",
    ".mat",
    ".csv",
    ".dat",
    ".txt",
    ".seq",
    ".rcd",
)


@lru_cache(maxsize=1)
def get_andes_builtin_case_paths() -> Tuple[str, ...]:
    """
    Return normalized relative paths under andes/cases that are likely loadable by andes.get_case().
    """
    try:
        import andes.cases as andes_cases  # type: ignore
    except Exception:
        return tuple()

    try:
        cases_root = andes_cases.__path__[0]
    except Exception:
        return tuple()

    case_paths = set()
    for root, dirs, files in os.walk(cases_root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for file_name in files:
            lower_name = file_name.lower()
            if not lower_name.endswith(_SUPPORTED_CASE_EXTENSIONS):
                continue
            abs_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(abs_path, cases_root).replace(os.sep, "/")
            case_paths.add(rel_path)

    return tuple(sorted(case_paths))


def suggest_andes_case_paths(path: str, max_suggestions: int = 3) -> List[str]:
    """Suggest close built-in case paths."""
    normalized = (path or "").replace("\\", "/")
    case_paths = list(get_andes_builtin_case_paths())
    if not case_paths:
        return []
    return difflib.get_close_matches(normalized, case_paths, n=max_suggestions, cutoff=0.5)


def build_andes_builtin_cases_section(limit: int = 40) -> str:
    """Build a compact section listing valid built-in case paths for prompt injection."""
    case_paths = list(get_andes_builtin_case_paths())
    if not case_paths or limit <= 0:
        return ""

    selected = case_paths[:limit]
    lines = [
        "Known ANDES Built-in Case Paths (for andes.get_case):",
        "- Use exact relative paths under andes/cases.",
        "- Examples:",
    ]
    lines.extend(f"  - {path}" for path in selected)
    remaining = len(case_paths) - len(selected)
    if remaining > 0:
        lines.append(f"  - ... and {remaining} more valid built-in case paths")
    return "\n".join(lines)
