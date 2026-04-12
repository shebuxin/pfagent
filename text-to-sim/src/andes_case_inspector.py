"""Inspect the active ANDES case to build an idx inventory for prompt context.

When the user says things like "trip line 18" or "modify the PQ at bus 8",
the model needs to know (a) the actual idx strings used in the case
(``"Line_18"`` vs ``"18"``) and (b) the device-to-bus topology. Without
this, the model's best guess is wrong about ~60% of the time (see the
user-feedback session on 2026-04-05 for IEEE39 trip-line failures).

This module loads the case headlessly via ``andes.load`` and emits a
compact prompt-ready text block listing every device's ``idx.v`` plus
its connection(s). Results are cached per case reference so only the
first turn in a case pays the ~1-second ANDES-load cost.

Layout of the rendered inventory:

    ANDES case idx inventory for ieee14/ieee14.raw:

    Bus (14 entries):
      idx = [1, 2, 3, ..., 14]

    Line (20 entries):
      idx  = ["Line_1", "Line_2", ..., "Line_20"]
      bus1 = [1, 1, 2, ..., 13]
      bus2 = [2, 5, 3, ..., 14]

    PQ (11 entries):
      idx  = ["PQ_1", ..., "PQ_11"]
      bus  = [2, 3, 4, ..., 14]
    ...

This is appended to the chat prompt so the model has ground-truth idx
values to reference, eliminating the "hardcoded '18'" class of bug.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Models to inspect, in the order they will appear in the rendered
# inventory. Bus comes first because many prompts resolve devices by
# bus number; Line / PQ / PV / Slack / Shunt cover ~99% of requests.
INSPECTED_MODELS: Tuple[str, ...] = ("Bus", "Line", "PQ", "PV", "Slack", "Shunt")

# Which attribute names to look for per model. Every model gets its
# idx list; models with topology get their bus linkage too.
_TOPOLOGY_ATTRS: Dict[str, Tuple[str, ...]] = {
    "Bus": (),
    "Line": ("bus1", "bus2"),
    "PQ": ("bus",),
    "PV": ("bus",),
    "Slack": ("bus",),
    "Shunt": ("bus",),
}

# Cap per-list length in the rendered output. Above this, keep the
# first/last halves separated by "...". Balances token cost vs
# information completeness. 100 entries comfortably covers IEEE39
# (46 lines, 19 PQ, 10 PV) in full and starts truncating on much
# larger systems (e.g. WECC).
MAX_ENTRIES_PER_FIELD: int = 100


# Session-level cache: cache_key -> rendered inventory string.
_INVENTORY_CACHE: Dict[str, str] = {}


def clear_inventory_cache() -> None:
    """Drop every cached inventory. Used by tests and by callers that
    know the case content has changed (e.g. user re-uploaded a file).
    """
    _INVENTORY_CACHE.clear()


def _resolve_case_path(
    case_source: str, case_reference: str, uploaded_dir: Optional[str]
) -> Optional[str]:
    """Return an absolute path suitable for ``andes.load`` or ``None``
    when the case cannot be located.
    """
    if case_source == "builtin":
        try:
            import andes  # heavy; defer import
        except Exception as exc:
            logger.warning("andes import failed: %s", exc)
            return None
        try:
            return andes.get_case(case_reference)
        except Exception as exc:
            logger.warning("get_case failed for %r: %s", case_reference, exc)
            return None

    basename = os.path.basename(case_reference or "")
    if case_source == "uploaded" and uploaded_dir and basename:
        candidate = os.path.join(uploaded_dir, basename)
        return candidate if os.path.exists(candidate) else None

    if case_source == "local":
        return case_reference if case_reference and os.path.exists(case_reference) else None

    return None


def _cache_key(
    case_source: str, case_reference: str, uploaded_dir: Optional[str]
) -> str:
    """Build a cache key. For uploaded cases, include the file mtime so
    a re-upload invalidates the cached inventory.
    """
    key = f"{case_source}::{case_reference}::{uploaded_dir or ''}"
    if case_source == "uploaded" and uploaded_dir:
        path = os.path.join(uploaded_dir, os.path.basename(case_reference))
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        key = f"{key}::mtime={mtime}"
    return key


def _format_list(values: List[Any]) -> str:
    """Render a list of idx/bus values as a compact Python-like literal.

    Strings get quoted, ints do not. Long lists get truncated in the
    middle (see ``MAX_ENTRIES_PER_FIELD``).
    """
    def _repr(v: Any) -> str:
        if isinstance(v, str):
            return f'"{v}"'
        try:
            # numpy scalars render as np.int64(...) by default; coerce
            # to a plain Python type for a clean literal.
            return str(int(v)) if float(v).is_integer() else str(float(v))
        except (TypeError, ValueError):
            return str(v)

    total = len(values)
    if total <= MAX_ENTRIES_PER_FIELD:
        return "[" + ", ".join(_repr(v) for v in values) + "]"

    half = MAX_ENTRIES_PER_FIELD // 2
    head = [_repr(v) for v in values[:half]]
    tail = [_repr(v) for v in values[-half:]]
    return "[" + ", ".join(head) + ", ..., " + ", ".join(tail) + "]"


def _inspect_model(ssa: Any, model_name: str) -> Optional[str]:
    """Render a single model's inventory section, or ``None`` when the
    model is empty or missing.
    """
    model = getattr(ssa, model_name, None)
    if model is None:
        return None

    idx_attr = getattr(model, "idx", None)
    if idx_attr is None:
        return None

    try:
        idx_values = list(getattr(idx_attr, "v", []) or [])
    except Exception:
        return None
    if not idx_values:
        return None

    count = len(idx_values)
    lines = [f"{model_name} ({count} {'entry' if count == 1 else 'entries'}):"]
    lines.append(f"  idx  = {_format_list(idx_values)}")

    for attr_name in _TOPOLOGY_ATTRS.get(model_name, ()):
        attr = getattr(model, attr_name, None)
        if attr is None:
            continue
        try:
            values = list(getattr(attr, "v", []) or [])
        except Exception:
            continue
        if not values:
            continue
        padding = " " * max(0, 4 - len(attr_name))
        lines.append(f"  {attr_name}{padding} = {_format_list(values)}")

    return "\n".join(lines)


_GUIDANCE_FOOTER: str = (
    "When the user refers to a device by a small integer "
    '("line 18", "bus 8", "PQ at bus 5"), resolve it against the inventory '
    "above:\n"
    '- "line N" with small N → Line.idx.v[N-1] (1-indexed position); '
    'do NOT assume the idx literal is "N".\n'
    '- "line between bus X and Y" → find the index where bus1==X and '
    "bus2==Y (pair is symmetric).\n"
    '- "bus N" → the literal integer N (Bus.idx values are the bus '
    "numbers themselves).\n"
    '- "PQ at bus N" → resolve to the PQ entry whose bus==N; use that '
    "entry's idx.\n"
    "- If the user gives an exact idx string that matches the inventory "
    "(e.g. \"Line_18\"), use it directly."
)


def build_case_idx_inventory(
    case_source: str,
    case_reference: str,
    uploaded_dir: Optional[str] = None,
) -> str:
    """Return a prompt-ready inventory of device idx values for a case.

    Parameters
    ----------
    case_source:
        One of ``"builtin"``, ``"uploaded"``, ``"local"``. Matches the
        schema used by ``session_state.active_andes_case``.
    case_reference:
        The case path / filename. For ``builtin`` this is a relative
        path under the ANDES case directory (e.g. ``"ieee14/ieee14.raw"``).
        For ``uploaded`` this is just the file basename.
    uploaded_dir:
        The session's runtime data directory. Only used when
        ``case_source == "uploaded"``. Ignored otherwise.

    Returns
    -------
    A multi-line string ready to be concatenated into the chat prompt,
    or ``""`` when the case cannot be loaded (caller should silently
    proceed without the inventory block).
    """
    if not (case_source and case_reference):
        return ""

    key = _cache_key(case_source, case_reference, uploaded_dir)
    cached = _INVENTORY_CACHE.get(key)
    if cached is not None:
        return cached

    path = _resolve_case_path(case_source, case_reference, uploaded_dir)
    if not path:
        return ""

    try:
        import andes  # heavy; defer import
        ssa = andes.load(path, setup=True, no_output=True, log=False)
    except Exception as exc:
        logger.warning(
            "Failed to load case for inventory (source=%s, ref=%s): %s",
            case_source, case_reference, exc,
        )
        return ""

    sections: List[str] = [f"ANDES case idx inventory for {case_reference}:"]
    for model_name in INSPECTED_MODELS:
        section = _inspect_model(ssa, model_name)
        if section:
            sections.append(section)

    if len(sections) < 2:
        return ""

    sections.append(_GUIDANCE_FOOTER)
    rendered = "\n\n".join(sections)
    _INVENTORY_CACHE[key] = rendered
    return rendered
