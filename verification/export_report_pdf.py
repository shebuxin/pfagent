#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKDOWN = REPO_ROOT / "verification" / "current_test_report_20260328.md"
DEFAULT_OUTPUT = REPO_ROOT / "verification" / "current_test_report_20260328.pdf"
PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
LEFT = 0.08
RIGHT = 0.92
TOP = 0.94
BOTTOM = 0.06
LINE_HEIGHT = 0.028


def clean_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = text.replace("**", "")
    return text.strip()


def parse_markdown(markdown_path: Path) -> List[Tuple[str, object]]:
    content = markdown_path.read_text(encoding="utf-8").splitlines()
    tokens: List[Tuple[str, object]] = []
    base_dir = markdown_path.parent

    for raw_line in content:
        line = raw_line.rstrip()
        if not line:
            tokens.append(("blank", ""))
            continue

        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if image_match:
            alt_text = image_match.group(1).strip() or "Figure"
            image_path = (base_dir / image_match.group(2)).resolve()
            tokens.append(("image", {"caption": alt_text, "path": image_path}))
            continue

        if line.startswith("# "):
            tokens.append(("h1", clean_inline_markdown(line[2:])))
            continue
        if line.startswith("## "):
            tokens.append(("h2", clean_inline_markdown(line[3:])))
            continue
        if line.startswith("### "):
            tokens.append(("h3", clean_inline_markdown(line[4:])))
            continue
        if re.match(r"^\d+\.\s+", line):
            tokens.append(("numbered", clean_inline_markdown(line)))
            continue
        if line.startswith("- "):
            tokens.append(("bullet", clean_inline_markdown(line[2:])))
            continue
        if line.startswith("|"):
            tokens.append(("table", clean_inline_markdown(line)))
            continue

        tokens.append(("paragraph", clean_inline_markdown(line)))
    return tokens


def wrap_text(text: str, width: int) -> Iterable[str]:
    if not text:
        return [""]
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text]


def new_text_page():
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    fig.patch.set_facecolor("white")
    return fig


def draw_text(fig, y: float, text: str, *, size: int = 11, weight: str = "normal", color: str = "#111111", x: float = LEFT):
    fig.text(x, y, text, fontsize=size, fontweight=weight, color=color, va="top", ha="left", family="DejaVu Sans")


def render_text_pages(tokens: List[Tuple[str, object]], pdf: PdfPages) -> None:
    fig = new_text_page()
    y = TOP

    def ensure_space(lines_needed: int = 1):
        nonlocal fig, y
        needed = lines_needed * LINE_HEIGHT
        if y - needed < BOTTOM:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = new_text_page()
            y = TOP

    for kind, payload in tokens:
        if kind == "image":
            if y < TOP:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            render_image_page(payload["path"], payload["caption"], pdf)
            fig = new_text_page()
            y = TOP
            continue

        if kind == "blank":
            y -= LINE_HEIGHT * 0.6
            continue

        if kind == "h1":
            ensure_space(2)
            draw_text(fig, y, payload, size=22, weight="bold", color="#0c2d48")
            y -= LINE_HEIGHT * 1.8
            continue

        if kind == "h2":
            ensure_space(2)
            draw_text(fig, y, payload, size=16, weight="bold", color="#145374")
            y -= LINE_HEIGHT * 1.5
            continue

        if kind == "h3":
            ensure_space(2)
            draw_text(fig, y, payload, size=13, weight="bold", color="#1b4965")
            y -= LINE_HEIGHT * 1.25
            continue

        if kind == "table":
            width = 92
            lines = list(wrap_text(payload, width))
            ensure_space(len(lines) + 1)
            for line in lines:
                draw_text(fig, y, line, size=8, x=LEFT, color="#222222")
                y -= LINE_HEIGHT * 0.82
            continue

        if kind == "bullet":
            lines = list(wrap_text(payload, 88))
            ensure_space(len(lines) + 1)
            for idx, line in enumerate(lines):
                prefix = "• " if idx == 0 else "  "
                draw_text(fig, y, prefix + line, size=10)
                y -= LINE_HEIGHT * 0.95
            continue

        if kind == "numbered":
            lines = list(wrap_text(payload, 88))
            ensure_space(len(lines) + 1)
            for line in lines:
                draw_text(fig, y, line, size=10)
                y -= LINE_HEIGHT * 0.95
            continue

        if kind == "paragraph":
            lines = list(wrap_text(payload, 95))
            ensure_space(len(lines) + 1)
            for line in lines:
                draw_text(fig, y, line, size=10)
                y -= LINE_HEIGHT * 0.95
            continue

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_image_page(image_path: Path, caption: str, pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    fig.patch.set_facecolor("white")

    draw_text(fig, TOP, caption, size=16, weight="bold", color="#145374")

    ax = fig.add_axes([0.08, 0.14, 0.84, 0.72])
    ax.axis("off")

    if image_path.exists():
        image = mpimg.imread(image_path)
        ax.imshow(image)
    else:
        ax.text(0.5, 0.5, f"Missing image:\n{image_path}", ha="center", va="center", fontsize=12)

    fig.text(LEFT, 0.08, str(image_path.relative_to(REPO_ROOT)), fontsize=8, color="#555555", ha="left", va="bottom")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def export_pdf(markdown_path: Path, output_path: Path) -> None:
    tokens = parse_markdown(markdown_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        render_text_pages(tokens, pdf)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the current verification Markdown report to PDF.")
    parser.add_argument("--input", default=str(DEFAULT_MARKDOWN), help="Markdown input path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="PDF output path.")
    args = parser.parse_args()

    export_pdf(Path(args.input).resolve(), Path(args.output).resolve())
    print(f"PDF written to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
