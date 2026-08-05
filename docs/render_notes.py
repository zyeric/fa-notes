#!/usr/bin/env python3
"""Render the FlashAttention Markdown notes into one static HTML page."""

from __future__ import annotations

import html
import re
from pathlib import Path


BASE = Path(__file__).resolve().parent
NOTES = BASE / "notes"
OUTPUT = BASE / "notes.html"
TICK = chr(96)

DOCS = [
    ("README.md", "Reading Guide", "Reader paths, evidence contract, and scope."),
    ("tile-spec.md", "High-Level Tile Spec", "Canonical tiles, owner/reduction contracts, and representative FA1--FA4 lowerings."),
    ("evolution.md", "FA1 To FA4 Tile Atlas", "One tile grid, forward/backward loops, ownership, and the cross-generation mental model."),
    ("fa1-checkpoint.md", "FA1 Checkpoint", "Compact ownership, memory, and determinism re-entry."),
    ("fa1-foundations.md", "FA1 Foundations", "Online softmax through physical forward/backward execution."),
    ("fa2-forward.md", "FA2 Forward", "Q-block ownership and improved warp work partition."),
    ("fa2-backward.md", "FA2 Backward", "Backward ownership, partial dQ, and deterministic combine."),
    ("fa3.md", "FA3 / Hopper", "TMA, WGMMA, warp specialization, and two-level overlap."),
    ("fa4.md", "FA4 / Blackwell", "TMEM, tcgen05, 2-CTA cooperation, and softmax changes."),
    ("current-implementation-and-determinism.md", "Current Determinism Audit", "Resolved ownership and ordered-reduction mechanisms."),
    ("rubin-attention-projection.md", "Rubin Projection", "Attention opportunities, bottleneck migration, exactness boundaries, and validation questions."),
]


def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text).strip("-")
    return text or "section"


DOC_IDS = {name: "doc-" + slugify(Path(name).stem) for name, _, _ in DOCS}


def rewrite_href(href: str) -> str:
    path, marker, fragment = href.partition("#")
    name = Path(path).name
    if name in DOC_IDS:
        target = "#" + DOC_IDS[name]
        return target + ("-" + slugify(fragment) if marker and fragment else "")
    if path.startswith("../slides/"):
        return "slides/" + name + (("#" + fragment) if marker else "")
    return href


class Renderer:
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        self.ids: dict[str, int] = {}

    def unique_id(self, text: str) -> str:
        base = self.doc_id + "-" + slugify(text)
        count = self.ids.get(base, 0)
        self.ids[base] = count + 1
        return base if count == 0 else base + "-" + str(count + 1)

    def inline(self, text: str) -> str:
        tokens: list[str] = []

        def keep(value: str) -> str:
            tokens.append(value)
            return "\x00TOKEN" + str(len(tokens) - 1) + "\x00"

        code_pattern = re.escape(TICK) + r"([^" + re.escape(TICK) + r"]+)" + re.escape(TICK)
        text = re.sub(code_pattern, lambda m: keep("<code>" + html.escape(m.group(1)) + "</code>"), text)

        def image(match: re.Match[str]) -> str:
            alt = html.escape(match.group(1), quote=True)
            src = html.escape(rewrite_href(match.group(2)), quote=True)
            return keep('<img class="doc-image" src="' + src + '" alt="' + alt + '">')

        def link(match: re.Match[str]) -> str:
            label = html.escape(match.group(1))
            for index, token in enumerate(tokens):
                label = label.replace("\x00TOKEN" + str(index) + "\x00", token)
            href = html.escape(rewrite_href(match.group(2)), quote=True)
            return keep('<a href="' + href + '">' + label + "</a>")

        text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image, text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)
        text = html.escape(text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
        for index, token in enumerate(tokens):
            text = text.replace("\x00TOKEN" + str(index) + "\x00", token)
        return text

    @staticmethod
    def table_cells(line: str) -> list[str]:
        line = line.strip().strip("|")
        return [cell.strip() for cell in line.split("|")]

    @classmethod
    def table_separator(cls, line: str) -> bool:
        cells = cls.table_cells(line)
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)

    def render(self, source: str) -> str:
        lines = source.splitlines()
        out: list[str] = []
        paragraph: list[str] = []
        index = 0

        def flush() -> None:
            nonlocal paragraph
            if paragraph:
                out.append("<p>" + self.inline(" ".join(part.strip() for part in paragraph)) + "</p>")
                paragraph = []

        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""

            if not stripped:
                flush()
                index += 1
                continue

            if stripped.startswith(TICK * 3):
                flush()
                language = stripped[3:].strip() or "text"
                index += 1
                code_lines: list[str] = []
                while index < len(lines) and not lines[index].strip().startswith(TICK * 3):
                    code_lines.append(lines[index])
                    index += 1
                if index < len(lines):
                    index += 1
                out.append(
                    '<pre><span class="code-label">' + html.escape(language) + "</span><code>"
                    + html.escape("\n".join(code_lines)) + "</code></pre>"
                )
                continue

            heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading:
                flush()
                level = min(len(heading.group(1)) + 1, 6)
                title = heading.group(2)
                anchor = self.unique_id(title)
                out.append(
                    "<h" + str(level) + ' id="' + anchor + '">' + self.inline(title)
                    + '<a class="anchor" href="#' + anchor + '">#</a></h' + str(level) + ">"
                )
                index += 1
                continue

            if stripped.startswith("|") and next_line and self.table_separator(next_line):
                flush()
                header = self.table_cells(stripped)
                index += 2
                rows: list[list[str]] = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    rows.append(self.table_cells(lines[index]))
                    index += 1
                table = ['<div class="table-wrap"><table><thead><tr>']
                table.extend("<th>" + self.inline(cell) + "</th>" for cell in header)
                table.append("</tr></thead><tbody>")
                for row in rows:
                    table.append("<tr>")
                    table.extend("<td>" + self.inline(cell) + "</td>" for cell in row)
                    table.append("</tr>")
                table.append("</tbody></table></div>")
                out.append("".join(table))
                continue

            list_match = re.match(r"^([-*]|\d+\.)\s+(.+)$", stripped)
            if list_match:
                flush()
                ordered = list_match.group(1)[0].isdigit()
                tag = "ol" if ordered else "ul"
                items: list[str] = []
                while index < len(lines):
                    match = re.match(r"^([-*]|\d+\.)\s+(.+)$", lines[index].strip())
                    if not match or match.group(1)[0].isdigit() != ordered:
                        break
                    item = match.group(2)
                    index += 1
                    while index < len(lines) and lines[index].startswith(("  ", "\t")) and lines[index].strip():
                        item += " " + lines[index].strip()
                        index += 1
                    items.append("<li>" + self.inline(item) + "</li>")
                out.append("<" + tag + ">" + "".join(items) + "</" + tag + ">")
                continue

            if stripped.startswith(">"):
                flush()
                quotes: list[str] = []
                while index < len(lines) and lines[index].strip().startswith(">"):
                    quotes.append(lines[index].strip()[1:].strip())
                    index += 1
                out.append("<blockquote>" + self.inline(" ".join(quotes)) + "</blockquote>")
                continue

            paragraph.append(stripped)
            index += 1

        flush()
        return "\n".join(out)


STYLE = """
:root{--bg:#f3f6f5;--paper:#fff;--ink:#17202a;--muted:#64717f;--line:#d7e0e3;--teal:#00756a;--blue:#315f91;--orange:#9a571c;--code:#101820;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);line-height:1.64}
a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}.layout{display:grid;grid-template-columns:290px minmax(0,1fr);min-height:100vh}
aside{position:sticky;top:0;height:100vh;overflow:auto;padding:24px 18px;border-right:1px solid var(--line);background:#fbfcfc}
.brand{font-size:20px;font-weight:800;line-height:1.18}.meta{margin:8px 0 20px;color:var(--muted);font-size:12px}.nav a{display:block;padding:5px 0;color:#34414f;font-size:13px}
main{width:100%;max-width:1120px;padding:34px 42px 80px}.hero,.doc{margin-bottom:18px;padding:28px 30px;border:1px solid var(--line);border-radius:10px;background:var(--paper)}
.hero h1{margin:0 0 10px;font-size:38px;line-height:1.08}.subtitle{max-width:820px;color:var(--muted)}.actions{display:flex;flex-wrap:wrap;gap:9px;margin-top:20px}
.button{padding:9px 13px;border:1px solid var(--line);border-radius:7px;background:var(--ink);color:#fff;font-weight:750}.button.secondary{background:#fff;color:var(--ink)}
.source-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:22px}.source-card{padding:13px;border:1px solid var(--line);border-radius:8px;background:#fafcfc}.source-card strong{display:block}.source-card span{color:var(--muted);font-size:12px}
.doc-kicker{color:var(--orange);font-size:12px;font-weight:800;letter-spacing:.05em;text-transform:uppercase}.doc h2{margin:5px 0 16px;font-size:30px}.doc h3{margin:28px 0 10px;padding-top:18px;border-top:1px solid #edf0f2;font-size:23px}.doc h4{font-size:19px}.doc h5,.doc h6{font-size:16px}
.anchor{margin-left:7px;color:#a2adb6;font-size:.75em;opacity:0}.doc h2:hover .anchor,.doc h3:hover .anchor,.doc h4:hover .anchor{opacity:1}
p{margin:10px 0}ul,ol{padding-left:23px}li{margin:5px 0}code{padding:1px 4px;border-radius:4px;background:#edf2f5;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.92em}
pre{position:relative;overflow:auto;margin:14px 0;padding:34px 16px 16px;border-radius:8px;background:var(--code);color:#e8eef6}pre code{padding:0;background:transparent;color:inherit}.code-label{position:absolute;top:8px;left:12px;color:#9aa8b7;font-size:11px;text-transform:uppercase}
.table-wrap{overflow:auto;margin:14px 0;border:1px solid var(--line);border-radius:8px}table{width:100%;min-width:620px;border-collapse:collapse;font-size:14px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#eef4f3}
blockquote{margin:14px 0;padding:9px 15px;border-left:4px solid var(--teal);background:#f1faf8}.doc-image{display:block;max-width:100%;max-height:640px;margin:18px auto;border:1px solid var(--line);border-radius:8px;background:#fff}
@media(max-width:900px){.layout{display:block}aside{position:relative;height:auto;border-right:0;border-bottom:1px solid var(--line)}main{padding:20px 14px 60px}.hero,.doc{padding:20px}.source-grid{grid-template-columns:1fr}.hero h1{font-size:30px}}
@media print{aside{display:none}.layout{display:block}main{max-width:none;padding:0}.hero,.doc{border:0;break-inside:auto}body{background:#fff}}
"""


def build() -> str:
    cards = []
    nav = []
    sections = []
    for name, label, summary in DOCS:
        doc_id = DOC_IDS[name]
        cards.append(
            '<div class="source-card"><strong>' + html.escape(label) + "</strong><span>"
            + html.escape(summary) + "</span></div>"
        )
        nav.append('<a href="#' + doc_id + '">' + html.escape(label) + "</a>")
        source = (NOTES / name).read_text()
        rendered = Renderer(doc_id).render(source)
        sections.append(
            '<section class="doc" id="' + doc_id + '"><div class="doc-kicker">'
            + html.escape(name) + "</div>" + rendered + "</section>"
        )

    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlashAttention Reading Notes</title><style>""" + STYLE + """</style></head><body>
<div class="layout"><aside><div class="brand">FlashAttention<br>Reading Notes</div>
<div class="meta">Generated 2026-08-03<br>Markdown is the source of truth</div>
<nav class="nav"><a href="#top">Overview</a>""" + "".join(nav) + """</nav></aside>
<main><section class="hero" id="top"><h1>FlashAttention Reading Notes</h1>
<p class="subtitle">A progressive FA1-to-FA4 reading surface from online softmax and CTA ownership to Hopper/Blackwell pipelines and deterministic backward protocols.</p>
<div class="actions"><a class="button" href="slides/fa1-forward.html">Start with FA1 forward</a><a class="button secondary" href="index.html">Landing page</a><a class="button secondary" href="https://github.com/zyeric/fa-notes">GitHub</a></div>
<div class="source-grid">""" + "".join(cards) + """</div></section>
""" + "".join(sections) + """</main></div></body></html>"""


if __name__ == "__main__":
    OUTPUT.write_text(build())
    print("wrote", OUTPUT)
