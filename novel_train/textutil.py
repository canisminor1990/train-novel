"""Clean novel text and cut it into continuation windows. Standard library only."""

from __future__ import annotations

import hashlib
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

_SENT_END = set("。！？…!?~～”’」』）)》】—\"'")
_AD_RE = re.compile(
    "|".join(
        [
            r"https?://\S+",
            r"www\.\S+",
            r"[A-Za-z0-9]+\.(?:com|net|org|cc|la|info|me)\b",
            r"^\s*(?:PS|Ps|ps|P\.S\.)\s*[:：.,，]",
            r"^\s*作者有话[说要]",
            r"^\s*作者的话",
            r"手机用户请",
            r"txt\s*下载",
            r"请收藏本站",
            r"请记住本站",
            r"一秒记住",
            r"天才一秒记住",
            r"本书来自",
            r"最新网址",
            r"笔趣阁",
            r"顶点小说",
            r"无弹窗",
            r"本章未完",
            r"更新最快",
            r"纯文字在线阅读",
            r"QQ群[:： ]?\d*",
            r"加群[:： ]?\d*",
        ]
    )
)
_AD_SHORT_RE = re.compile(
    r"手机阅读|免费阅读|最新章节|章节报错|本章完|未完待续|求(?:月票|推荐票|订阅|收藏|打赏)"
    r"|感谢.{0,24}(?:打赏|月票|推荐票|盟主)"
)
_AD_SHORT_MAX = 50
_CHAPTER_RE = re.compile(
    r"^(?:正文\s*)?(第[0-9零〇一二三四五六七八九十百千两]+[章节回部卷集])"
    r"(?:[\s：:.\-—]*(\S.{0,30}?))?\s*$"
    r"|^(?:chapter|Chapter|CHAPTER)\s*\d+(?:[\s：:.\-—]+\S.{0,30})?\s*$"
)
_ZERO_WIDTH_RE = re.compile("[\u200b-\u200f\u2028\u2029\u2060\ufeff\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CJK = "\u4e00-\u9fff"
_CJK_SPACE_RE = re.compile(f"(?<=[{_CJK}，。！？、：；“”‘’]) +(?=[{_CJK}，。！？、：；“”‘’])")
_ASCII_PUNCT = {",": "，", "!": "！", "?": "？", ";": "；", ":": "："}
_ASCII_PUNCT_RE = re.compile(f"(?<=[{_CJK}])([,!?;:])")
_ELLIPSIS_RE = re.compile(r"(?:\.{3,}|。{3,}|…+|·{3,})")
_DASH_RE = re.compile(r"(?:—|－|-){2,}")
_SPECIAL_CHAPTER_RE = re.compile(r"^(序章|楔子|尾声|终章|番外|后记|前言|引子)(?:\s+\S.{0,20})?$")
_PAGE_RE = re.compile(r"^(?:第?\s*)?[0-9０-９]{1,4}\s*页?$")
_RULE_RE = re.compile(r"^[-—_=*·.。]{3,}$")


def count_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gb18030", errors="replace")


class _HtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self.skip += 1
        if tag in {"p", "div", "br", "h1", "h2", "h3", "h4", "li", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.skip:
            self.skip -= 1
        if tag in {"p", "div", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip == 0:
            self.parts.append(data)


def _epub_order(zf: zipfile.ZipFile) -> list[str]:
    names = set(zf.namelist())
    htmls = [
        name
        for name in zf.namelist()
        if name.lower().endswith((".xhtml", ".html", ".htm"))
        and "toc" not in name.lower()
        and not name.lower().endswith("nav.xhtml")
    ]
    opfs = [name for name in zf.namelist() if name.lower().endswith(".opf")]
    if not opfs:
        return sorted(htmls)
    try:
        root = ET.fromstring(zf.read(opfs[0]))
    except ET.ParseError:
        return sorted(htmls)

    manifest: dict[str, str] = {}
    spine: list[str] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "item":
            item_id = el.attrib.get("id")
            href = el.attrib.get("href")
            media = el.attrib.get("media-type", "")
            if item_id and href and ("html" in media or href.lower().endswith((".xhtml", ".html", ".htm"))):
                manifest[item_id] = href.split("#", 1)[0]
        elif tag == "itemref":
            ref = el.attrib.get("idref")
            if ref:
                spine.append(ref)

    base = posixpath.dirname(opfs[0])
    ordered: list[str] = []
    for item_id in spine:
        href = manifest.get(item_id)
        if not href:
            continue
        full = posixpath.normpath(posixpath.join(base, href) if base else href)
        if full in names:
            ordered.append(full)
            continue
        base_name = posixpath.basename(href)
        matches = [name for name in htmls if name.endswith("/" + base_name) or name == base_name]
        if matches:
            ordered.append(matches[0])
    return ordered or sorted(htmls)


def read_epub(path: Path) -> str:
    chunks: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for name in _epub_order(zf):
            data = zf.read(name)
            try:
                html = data.decode("utf-8")
            except UnicodeDecodeError:
                html = data.decode("gb18030", errors="replace")
            parser = _HtmlText()
            parser.feed(html)
            chunks.append("".join(parser.parts))
    return "\n".join(chunks)


def load_document(path: Path) -> str:
    if path.suffix.lower() == ".epub":
        return read_epub(path)
    return read_text_file(path)


def _fullwidth_alnum(text: str) -> str:
    return "".join(
        chr(ord(ch) - 0xFEE0) if "０" <= ch <= "９" or "Ａ" <= ch <= "Ｚ" or "ａ" <= ch <= "ｚ" else ch
        for ch in text
    )


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\u00a0", " ").replace("\u3000", " ").replace("\t", " ")
    return _fullwidth_alnum(text)


def _tidy_line(line: str) -> str:
    """Normalise spacing and punctuation inside one Chinese line."""
    line = re.sub(r" {2,}", " ", line).strip()
    line = _CJK_SPACE_RE.sub("", line)
    line = _ASCII_PUNCT_RE.sub(lambda match: _ASCII_PUNCT[match.group(1)], line)
    line = _ELLIPSIS_RE.sub("……", line)
    line = _DASH_RE.sub("——", line)
    return line


def _is_chapter_title(line: str) -> bool:
    if len(line) > 40:
        return False
    return bool(_CHAPTER_RE.match(line) or _SPECIAL_CHAPTER_RE.match(line))


def _is_junk(line: str) -> bool:
    if _AD_RE.search(line):
        return True
    if len(line) <= _AD_SHORT_MAX and _AD_SHORT_RE.search(line):
        return True
    if _PAGE_RE.match(line) or _RULE_RE.match(line):
        return True
    if not has_cjk(line) and len(line) < 12:
        return True
    return False


_MID_SENTENCE = set("，、；：“‘《（(")
_LONG_PARAGRAPH = 600


def _split_long_line(line: str) -> list[str]:
    """Break a runaway paragraph (often a whole chapter on one line) at sentence ends."""
    if count_chars(line) <= _LONG_PARAGRAPH:
        return [line]
    sentences = [part for part in re.split(r"(?<=[。！？…][”’」』]?)", line) if part.strip()]
    parts: list[str] = []
    buf = ""
    for sentence in sentences:
        buf += sentence
        if count_chars(buf) >= _LONG_PARAGRAPH // 2:
            parts.append(buf.strip())
            buf = ""
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _is_hard_wrapped(lines: list[str]) -> bool:
    """A block typed with fixed-width line breaks: most lines stop mid-sentence."""
    if len(lines) < 3:
        return False
    open_ends = sum(1 for line in lines[:-1] if line[-1] not in _SENT_END)
    return open_ends / (len(lines) - 1) >= 0.5


def _join_lines(lines: list[str], hard_wrapped: bool) -> tuple[list[str], int]:
    paragraphs: list[str] = []
    merged = 0
    buf = ""
    for line in lines:
        if buf and count_chars(buf) < _LONG_PARAGRAPH * 2 and (
            buf[-1] in _MID_SENTENCE or (hard_wrapped and buf[-1] not in _SENT_END)
        ):
            buf += line
            merged += 1
            continue
        if buf:
            paragraphs.append(buf)
        buf = line
    if buf:
        paragraphs.append(buf)
    return paragraphs, merged


def _paragraphs(lines: list[str]) -> tuple[list[str], int]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)

    paragraphs: list[str] = []
    merged = 0
    for block in blocks:
        joined, count = _join_lines(block, _is_hard_wrapped(block))
        merged += count
        for paragraph in joined:
            paragraphs.extend(_split_long_line(paragraph))
    return [paragraph for paragraph in paragraphs if has_cjk(paragraph)], merged


def clean_document(text: str) -> tuple[list[tuple[str, list[str]]], dict]:
    """Return (chapters, report). Each chapter is (title, paragraphs)."""
    report = {"junk_lines": 0, "duplicate_lines": 0, "merged_lines": 0, "duplicate_chapters": 0}
    lines: list[str] = []
    for raw in _normalize(text).splitlines():
        line = raw.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if _is_junk(line):
            report["junk_lines"] += 1
            continue
        line = re.sub(r" {2,}", " ", line)
        if not _is_chapter_title(line):
            line = _tidy_line(line)
        previous = next((item for item in reversed(lines) if item), None)
        if line == previous and count_chars(line) >= 10:
            report["duplicate_lines"] += 1
            continue
        lines.append(line)

    raw_chapters: list[tuple[str, list[str]]] = []
    title = "正文"
    body: list[str] = []
    for line in lines:
        if line and _is_chapter_title(line):
            if any(body):
                raw_chapters.append((title, body))
            title = line
            body = []
            continue
        if line and line == title and not any(body):
            report["duplicate_lines"] += 1
            continue
        body.append(line)
    if any(body):
        raw_chapters.append((title, body))

    chapters: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for chapter_title, chapter_lines in raw_chapters:
        paragraphs, merged = _paragraphs(chapter_lines)
        if not paragraphs:
            continue
        digest = hashlib.sha1("\n".join(paragraphs).encode("utf-8")).hexdigest()
        if digest in seen:
            report["duplicate_chapters"] += 1
            continue
        seen.add(digest)
        report["merged_lines"] += merged
        chapters.append((chapter_title, paragraphs))
    return chapters, report


def cut_at_sentence(text: str, max_chars: int) -> str:
    if count_chars(text) <= max_chars:
        return text.strip()
    kept = 0
    last_end = -1
    for index, char in enumerate(text):
        if not char.isspace():
            kept += 1
        if char in _SENT_END and kept <= max_chars:
            last_end = index
        if kept >= max_chars:
            if last_end >= 0:
                return text[: last_end + 1].strip()
            return text[: index + 1].strip()
    if last_end >= 0:
        return text[: last_end + 1].strip()
    return text.strip()


def tail_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if count_chars(text) <= max_chars:
        return text
    kept = 0
    start = 0
    for index in range(len(text) - 1, -1, -1):
        if not text[index].isspace():
            kept += 1
        if kept >= max_chars:
            start = index
            break
    snippet = text[start:].strip()
    newline = snippet.find("\n")
    if newline != -1 and count_chars(snippet[newline + 1 :]) >= max(80, max_chars // 2):
        snippet = snippet[newline + 1 :].strip()
    return snippet


def iter_windows(
    paragraphs: list[str],
    context_chars: int,
    context_min: int,
    output_min: int,
    output_max: int,
    stride_chars: int,
):
    lengths = [count_chars(paragraph) for paragraph in paragraphs]
    total = len(paragraphs)
    start = 0
    while start < total:
        output_chars = 0
        end = start
        while end < total and output_chars < output_min:
            output_chars += lengths[end]
            end += 1
        if output_chars < output_min:
            return

        while end - start > 1 and output_chars - lengths[end - 1] >= output_min and output_chars > output_max:
            end -= 1
            output_chars -= lengths[end]

        output = "\n".join(paragraphs[start:end]).strip()
        if count_chars(output) > int(output_max * 1.15):
            output = cut_at_sentence(output, output_max)
        if count_chars(output) < output_min:
            start += 1
            continue

        context_count = 0
        context_start = start
        while context_start > 0 and context_count < context_chars:
            context_start -= 1
            context_count += lengths[context_start]
        while context_start < start - 1 and context_count > context_chars:
            context_count -= lengths[context_start]
            context_start += 1
        context = "\n".join(paragraphs[context_start:start]).strip()
        if count_chars(context) > context_chars:
            context = tail_text(context, context_chars)
        if count_chars(context) >= context_min and output != context:
            yield context, output, start

        advanced = 0
        step = 0
        while start + step < total and advanced < stride_chars:
            advanced += lengths[start + step]
            step += 1
        start += max(step, 1)


def self_check() -> None:
    body = "\n".join(
        f"第{index + 1}滴雨打在铁皮棚上，声音密得像有人筛豆子。"
        for index in range(80)
    )
    raw = "请收藏本站 www.bad.example\n第一章 雨夜\n" + body + "\n第一章 雨夜\n" + body
    chapters, report = clean_document(raw)
    ads = report["junk_lines"]
    if ads < 1:
        raise SystemExit("self-check failed: ad line was kept")

    sample = "\n".join(
        [
            "第二章 码头",
            "第二章 码头",
            "",
            "",
            "　　他把伞收起来，雨水顺着伞骨",
            "滴到鞋面上，凉得发麻，可他没有",
            "动，只盯着对岸那盏灯。",
            "",
            "“走。”",
            "“去哪？”",
            "“别问。”",
            "她 说 完 就笑了......",
            "（本章完）",
            "要求票价便宜一点的那班船，他说。",
        ]
    )
    (title, paragraphs), = clean_document(sample)[0]
    expected = [
        "他把伞收起来，雨水顺着伞骨滴到鞋面上，凉得发麻，可他没有动，只盯着对岸那盏灯。",
        "“走。”",
        "“去哪？”",
        "“别问。”",
        "她说完就笑了……",
        "要求票价便宜一点的那班船，他说。",
    ]
    if title != "第二章 码头" or paragraphs != expected:
        raise SystemExit(f"self-check failed: cleaning gave {title!r} {paragraphs!r}")
    if len(chapters) != 1:
        raise SystemExit(f"self-check failed: expected 1 chapter, got {len(chapters)}")
    windows = list(iter_windows(chapters[0][1], 1200, 300, 200, 800, 400))
    if not windows:
        raise SystemExit("self-check failed: no windows")
    context, output, _start = windows[0]
    output_size = count_chars(output)
    if not 200 <= output_size <= 920:
        raise SystemExit(f"self-check failed: output length {output_size}")
    if count_chars(context) < 300:
        raise SystemExit("self-check failed: context too short")
    print(f"self-check ok  ads={ads}  windows={len(windows)}  output_chars={output_size}")
