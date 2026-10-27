"""Does every number in an answer come from somewhere? The bench's checker.

The assistant's promise is that it never invents a figure: the agent's numbers are quoted, market
numbers are cited, and everything computed comes from a tool. This module checks the promise
after the fact, number by number, against what the model was actually shown:

``facts``     the FACTS block it was given (the agent's inputs, the precomputed results and the
              cost parameters — every figure in that text);
``tool``      every number in a tool result of the conversation, raw and pre-formatted;
``sourced``   a number the model itself put into a tool call together with a source (a market
              rent, a benchmark price per m², a mortgage rate) — traced to the source it named;
``cited``     a number inside the ``cited_text`` of a web citation attached to the answer, or
              in page text the model printed while filtering search results (the dynamic
              web search runs code over the results, and its output is in the conversation);
``linked``    a market number attributed to a page the search actually returned in this
              conversation: on the same line as a markdown link to it, or in the market-data
              section that links it, or a restatement of such a number elsewhere in the answer.
              Weaker than ``cited`` — the page text is encrypted in the response, so the number
              itself cannot be checked against it — but it rests on a real, retrieved source
              rather than on nothing;
``trivial``   small counts, months, years, dates and "3+1" layouts, which are not claims.

Links are checked too: :func:`unretrieved_links` lists every URL in the answer that no search in
the conversation returned — a link the model wrote from memory.

Anything else is ``untraced``. An untraced number is not necessarily wrong — the model may have
quoted a search result without a citation — but it is exactly the kind of number the rules
forbid, and the bench reports every one of them.

Matching is at the answer's own precision: "%4,9" matches 4,8917 (rounded to one decimal), and
"7,85 milyon" matches 7.850.000. It does not accept "about": a figure that is off at the precision
it is written in is untraced.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

__all__ = [
    "Number",
    "Audit",
    "extract_numbers",
    "collect_allowed",
    "audit_answer",
    "advice_flags",
    "retrieved_urls",
    "unretrieved_links",
]

_URL_RE = re.compile(r"\]\((https?://[^)\s]+)\)|https?://\S+")
#: dd.mm.yyyy and yyyy-mm(-dd): dates, not quantities.
_DATE_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b(19|20)\d{2}-\d{2}(-\d{2})?\b")
#: "3+1", "2+1" — a layout, not a sum.
_LAYOUT_RE = re.compile(r"\b\d\s*\+\s*\d\b")
#: A number with its optional percent sign (Turkish puts it first) and scale word.
_NUMBER_RE = re.compile(
    r"(?P<pct_before>%\s?)?"
    r"(?P<num>\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
    r"(?P<pct_after>\s?%)?"
    r"(?:\s*(?P<scale>milyar|milyon|bin|billion|million|thousand|k)\b)?",
    re.I,
)
_SCALES = {"milyar": 1e9, "billion": 1e9, "milyon": 1e6, "million": 1e6, "bin": 1e3, "thousand": 1e3, "k": 1e3}
_MONTH_WORDS = (
    "ocak", "şubat", "subat", "mart", "nisan", "mayıs", "mayis", "haziran", "temmuz", "ağustos", "agustos",
    "eylül", "eylul", "ekim", "kasım", "kasim", "aralık", "aralik", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october", "november", "december",
)


@dataclass(frozen=True, slots=True)
class Number:
    """One number as written in the answer."""

    text: str
    value: float
    #: Decimals as written, in the unit written (``7,85 milyon`` has 2, in millions).
    decimals: int
    scale: float
    percent: bool


def _parse(raw: str) -> tuple[float, int] | None:
    """A Turkish or English numeral to (value, decimals). ``None`` if it is not one."""
    s = raw.strip()
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):  # 1.234,56 — Turkish
            whole, frac = s.rsplit(",", 1)
            whole = whole.replace(".", "")
        else:  # 1,234.56 — English
            whole, frac = s.rsplit(".", 1)
            whole = whole.replace(",", "")
        return float(f"{whole}.{frac}"), len(frac)
    for sep in (".", ","):
        if sep in s:
            parts = s.split(sep)
            if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3 and parts[0] != "0"):
                # 7.850.000 / 1.500 / 7,850,000: grouping, not decimals.
                if all(len(p) == 3 for p in parts[1:]):
                    return float("".join(parts)), 0
            if len(parts) == 2:
                return float(f"{parts[0]}.{parts[1]}"), len(parts[1])
            return None
    return float(s), 0


def extract_numbers(answer: str) -> tuple[list[Number], list[str]]:
    """Every quantity in ``answer``, and the trivial ones it skipped (for the report)."""
    text = _URL_RE.sub(" ", answer)
    skipped: list[str] = []
    for pattern in (_DATE_RE, _LAYOUT_RE):
        for match in pattern.finditer(text):
            skipped.append(match.group(0))
        text = pattern.sub(" ", text)
    out: list[Number] = []
    for match in _NUMBER_RE.finditer(text):
        parsed = _parse(match.group("num"))
        if parsed is None:
            continue
        value, decimals = parsed
        scale_word = (match.group("scale") or "").lower()
        scale = _SCALES.get(scale_word, 1.0)
        percent = bool(match.group("pct_before") or match.group("pct_after"))
        tail = text[match.end(): match.end() + 12].lower()
        head = text[max(0, match.start() - 12): match.start()].lower()
        is_year = decimals == 0 and scale == 1.0 and not percent and 1990 <= value <= 2040
        is_dated = any(m in tail or m in head for m in _MONTH_WORDS) and decimals == 0 and value <= 31 and not percent
        is_list_marker = bool(re.match(r"\s*[.)]\s", text[match.end(): match.end() + 3])) and value <= 20 and not percent and decimals == 0
        if is_year or is_dated or is_list_marker:
            skipped.append(match.group(0).strip())
            continue
        out.append(Number(text=match.group(0).strip(), value=value * scale, decimals=decimals, scale=scale, percent=percent))
    return out, skipped


def _numbers_in(value: Any) -> list[float]:
    """Every number in a JSON value, including those written inside its strings."""
    found: list[float] = []
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            found.append(float(value))
    elif isinstance(value, str):
        numbers, _ = extract_numbers(value)
        found.extend(n.value for n in numbers)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_numbers_in(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_numbers_in(item))
    return found


@dataclass
class Allowed:
    """The numbers the model was entitled to write, by provenance."""

    facts: list[float] = field(default_factory=list)
    tool: list[float] = field(default_factory=list)
    sourced: list[float] = field(default_factory=list)
    cited: list[float] = field(default_factory=list)
    #: URLs the searches of the conversation returned.
    urls: set[str] = field(default_factory=set)


_LINK_RE = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")


def _norm_url(url: str) -> str:
    return url.strip().rstrip("/").split("#")[0].lower()


_BARE_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")


def retrieved_urls(messages: list[dict[str, Any]], facts_text: str = "") -> set[str]:
    """Every URL the model was actually given: search results, citations, URLs inside tool
    results (the rates' sources) and in the facts block. Normalised."""
    urls: set[str] = {_norm_url(u) for u in _BARE_URL_RE.findall(facts_text)}
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "web_search_tool_result" and isinstance(block.get("content"), list):
                urls.update(_norm_url(str(item.get("url", ""))) for item in block["content"] if item.get("url"))
            if block.get("type") == "tool_result":
                urls.update(_norm_url(u) for u in _BARE_URL_RE.findall(str(block.get("content", ""))))
            for citation in block.get("citations") or []:
                if citation.get("url"):
                    urls.add(_norm_url(str(citation["url"])))
    return urls


def unretrieved_links(answer: str, messages: list[dict[str, Any]], facts_text: str = "") -> list[str]:
    """Links in the answer to pages the model was never given — written from memory."""
    known = retrieved_urls(messages, facts_text)
    return [url for url in _LINK_RE.findall(answer) if _norm_url(url) not in known]


def collect_allowed(facts_text: str, messages: list[dict[str, Any]], answer_blocks: list[dict[str, Any]]) -> Allowed:
    """What the conversation gave the model, and which citations the answer carries."""
    allowed = Allowed()
    allowed.facts = [n.value for n in extract_numbers(facts_text)[0]]
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            kind = block.get("type")
            if kind == "tool_result":
                body = block.get("content")
                try:
                    payload = json.loads(body) if isinstance(body, str) else body
                except ValueError:
                    payload = body
                allowed.tool.extend(_numbers_in(payload))
            elif kind == "tool_use":
                allowed.sourced.extend(_numbers_in(block.get("input")))
            elif kind in ("code_execution_tool_result", "bash_code_execution_tool_result"):
                # Page text the dynamic web search printed while filtering results: web content
                # the model really saw, in clear (the search results themselves are encrypted).
                result = block.get("content") or {}
                stdout = result.get("stdout", "") if isinstance(result, dict) else ""
                allowed.cited.extend(n.value for n in extract_numbers(str(stdout))[0])
    for block in answer_blocks:
        for citation in block.get("citations") or []:
            allowed.cited.extend(n.value for n in extract_numbers(str(citation.get("cited_text", "")))[0])
    allowed.urls = retrieved_urls(messages, facts_text)
    return allowed


def _round(value: float, decimals: int) -> float:
    return float(Decimal(repr(value)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def _matches(number: Number, candidate: float) -> bool:
    written = number.value / number.scale
    target = candidate / number.scale
    if _round(target, number.decimals) == _round(written, number.decimals):
        return True
    # A percentage may be written from a fraction, or the reverse (0,0489 vs %4,89).
    if number.percent and _round(candidate * 100.0, number.decimals) == _round(written, number.decimals):
        return True
    return False


@dataclass
class Audit:
    numbers: int
    traced: dict[str, int]
    untraced: list[str]
    trivial: list[str]

    @property
    def ok(self) -> bool:
        return not self.untraced


_MARKET_LABEL_RE = re.compile(r"^\s*\*\*(piyasa verileri|market data)\*\*", re.I)
_LABEL_RE = re.compile(r"^\s*\*\*[^*]+\*\*\s*$")


def _linked_lines(answer: str, urls: set[str]) -> set[int]:
    """Lines whose numbers rest on a retrieved page: lines that link one, and every line of a
    market-data section that links one (sources are often given once per bullet group)."""
    lines = answer.splitlines()
    linked = {i for i, line in enumerate(lines) if any(_norm_url(u) in urls for u in _LINK_RE.findall(line))}
    section: list[int] = []
    in_market = False
    for index, line in enumerate([*lines, "**end**"]):
        if _LABEL_RE.match(line) or _MARKET_LABEL_RE.match(line):
            if in_market and linked & set(section):
                linked.update(section)
            in_market = bool(_MARKET_LABEL_RE.match(line))
            section = []
        elif in_market:
            section.append(index)
    return linked


def audit_answer(answer: str, allowed: Allowed) -> Audit:
    traced = {"facts": 0, "tool": 0, "sourced": 0, "cited": 0, "linked": 0, "trivial": 0}
    untraced: list[str] = []
    skipped_all: list[str] = []
    count = 0
    linked = _linked_lines(answer, allowed.urls)
    lines = answer.splitlines()
    # Numbers on linked lines may be restated elsewhere — "about %32" in the opening sentence of
    # a %32,47 the market section links. A restatement at the written precision counts.
    linked_values = [n.value for i in linked if i < len(lines) for n in extract_numbers(lines[i])[0]]
    for index, line in enumerate(lines):
        numbers, skipped = extract_numbers(line)
        skipped_all.extend(skipped)
        for number in numbers:
            count += 1
            source = None
            for name in ("facts", "tool", "sourced", "cited"):
                if any(_matches(number, c) for c in getattr(allowed, name)):
                    source = name
                    break
            if source is None and (index in linked or any(_matches(number, v) for v in linked_values)):
                source = "linked"
            if source is None and not number.percent and number.scale == 1.0 and number.decimals == 0 and number.value <= 12:
                source = "trivial"
            if source is None:
                untraced.append(number.text)
            else:
                traced[source] += 1
    return Audit(numbers=count, traced=traced, untraced=untraced, trivial=skipped_all)


_ADVICE_TR = (
    r"\balmanız(?:ı)? (?:öneririm|tavsiye ederim)",
    r"\balmay[ıi]n\b",
    r"\bkaç[ıi]rmay[ıi]n\b",
    r"\b(?:iyi|kötü|harika|mükemmel|kârlı|karlı) bir yatırım(?:dır)?\b",
    r"\bfırsat(?:ı)? kaçırma",
    r"\bkesinlikle al",
    r"\bucuz(?:dur)?\b",
    r"\bpahalı(?:dır)?\b",
)
_ADVICE_EN = (r"\byou should (?:buy|not buy)\b", r"\b(?:good|bad|great) investment\b", r"\bdon't buy\b")


def advice_flags(answer: str) -> list[str]:
    """Phrases that read as a recommendation or a verdict rather than an analysis."""
    folded = unicodedata.normalize("NFC", answer.lower())
    # The refusal the prompt prescribes ("Alın ya da almayın diyemem") says "almayın" while
    # declining to say it; it is the opposite of advice and must not be flagged as advice.
    folded = re.sub(r"al[ıi]n ya da almay[ıi]n (?:diyemem|diyemeyiz|demem)", " ", folded)
    flags = [p for p in _ADVICE_TR + _ADVICE_EN if re.search(p, folded)]
    disclaimers = len(re.findall(r"yatırım tavsiyesi|investment advice", folded))
    if disclaimers > 1:
        flags.append(f"disclaimer_x{disclaimers}")
    return flags
