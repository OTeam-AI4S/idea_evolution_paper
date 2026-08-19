"""Purely local bibliography parsing and cited-reference selection.

This module identifies which bibliography entries are actually cited and
extracts stable identifiers. It does not invent summaries. Complete abstracts
are attached later by :mod:`reference_abstracts`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .section_splitter import SectionSplitter


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
ARXIV_RE = re.compile(
    r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/|10\.48550/arxiv\.)"
    r"([a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
LEADING_LABEL_RE = re.compile(
    r"^\s*(?:\[\s*(?P<bracket>\d+)\s*\]|(?P<plain>\d+)[.)])\s*"
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PMID_RE = re.compile(r"\bPMID\s*[: ]\s*(\d{5,10})\b", re.IGNORECASE)
REFERENCE_HEADER_RE = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?"
    r"(?:references?|bibliography|works\s+cited)\s*$"
)
NUMERIC_CITATION_RE = re.compile(r"\[([0-9,\s;–—-]+)\]")


@dataclass(frozen=True)
class ReferenceEntry:
    """One locally parsed bibliography entry."""

    raw: str
    number: Optional[int]
    identifier: str
    title: str
    author_year: Optional[Tuple[str, str]]
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    authors: str = ""
    year: str = ""
    venue: str = ""
    url: str = ""


def _clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clean_reference_text(text: str) -> str:
    """Normalize PDF typography without changing reference semantics."""
    text = (text or "").translate(str.maketrans({
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "\u00ad": "",
    }))
    # Join words hyphenated only because a PDF line ended mid-word.
    text = re.sub(r"(?<=[a-z])-\s+(?=[a-z])", "", text)
    return _clean_space(text)


def _strip_terminal_punctuation(text: str) -> str:
    return text.strip().rstrip(".,;:)]}'\"")


def _normalize_title(text: str) -> str:
    return re.sub(
        r"[^a-z0-9]+", " ", _clean_reference_text(text).lower()
    ).strip()


class LocalReferenceExtractor:
    """Extract structured references and their in-paper citation contexts."""

    def __init__(
        self,
        max_references: int = 50,
        max_contexts_per_reference: int = 2,
        max_summary_chars: int = 450,
        require_context: bool = True,
        context_sentences_before: int = 1,
        context_sentences_after: int = 1,
    ):
        self.max_references = max_references
        self.max_contexts_per_reference = max_contexts_per_reference
        self.max_summary_chars = max_summary_chars
        self.require_context = require_context
        self.context_sentences_before = max(
            0, min(2, int(context_sentences_before))
        )
        self.context_sentences_after = max(
            0, min(2, int(context_sentences_after))
        )
        self._section_patterns = [
            (name, [re.compile(pattern) for pattern in patterns])
            for name, patterns in SectionSplitter.DEFAULT_PATTERNS
        ]

    def extract(
        self,
        full_text: str,
        references_text: Optional[str] = None,
    ) -> List[dict]:
        """Compatibility API returning citation contexts as ``summary``."""
        candidates = self.extract_candidates(full_text, references_text)
        return [
            {
                "id": candidate["id"],
                "title": candidate["title"],
                "summary": candidate["citation_context"],
            }
            for candidate in candidates
        ]

    def extract_candidates(
        self,
        full_text: str,
        references_text: Optional[str] = None,
        structured_references: Optional[Sequence[object]] = None,
        sections: Optional[Dict[str, str]] = None,
    ) -> List[dict]:
        """Return cited bibliography candidates for identity resolution."""
        references_text = references_text or self._find_references_text(full_text)
        if not references_text:
            return []

        body_text = self._body_before_references(full_text, references_text)
        parsed_entries = [
            entry
            for raw in self.split_references(references_text)
            if (entry := self._parse_entry(raw)) is not None
        ]
        entries = self._merge_structured_entries(
            parsed_entries,
            structured_references or [],
        )
        sentence_records = self._sentence_records(body_text)
        section_boundaries = self._section_boundaries(body_text, sections)
        numeric_occurrences = self._numeric_occurrences(
            body_text,
            sentence_records,
            section_boundaries,
        )

        output: List[dict] = []
        seen_ids = set()
        seen_titles = set()

        for entry in entries:
            occurrences = (
                numeric_occurrences.get(entry.number, [])
                if entry.number is not None
                else []
            )
            if not occurrences and entry.author_year:
                occurrences = self._author_year_occurrences(
                    sentence_records,
                    section_boundaries,
                    *entry.author_year,
                )

            contexts = [
                self._clean_context(item["citation_sentence"])
                for item in occurrences
            ]
            summary = self._build_summary(contexts)
            if self.require_context and not summary:
                continue

            title_key = _normalize_title(entry.title)
            if entry.identifier in seen_ids or title_key in seen_titles:
                continue
            seen_ids.add(entry.identifier)
            seen_titles.add(title_key)

            output.append({
                "id": entry.identifier,
                "title": entry.title,
                "raw": entry.raw,
                "citation_context": summary,
                "citation_contexts": occurrences,
                "doi": entry.doi,
                "arxiv_id": entry.arxiv_id,
                "pmid": entry.pmid,
                "authors": entry.authors,
                "year": entry.year,
                "venue": entry.venue,
                "url": entry.url,
            })
            if len(output) >= self.max_references:
                break

        return output

    def _merge_structured_entries(
        self,
        parsed_entries: Sequence[ReferenceEntry],
        structured_references: Sequence[object],
    ) -> List[ReferenceEntry]:
        """Prefer BibTeX/JATS fields; retain text parsing only as fallback."""
        if not structured_references:
            return list(parsed_entries)

        parsed_by_number = {
            entry.number: entry
            for entry in parsed_entries
            if entry.number is not None
        }
        merged: List[ReferenceEntry] = []
        structured_numbers = set()
        for metadata in structured_references:
            number = getattr(metadata, "number", None)
            if number is None:
                continue
            structured_numbers.add(number)
            fallback = parsed_by_number.get(number)
            title = _clean_space(getattr(metadata, "title", "")) or (
                fallback.title if fallback else ""
            )
            raw = _clean_space(getattr(metadata, "raw", "")) or (
                fallback.raw if fallback else ""
            )
            if not title:
                continue

            arxiv_id = _clean_space(
                getattr(metadata, "arxiv_id", "")
            ) or (fallback.arxiv_id if fallback else None)
            doi = _clean_space(getattr(metadata, "doi", "")) or (
                fallback.doi if fallback else None
            )
            pmid = _clean_space(getattr(metadata, "pmid", "")) or (
                fallback.pmid if fallback else None
            )
            identifier = self._identifier(title, arxiv_id, doi, pmid)
            authors = _clean_space(getattr(metadata, "authors", ""))
            year = _clean_space(getattr(metadata, "year", ""))
            author_year = (
                self._extract_author_year(f"{authors} {year}")
                if authors and year else
                (fallback.author_year if fallback else None)
            )
            merged.append(ReferenceEntry(
                raw=raw,
                number=number,
                identifier=identifier,
                title=title,
                author_year=author_year,
                doi=doi or None,
                arxiv_id=arxiv_id or None,
                pmid=pmid or None,
                authors=authors,
                year=year,
                venue=_clean_space(getattr(metadata, "venue", "")),
                url=_clean_space(getattr(metadata, "url", "")),
            ))

        merged.extend(
            entry for entry in parsed_entries
            if entry.number not in structured_numbers
        )
        return sorted(
            merged,
            key=lambda entry: (
                entry.number is None,
                entry.number if entry.number is not None else 10**9,
            ),
        )

    @staticmethod
    def _identifier(
        title: str,
        arxiv_id: Optional[str],
        doi: Optional[str],
        pmid: Optional[str],
    ) -> str:
        if arxiv_id:
            return f"arxiv:{arxiv_id.lower()}"
        if doi:
            return f"doi:{_strip_terminal_punctuation(doi).lower()}"
        if pmid:
            return f"pmid:{pmid}"
        digest = hashlib.sha1(
            _normalize_title(title).encode("utf-8")
        ).hexdigest()[:12]
        return f"local:{digest}"

    def split_references(self, text: str) -> List[str]:
        """Split numeric or author-year bibliographies into entries."""
        cleaned = REFERENCE_HEADER_RE.sub("", text, count=1).strip()
        if not cleaned:
            return []

        marker = re.compile(
            r"(?m)^\s*(?:\[\s*\d+\s*\]|\d+[.)]\s+)"
        )
        starts = [match.start() for match in marker.finditer(cleaned)]
        if len(starts) >= 2:
            starts.append(len(cleaned))
            return [
                _clean_space(cleaned[starts[i]:starts[i + 1]])
                for i in range(len(starts) - 1)
                if len(_clean_space(cleaned[starts[i]:starts[i + 1]])) >= 20
            ]

        blocks = [
            _clean_space(block)
            for block in re.split(r"\n\s*\n+", cleaned)
            if len(_clean_space(block)) >= 20
        ]
        if len(blocks) >= 2:
            return blocks

        return self._split_author_year_lines(cleaned)

    def _find_references_text(self, full_text: str) -> str:
        matches = list(REFERENCE_HEADER_RE.finditer(full_text or ""))
        if not matches:
            return ""
        return full_text[matches[-1].end():].strip()

    def _body_before_references(
        self, full_text: str, references_text: str
    ) -> str:
        if not full_text:
            return ""
        anchor = references_text[:200].strip()
        if anchor:
            index = full_text.rfind(anchor)
            if index > 0:
                return full_text[:index]
        matches = list(REFERENCE_HEADER_RE.finditer(full_text))
        return full_text[:matches[-1].start()] if matches else full_text

    def _split_author_year_lines(self, text: str) -> List[str]:
        entries: List[str] = []
        current: List[str] = []
        for raw_line in text.splitlines():
            line = _clean_space(raw_line)
            if not line:
                continue
            looks_like_start = bool(
                re.match(r"^[A-Z][A-Za-z'’\-]+,\s*(?:[A-Z]\.|[A-Z][a-z]+)", line)
                and YEAR_RE.search(line)
            )
            if looks_like_start and current:
                entries.append(_clean_space(" ".join(current)))
                current = [line]
            else:
                current.append(line)
        if current:
            entries.append(_clean_space(" ".join(current)))
        return [entry for entry in entries if len(entry) >= 20]

    def _parse_entry(self, raw: str) -> Optional[ReferenceEntry]:
        raw = _clean_reference_text(raw)
        if len(raw) < 20:
            return None

        number = None
        label_match = LEADING_LABEL_RE.match(raw)
        if label_match:
            number = int(label_match.group("bracket") or label_match.group("plain"))
            content = raw[label_match.end():]
        else:
            content = raw

        title = self._extract_title(content)
        if not title:
            return None

        arxiv_match = ARXIV_RE.search(content)
        doi_match = DOI_RE.search(content)
        pmid_match = PMID_RE.search(content)
        arxiv_id = arxiv_match.group(1).lower() if arxiv_match else None
        doi = (
            _strip_terminal_punctuation(doi_match.group(0)).lower()
            if doi_match else None
        )
        pmid = pmid_match.group(1) if pmid_match else None
        identifier = self._identifier(title, arxiv_id, doi, pmid)

        author_year = self._extract_author_year(content)
        title_start = content.find(title)
        authors = (
            _strip_terminal_punctuation(content[:title_start])
            if title_start > 0 else ""
        )
        years = YEAR_RE.findall(content)
        year = years[-1][:4] if years else ""
        return ReferenceEntry(
            raw=raw,
            number=number,
            identifier=identifier,
            title=title,
            author_year=author_year,
            doi=doi,
            arxiv_id=arxiv_id,
            pmid=pmid,
            authors=authors,
            year=year,
        )

    def _extract_title(self, text: str) -> str:
        text = _clean_reference_text(URL_RE.sub("", text))
        quoted = re.search(
            r"[“\"]([^”\"]{12,300})[”\"]|[‘']([^’']{12,300})[’']",
            text,
        )
        if quoted:
            return _strip_terminal_punctuation(
                _clean_space(quoted.group(1) or quoted.group(2))
            )

        parts = [
            _strip_terminal_punctuation(part)
            for part in re.split(r"(?<=[.!?])\s+", text)
        ]
        scored: List[Tuple[float, str]] = []
        venue_words = re.compile(
            r"(?i)^(?:in\s+)?(?:proceedings|proc\.?|journal|transactions|"
            r"conference|workshop|symposium|vol\.?|volume|pp\.?|pages)\b"
        )

        for index, part in enumerate(parts):
            words = re.findall(r"[A-Za-z][A-Za-z0-9'’\-]*", part)
            if not 2 <= len(words) <= 35:
                continue
            if DOI_RE.search(part) or URL_RE.search(part):
                continue

            score = 3.0
            if 1 <= index < len(parts) - 1:
                score += 1.5
            if YEAR_RE.fullmatch(part.strip("() ")):
                score -= 5
            if venue_words.search(part) or re.match(r"(?i)^in\s+", part):
                score -= 8
            if re.match(r"(?i)^et\s+al\b", part):
                score -= 3
            if (
                re.search(r"(?i)\s+and\s+", part)
                and (
                    part.count(",") >= 1
                    or len(re.findall(r"\b[A-Z]\.", part)) >= 1
                    or (
                        index <= 1
                        and all(
                            token[:1].isupper()
                            for token in words
                            if token.lower() != "and"
                        )
                    )
                )
            ):
                score -= 6

            initial_tokens = sum(
                1 for token in part.split() if re.fullmatch(r"[A-Z]\.?", token)
            )
            if initial_tokens / max(len(part.split()), 1) > 0.25:
                score -= 4
            if part.count(",") >= 3:
                score -= 2
            if re.search(r"\b(?:19|20)\d{2}\b", part):
                score -= 1

            scored.append((score, part))

        if not scored:
            return ""
        best_score, best = max(scored, key=lambda item: (item[0], len(item[1])))
        if best_score <= 0:
            return ""
        best = re.sub(
            r"(?i)[,;]?\s*\(?\b(?:19|20)\d{2}[a-z]?\b\)?\s*$",
            "",
            best,
        )
        return _strip_terminal_punctuation(best)

    def _extract_author_year(self, text: str) -> Optional[Tuple[str, str]]:
        year_match = YEAR_RE.search(text)
        surname_match = re.search(r"\b([A-Z][A-Za-z'’\-]{2,})\b", text)
        if not year_match or not surname_match:
            return None
        return surname_match.group(1), year_match.group(0)[:4]

    def _sentences(self, body_text: str) -> List[str]:
        return [record["text"] for record in self._sentence_records(body_text)]

    def _sentence_records(self, body_text: str) -> List[dict]:
        """Return sentence text with source offsets for context windows."""
        if not body_text or not body_text.strip():
            return []
        placeholder = "\ue000"
        protected = re.sub(
            r"\b(et\s+al|e\.g|i\.e|Fig|Eq|Sec|Dr|Prof)\.",
            lambda match: match.group(0)[:-1] + placeholder,
            body_text,
            flags=re.IGNORECASE,
        )
        records: List[dict] = []
        start = 0
        ends = [
            match.end()
            for match in re.finditer(
                r"[.!?](?:[\"')\]]*)(?=\s+(?:[A-Z0-9(\[])|\s*$)",
                protected,
            )
        ]
        ends.append(len(body_text))
        for end in ends:
            raw = body_text[start:end]
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            source_start = start + leading
            source_end = start + trailing
            sentence = _clean_space(
                body_text[source_start:source_end].replace(placeholder, ".")
            )
            if 25 <= len(sentence) <= 1400:
                records.append({
                    "text": sentence,
                    "start": source_start,
                    "end": source_end,
                })
            start = end
        return records

    def _section_boundaries(
        self,
        body_text: str,
        sections: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[int, str]]:
        """Locate canonical section headings while preserving source offsets."""
        boundaries: List[Tuple[int, str]] = [(0, "unknown")]
        for name, section_text in (sections or {}).items():
            if name in {"abstract", "references", "full_text"}:
                continue
            section_text = (section_text or "").strip()
            if not section_text:
                continue
            position = body_text.find(section_text)
            if position >= 0:
                boundaries.append((position, name))
        offset = 0
        for raw_line in body_text.splitlines(keepends=True):
            line = raw_line.strip()
            if line and len(line) <= 100:
                for name, patterns in self._section_patterns:
                    if any(pattern.match(line) for pattern in patterns):
                        boundaries.append((offset, name))
                        break
            offset += len(raw_line)
        by_offset = {}
        for boundary, name in boundaries:
            by_offset[boundary] = name
        return sorted(by_offset.items())

    @staticmethod
    def _section_for_offset(
        boundaries: Sequence[Tuple[int, str]],
        offset: int,
    ) -> str:
        section = "unknown"
        for boundary, name in boundaries:
            if boundary > offset:
                break
            section = name
        return section

    def _context_occurrence(
        self,
        records: Sequence[dict],
        index: int,
        marker: str,
        marker_offset: int,
        boundaries: Sequence[Tuple[int, str]],
    ) -> Optional[dict]:
        citation_sentence = self._clean_context(
            records[index]["text"],
            keep_citations=True,
        )
        if not citation_sentence:
            return None
        before = [
            self._clean_context(records[position]["text"], keep_citations=True)
            for position in range(
                max(0, index - self.context_sentences_before), index
            )
        ]
        after = [
            self._clean_context(records[position]["text"], keep_citations=True)
            for position in range(
                index + 1,
                min(len(records), index + self.context_sentences_after + 1),
            )
        ]
        source_section = self._section_for_offset(
            boundaries, marker_offset
        )
        return {
            "source_section": source_section,
            "context_before": [value for value in before if value],
            "citation_sentence": citation_sentence,
            "context_after": [value for value in after if value],
        }

    def _numeric_occurrences(
        self,
        body_text: str,
        records: Sequence[dict],
        boundaries: Sequence[Tuple[int, str]],
    ) -> Dict[int, List[dict]]:
        occurrences: Dict[int, List[dict]] = {}
        for index, record in enumerate(records):
            source = body_text[record["start"]:record["end"]]
            for match in NUMERIC_CITATION_RE.finditer(source):
                marker_offset = record["start"] + match.start()
                occurrence = self._context_occurrence(
                    records,
                    index,
                    match.group(0),
                    marker_offset,
                    boundaries,
                )
                if not occurrence:
                    continue
                for number in set(self._expand_numeric_group(match.group(1))):
                    bucket = occurrences.setdefault(number, [])
                    if occurrence not in bucket and len(
                        bucket
                    ) < self.max_contexts_per_reference:
                        bucket.append(occurrence)
        return occurrences

    def _numeric_contexts(self, body_text: str) -> Dict[int, List[str]]:
        records = self._sentence_records(body_text)
        occurrences = self._numeric_occurrences(
            body_text,
            records,
            self._section_boundaries(body_text),
        )
        return {
            number: [
                self._clean_context(item["citation_sentence"])
                for item in values
            ]
            for number, values in occurrences.items()
        }

    def _expand_numeric_group(self, group: str) -> Iterable[int]:
        for part in re.split(r"[,;]\s*", group):
            part = part.strip()
            range_match = re.fullmatch(r"(\d+)\s*[–—-]\s*(\d+)", part)
            if range_match:
                start, end = map(int, range_match.groups())
                if 0 <= end - start <= 30:
                    yield from range(start, end + 1)
            elif part.isdigit():
                yield int(part)

    def _author_year_contexts(
        self, sentences: Sequence[str], surname: str, year: str
    ) -> List[str]:
        pattern = re.compile(
            rf"\b{re.escape(surname)}\b.*?\b{re.escape(year)}\b|"
            rf"\b{re.escape(year)}\b.*?\b{re.escape(surname)}\b",
            re.IGNORECASE,
        )
        contexts = []
        for sentence in sentences:
            if pattern.search(sentence):
                cleaned = self._clean_context(sentence)
                if cleaned and cleaned not in contexts:
                    contexts.append(cleaned)
        return contexts

    def _author_year_occurrences(
        self,
        records: Sequence[dict],
        boundaries: Sequence[Tuple[int, str]],
        surname: str,
        year: str,
    ) -> List[dict]:
        pattern = re.compile(
            rf"\b{re.escape(surname)}\b.*?\b{re.escape(year)}\b|"
            rf"\b{re.escape(year)}\b.*?\b{re.escape(surname)}\b",
            re.IGNORECASE,
        )
        occurrences = []
        for index, record in enumerate(records):
            match = pattern.search(record["text"])
            if not match:
                continue
            occurrence = self._context_occurrence(
                records,
                index,
                match.group(0),
                record["start"] + match.start(),
                boundaries,
            )
            if occurrence and occurrence not in occurrences:
                occurrences.append(occurrence)
            if len(occurrences) >= self.max_contexts_per_reference:
                break
        return occurrences

    def _clean_context(self, sentence: str, keep_citations: bool = False) -> str:
        if not keep_citations:
            sentence = NUMERIC_CITATION_RE.sub("", sentence)
        sentence = URL_RE.sub("", sentence)
        sentence = re.sub(r"\[equation\](?:\s*#?\d+\w*)*", " ", sentence)
        sentence = _clean_space(sentence).strip()

        # PDF front matter is sometimes joined to the first Introduction
        # sentence.  Drop the preceding project-page/keyword/title material.
        heading = re.compile(
            r"(?i)(?:^|\s)(?:\d+(?:\.\d+)*)\s+"
            r"(?:introduction|related\s+work|background|methodology?|"
            r"approach|framework|experiments?|evaluation)\s+"
        )
        matches = list(heading.finditer(sentence[:400]))
        if matches:
            sentence = sentence[matches[-1].end():]

        sentence = re.sub(
            r"(?i)^.*?\bkeywords?\s*:\s*",
            "",
            sentence,
        )
        sentence = re.sub(r"\s+([,.;:!?])", r"\1", _clean_space(sentence))
        sentence = sentence.strip(" -–—")

        numbers = re.findall(
            r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?",
            sentence,
        )
        words = re.findall(r"[A-Za-z]+", sentence)
        metrics = re.findall(
            r"(?i)\b(?:AP|AP50|AP75|FLOPs?|top-?[15]|mAP|AUC|F1|params?)\b",
            sentence,
        )
        looks_like_table = (
            len(numbers) >= 10
            and len(numbers) >= max(5, len(words) // 3)
        ) or len(metrics) >= 5
        if looks_like_table:
            return ""
        if re.search(r"(?i)\btable\s+\d+\s*:", sentence) and len(sentence) > 180:
            return ""
        if "\x00" in sentence or re.search(r"(?:#\d+|\bc\|)", sentence):
            return ""
        if re.search(r"(?i)\bfig(?:ure)?\.?\s*\d+\s*:", sentence):
            return ""
        if re.search(r"(?i)\[t\]\s*input\s*:\s*output\s*:", sentence):
            return ""
        return sentence

    def _build_summary(self, contexts: Sequence[str]) -> str:
        selected: List[str] = []
        for context in contexts:
            if context and context not in selected:
                selected.append(context)
            if len(selected) >= self.max_contexts_per_reference:
                break
        summary = _clean_space(" ".join(selected))
        if len(summary) <= self.max_summary_chars:
            return summary
        shortened = summary[:self.max_summary_chars].rsplit(" ", 1)[0].rstrip()
        return shortened + "…"
