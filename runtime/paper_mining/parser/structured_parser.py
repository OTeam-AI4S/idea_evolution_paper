"""Local parsers for arXiv source bundles and PMC JATS XML."""

from __future__ import annotations

import gzip
import html
import io
import logging
import posixpath
import re
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class StructuredDocument:
    """Plain text plus sections recovered from a structured full-text source."""

    full_text: str
    sections: Dict[str, str] = field(default_factory=dict)
    references: List["StructuredReference"] = field(default_factory=list)


@dataclass
class StructuredReference:
    """Bibliography fields preserved before any plain-text rendering."""

    number: int
    key: str
    raw: str
    title: str = ""
    authors: str = ""
    year: str = ""
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    pmid: str = ""
    url: str = ""


def _clean_space(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _canonical_section(title: str) -> Optional[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    if not normalized:
        return None
    if normalized == "abstract":
        return "abstract"
    if "reference" in normalized or "bibliograph" in normalized:
        return "references"
    method_terms = (
        "method", "methodology", "approach", "model", "framework",
        "architecture", "algorithm", "materials and methods",
        "experimental setup", "implementation", "training",
    )
    if any(term in normalized for term in method_terms):
        return "method"
    if "introduction" in normalized:
        return "introduction"
    if "result" in normalized or "evaluation" in normalized:
        return "experiments"
    if "discussion" in normalized:
        return "discussion"
    if "limitation" in normalized:
        return "limitations"
    if "conclusion" in normalized or "future work" in normalized:
        return "conclusion"
    if "related work" in normalized or "background" in normalized:
        return "related_work"
    return None


class StructuredFullTextParser:
    """Parse structured publisher sources without heavyweight ML dependencies."""

    _MAX_MEMBER_BYTES = 20 * 1024 * 1024
    _MAX_TOTAL_BYTES = 100 * 1024 * 1024

    def parse_arxiv_source(self, path: str) -> StructuredDocument:
        return self.parse_arxiv_bytes(Path(path).read_bytes())

    def parse_arxiv_bytes(self, payload: bytes) -> StructuredDocument:
        files = self._read_source_bundle(payload)
        tex_files = {
            name: value
            for name, value in files.items()
            if name.lower().endswith((".tex", ".ltx"))
        }
        if not tex_files:
            raise ValueError("arXiv source bundle contains no TeX file")

        main_name = self._choose_main_tex(tex_files)
        source = self._inline_tex(main_name, tex_files)
        bib_files = {
            name: value
            for name, value in files.items()
            if name.lower().endswith(".bib")
        }
        return self._parse_tex(source, bib_files)

    def parse_pmc_xml(self, path: str) -> StructuredDocument:
        return self.parse_pmc_bytes(Path(path).read_bytes())

    def parse_pmc_bytes(self, payload: bytes) -> StructuredDocument:
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise ValueError(f"invalid PMC XML: {exc}") from exc

        abstract_parts = [
            self._xml_text(node)
            for node in root.iter()
            if _local_name(node.tag) == "abstract"
        ]
        abstract = _clean_space(" ".join(abstract_parts))

        sections: Dict[str, str] = {}
        body = next(
            (node for node in root.iter() if _local_name(node.tag) == "body"),
            None,
        )
        body_paragraphs: List[str] = []
        if body is not None:
            body_paragraphs = [
                _clean_space(self._xml_text(node))
                for node in body.iter()
                if _local_name(node.tag) == "p"
            ]

            def collect_section(
                section: ET.Element,
                inherited: Optional[str] = None,
            ):
                title_node = next(
                    (
                        child for child in list(section)
                        if _local_name(child.tag) == "title"
                    ),
                    None,
                )
                canonical = (
                    _canonical_section(self._xml_text(title_node)) or inherited
                )
                direct_paragraphs = [
                    _clean_space(self._xml_text(child))
                    for child in list(section)
                    if _local_name(child.tag) in {"p", "list", "disp-formula"}
                ]
                text = _clean_space(" ".join(direct_paragraphs))
                if canonical and text:
                    sections[canonical] = _clean_space(
                        f"{sections.get(canonical, '')} {text}"
                    )
                for child in list(section):
                    if _local_name(child.tag) == "sec":
                        collect_section(child, canonical)

            for section in (
                child for child in list(body) if _local_name(child.tag) == "sec"
            ):
                collect_section(section)

        if abstract:
            sections["abstract"] = abstract

        references, structured_references = self._pmc_references(root)
        full_parts = []
        if abstract:
            full_parts.extend(["Abstract", abstract])
        if body_paragraphs:
            full_parts.append("\n".join(body_paragraphs))
        if references:
            full_parts.extend(["References", "\n".join(references)])

        full_text = "\n".join(full_parts).strip()
        if len(full_text) < 200:
            raise ValueError("PMC XML contains too little article text")
        return StructuredDocument(
            full_text=full_text,
            sections=sections,
            references=structured_references,
        )

    def _read_source_bundle(self, payload: bytes) -> Dict[str, str]:
        files: Dict[str, str] = {}
        try:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
                total = 0
                for member in archive.getmembers():
                    if not member.isfile() or member.issym() or member.islnk():
                        continue
                    normalized = posixpath.normpath(member.name).lstrip("/")
                    if normalized.startswith("../") or member.size > self._MAX_MEMBER_BYTES:
                        continue
                    total += member.size
                    if total > self._MAX_TOTAL_BYTES:
                        raise ValueError("arXiv source bundle is too large")
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    files[normalized] = self._decode(extracted.read())
        except tarfile.ReadError:
            raw = payload
            if payload.startswith(b"\x1f\x8b"):
                try:
                    raw = gzip.decompress(payload)
                except OSError as exc:
                    raise ValueError("invalid arXiv gzip source") from exc
            files["main.tex"] = self._decode(raw)
        return files

    @staticmethod
    def _decode(payload: bytes) -> str:
        for encoding in ("utf-8", "latin-1"):
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                continue
        return payload.decode("utf-8", errors="replace")

    @staticmethod
    def _choose_main_tex(files: Dict[str, str]) -> str:
        def score(item):
            name, text = item
            value = len(text)
            if re.search(r"\\documentclass\b", text):
                value += 2_000_000
            if re.search(r"\\begin\s*{document}", text):
                value += 1_000_000
            if posixpath.basename(name).lower() in {
                "main.tex", "paper.tex", "manuscript.tex"
            }:
                value += 500_000
            return value

        return max(files.items(), key=score)[0]

    def _inline_tex(
        self,
        name: str,
        files: Dict[str, str],
        seen: Optional[set] = None,
        depth: int = 0,
    ) -> str:
        if depth > 12:
            return ""
        seen = seen or set()
        normalized = posixpath.normpath(name)
        if normalized in seen or normalized not in files:
            return ""
        seen.add(normalized)
        source = files[normalized]
        base = posixpath.dirname(normalized)

        def replace(match):
            child = match.group(1).strip()
            if not posixpath.splitext(child)[1]:
                child += ".tex"
            child_name = posixpath.normpath(posixpath.join(base, child))
            return self._inline_tex(child_name, files, seen, depth + 1)

        return re.sub(
            r"\\(?:input|include)\s*{([^}]+)}",
            replace,
            source,
        )

    def _parse_tex(
        self,
        source: str,
        bib_files: Dict[str, str],
    ) -> StructuredDocument:
        source = self._strip_comments(source)
        document_match = re.search(
            r"\\begin\s*{document}(.*?)\\end\s*{document}",
            source,
            flags=re.DOTALL,
        )
        if document_match:
            source = document_match.group(1)

        references, key_to_number, structured_references = (
            self._tex_references(source, bib_files)
        )
        source = self._replace_citations(source, key_to_number)

        abstract_match = re.search(
            r"\\begin\s*{abstract}(.*?)\\end\s*{abstract}",
            source,
            flags=re.DOTALL | re.IGNORECASE,
        )
        abstract = (
            self._latex_to_text(abstract_match.group(1))
            if abstract_match else ""
        )

        section_matches = list(re.finditer(
            r"\\(section|subsection|subsubsection)\*?\s*{([^{}]+)}",
            source,
            flags=re.IGNORECASE,
        ))
        sections: Dict[str, str] = {}
        full_parts: List[str] = []
        if abstract:
            sections["abstract"] = abstract
            full_parts.extend(["Abstract", abstract])

        for index, match in enumerate(section_matches):
            title = self._latex_to_text(match.group(2))
            end = (
                section_matches[index + 1].start()
                if index + 1 < len(section_matches)
                else len(source)
            )
            text = self._latex_to_text(source[match.end():end])
            if not text:
                continue
            full_parts.extend([title, text])
            canonical = _canonical_section(title)
            if canonical:
                sections[canonical] = _clean_space(
                    f"{sections.get(canonical, '')} {text}"
                )

        if not section_matches:
            body = self._latex_to_text(source)
            if body:
                full_parts.append(body)
        if references:
            full_parts.extend(["References", "\n".join(references)])

        full_text = "\n".join(full_parts).strip()
        if len(full_text) < 200:
            raise ValueError("arXiv source contains too little article text")
        return StructuredDocument(
            full_text=full_text,
            sections=sections,
            references=structured_references,
        )

    @staticmethod
    def _strip_comments(source: str) -> str:
        return re.sub(r"(?m)(?<!\\)%.*$", "", source)

    def _tex_references(
        self,
        source: str,
        bib_files: Dict[str, str],
    ) -> tuple[List[str], Dict[str, int], List[StructuredReference]]:
        entries: List[tuple[str, str, dict]] = []
        bibliography = re.search(
            r"\\begin\s*{thebibliography}.*?"
            r"(.*?)\\end\s*{thebibliography}",
            source,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if bibliography:
            chunks = re.split(
                r"\\bibitem(?:\s*\[[^\]]*\])?\s*{([^}]+)}",
                bibliography.group(1),
            )
            entries.extend(
                (
                    chunks[index].strip(),
                    self._latex_to_text(chunks[index + 1]),
                    {},
                )
                for index in range(1, len(chunks) - 1, 2)
            )

        requested = set()
        for match in re.finditer(r"\\bibliography\s*{([^}]+)}", source):
            requested.update(
                posixpath.splitext(posixpath.basename(part.strip()))[0]
                for part in match.group(1).split(",")
            )
        for name, bib_source in bib_files.items():
            stem = posixpath.splitext(posixpath.basename(name))[0]
            if requested and stem not in requested:
                continue
            entries.extend(self._parse_bibtex(bib_source))

        deduplicated: List[tuple[str, str, dict]] = []
        seen = set()
        for key, text, metadata in entries:
            if not key or not text or key in seen:
                continue
            seen.add(key)
            deduplicated.append((key, text, metadata))

        key_to_number = {
            key: index
            for index, (key, _, _) in enumerate(deduplicated, 1)
        }
        references = [
            f"[{index}] {text}"
            for index, (_, text, _) in enumerate(deduplicated, 1)
        ]
        structured = [
            StructuredReference(
                number=index,
                key=key,
                raw=text,
                **metadata,
            )
            for index, (key, text, metadata) in enumerate(deduplicated, 1)
        ]
        return references, key_to_number, structured

    def _parse_bibtex(self, source: str) -> List[tuple[str, str, dict]]:
        entries: List[tuple[str, str, dict]] = []
        start_pattern = re.compile(r"@\w+\s*([{(])\s*([^,\s]+)\s*,")
        for match in start_pattern.finditer(source):
            start = match.end()
            opening = match.group(1)
            closing = "}" if opening == "{" else ")"
            depth = 1
            index = start
            while index < len(source) and depth:
                if source[index] == opening:
                    depth += 1
                elif source[index] == closing:
                    depth -= 1
                index += 1
            block = source[start:index - 1]
            fields = {}
            for field in re.finditer(
                r"(?is)(\w+)\s*=\s*(?:{((?:[^{}]|{[^{}]*})*)}|"
                r'"([^"]*)")\s*,?',
                block,
            ):
                fields[field.group(1).lower()] = _clean_space(
                    field.group(2) or field.group(3)
                )
            title = self._latex_to_text(fields.get("title", ""))
            if not title:
                continue
            author = self._latex_to_text(fields.get("author", ""))
            year = self._latex_to_text(fields.get("year", ""))
            venue = self._latex_to_text(
                fields.get("journal") or fields.get("booktitle") or ""
            )
            doi = self._latex_to_text(fields.get("doi", ""))
            eprint = self._latex_to_text(fields.get("eprint", ""))
            archive = self._latex_to_text(fields.get("archiveprefix", ""))
            pmid = self._latex_to_text(fields.get("pmid", ""))
            url = self._latex_to_text(fields.get("url", ""))
            arxiv_id = ""
            if eprint and (archive.lower() == "arxiv" or re.fullmatch(
                r"\d{4}\.\d{4,5}(?:v\d+)?", eprint
            )):
                arxiv_id = eprint
                eprint = f"arXiv:{eprint}"
            if pmid:
                pmid = f"PMID:{pmid}"
            text = ". ".join(
                part for part in (
                    author, title, venue, year, doi, eprint, pmid, url
                ) if part
            )
            entries.append((
                match.group(2),
                text,
                {
                    "title": title,
                    "authors": author,
                    "year": year,
                    "venue": venue,
                    "doi": doi,
                    "arxiv_id": arxiv_id,
                    "pmid": self._latex_to_text(fields.get("pmid", "")),
                    "url": url,
                },
            ))
        return entries

    @staticmethod
    def _replace_citations(source: str, key_to_number: Dict[str, int]) -> str:
        def replace(match):
            numbers = [
                key_to_number[key.strip()]
                for key in match.group(1).split(",")
                if key.strip() in key_to_number
            ]
            return f"[{','.join(map(str, numbers))}]" if numbers else ""

        return re.sub(
            r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*{([^}]+)}",
            replace,
            source,
        )

    def _latex_to_text(self, source: str) -> str:
        text = source
        text = re.sub(
            r"\\begin\s*{((?:figure|table|equation|align|tikzpicture)\*?)}.*?"
            r"\\end\s*{\1}",
            " ",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"\$\$.*?\$\$|\$[^$]*\$", " [equation] ", text, flags=re.DOTALL)
        text = re.sub(r"\\(?:label|ref|eqref|pageref)\s*{[^}]*}", " ", text)
        text = re.sub(r"\\(?:begin|end)\s*{[^}]+}", "\n", text)
        text = re.sub(
            r"\\(?:textbf|textit|emph|mathrm|mathbf|textrm|texttt|underline)"
            r"\s*{([^{}]*)}",
            r"\1",
            text,
        )
        text = re.sub(r"\\(?:url|href)\s*{([^{}]*)}(?:\s*{([^{}]*)})?", r" \2 ", text)
        text = re.sub(r"\\[a-zA-Z@]+\*?(?:\s*\[[^\]]*\])?", " ", text)
        text = text.replace("{", " ").replace("}", " ")
        replacements = {
            r"\&": "&", r"\%": "%", r"\_": "_", r"\#": "#",
            r"\textasciitilde": "~", "~": " ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return _clean_space(html.unescape(text))

    def _xml_text(self, node: Optional[ET.Element]) -> str:
        if node is None:
            return ""
        parts: List[str] = []
        if node.text:
            parts.append(node.text)
        for child in list(node):
            child_text = self._xml_text(child)
            if _local_name(child.tag) == "xref" and child.attrib.get("ref-type") == "bibr":
                child_text = f"[{_clean_space(child_text)}]"
            parts.append(child_text)
            if child.tail:
                parts.append(child.tail)
        return _clean_space(" ".join(parts))

    def _pmc_references(
        self,
        root: ET.Element,
    ) -> tuple[List[str], List[StructuredReference]]:
        references = []
        structured = []
        for index, ref in enumerate(
            (node for node in root.iter() if _local_name(node.tag) == "ref"),
            1,
        ):
            label = next(
                (
                    self._xml_text(child) for child in list(ref)
                    if _local_name(child.tag) == "label"
                ),
                str(index),
            )
            citation = next(
                (
                    child for child in ref.iter()
                    if _local_name(child.tag) in {
                        "element-citation", "mixed-citation", "nlm-citation"
                    }
                ),
                ref,
            )
            raw = self._xml_text(citation)
            identifiers = []
            identifier_values = {}
            for node in citation.iter():
                if _local_name(node.tag) != "pub-id":
                    continue
                value = _clean_space(self._xml_text(node))
                kind = (
                    node.attrib.get("pub-id-type")
                    or node.attrib.get("id-type")
                    or ""
                ).lower()
                if value and kind in {"doi", "pmid", "pmcid"}:
                    identifiers.append(f"{kind.upper()}:{value}")
                    identifier_values[kind] = value
            if identifiers:
                raw = _clean_space(f"{raw} {' '.join(identifiers)}")
            if raw:
                references.append(f"[{_clean_space(label) or index}] {raw}")
                title_node = next(
                    (
                        node for node in citation.iter()
                        if _local_name(node.tag) == "article-title"
                    ),
                    None,
                )
                source_node = next(
                    (
                        node for node in citation.iter()
                        if _local_name(node.tag) == "source"
                    ),
                    None,
                )
                year_node = next(
                    (
                        node for node in citation.iter()
                        if _local_name(node.tag) == "year"
                    ),
                    None,
                )
                surnames = [
                    self._xml_text(node)
                    for node in citation.iter()
                    if _local_name(node.tag) == "surname"
                ]
                structured.append(StructuredReference(
                    number=(
                        int(label) if str(label).strip().isdigit() else index
                    ),
                    key=ref.attrib.get("id", str(index)),
                    raw=raw,
                    title=self._xml_text(title_node),
                    authors=" and ".join(filter(None, surnames)),
                    year=self._xml_text(year_node),
                    venue=self._xml_text(source_node),
                    doi=identifier_values.get("doi", ""),
                    pmid=identifier_values.get("pmid", ""),
                ))
        return references, structured
